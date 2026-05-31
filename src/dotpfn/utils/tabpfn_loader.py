import logging
import numpy as np

logger = logging.getLogger("DoTPFN.TabPFN")

TABPFN_BACKENDS = {
    "tabpfn_extensions": {
        "description": "TabPFN Extensions (tabpfn_extensions.TabPFNClassifier + TabPFNEmbedding)",
        "classifier_import": ("tabpfn_extensions", "TabPFNClassifier"),
        "embedding_import": ("tabpfn_extensions.embedding", "TabPFNEmbedding"),
    },
    "tabpfn": {
        "description": "TabPFN base package (tabpfn.TabPFNClassifier + tabpfn_extensions.embedding)",
        "classifier_import": ("tabpfn", "TabPFNClassifier"),
        "embedding_import": ("tabpfn_extensions.embedding", "TabPFNEmbedding"),
    },
    "tabpfn_v1": {
        "description": "TabPFN v1 (original Hollmann et al. 2022, no native embedding API)",
        "classifier_import": ("tabpfn", "TabPFNClassifier"),
        "embedding_import": None,
    },
}

def get_tabpfn_classes(version="auto"):
    """Dynamically load TabPFN classifier and embedding classes.

    Args:
        version: Which TabPFN backend to use.
            - "auto": Try tabpfn_extensions -> tabpfn -> dummy fallback
            - "tabpfn_extensions": Use tabpfn_extensions.TabPFNClassifier (default for tabpfn<=7.x)
            - "tabpfn": Use tabpfn.TabPFNClassifier with V2.6 factory if available (tabpfn>=8.x)
            - "tabpfn_v1": Original TabPFN v1 (predict_proba-based embedding fallback)
    """
    if version == "auto":
        for backend_name in ["tabpfn_extensions", "tabpfn"]:
            result = _try_load_backend(backend_name)
            if result is not None:
                return result
        logger.warning("No TabPFN backend found. Falling back to dummy embeddings.")
        return _make_dummy_classes()

    if version not in TABPFN_BACKENDS:
        raise ValueError(f"Unknown TabPFN version \'{version}\'. Supported: {list(TABPFN_BACKENDS.keys()) + ['auto']}")

    result = _try_load_backend(version)
    if result is not None:
        return result

    raise ImportError(f"TabPFN backend \'{version}\' could not be imported.")

def _try_load_backend(backend_name):
    backend = TABPFN_BACKENDS[backend_name]
    try:
        import importlib
        clf_module = importlib.import_module(backend["classifier_import"][0])
        RawTabPFNClassifier = getattr(clf_module, backend["classifier_import"][1])

        # For the "tabpfn" backend (base package), try the V2.6+ factory API
        # (available in tabpfn>=8.0). If it does not exist (tabpfn<=7.x),
        # fall back to using the constructor directly (V2.6 is already the default).
        if backend_name == "tabpfn":
            try:
                from tabpfn.constants import ModelVersion
                class V26Factory:
                    def __new__(cls, n_estimators=1, device="cpu", model_path=None, **kwargs):
                        if model_path:
                            return RawTabPFNClassifier(n_estimators=n_estimators, device=device, model_path=model_path)
                        return RawTabPFNClassifier.create_default_for_version(ModelVersion.V2_6, device=device)
                TabPFNClassifier = V26Factory
                logger.info("Using TabPFN V2.6 factory API (tabpfn>=8.0)")
            except (ImportError, AttributeError):
                # tabpfn<=7.x: V2.6 is already the default, just use constructor directly
                TabPFNClassifier = RawTabPFNClassifier
                logger.info("Using TabPFN default constructor (tabpfn<=7.x, default=V2.6)")
        else:
            TabPFNClassifier = RawTabPFNClassifier

        if backend["embedding_import"] is not None:
            emb_module = importlib.import_module(backend["embedding_import"][0])
            TabPFNEmbedding = getattr(emb_module, backend["embedding_import"][1])
        else:
            logger.info(f"Backend \'{backend_name}\' has no native embedding API. Using predict_proba fallback.")
            TabPFNEmbedding = _make_v1_embedding_wrapper(TabPFNClassifier)

        logger.info(f"Loaded TabPFN backend: {backend['description']}")
        return TabPFNClassifier, TabPFNEmbedding
    except Exception as e:
        logger.error(f"Could not load backend \'{backend_name}\': {type(e).__name__} - {e}")
        return None

def _make_v1_embedding_wrapper(TabPFNClassifier):
    class TabPFNv1EmbeddingWrapper:
        def __init__(self, tabpfn_clf=None, n_fold=5):
            self.n_fold = n_fold
            self.clf = tabpfn_clf
        def fit(self, X, y):
            if self.clf is None:
                self.clf = TabPFNClassifier(device="cpu")
            self.clf.fit(X, y)
        def get_embeddings(self, X_train, y_train, X_val, data_source="train"):
            X_target = X_train if data_source == "train" else X_val
            self.clf.fit(X_train, y_train)
            proba = self.clf.predict_proba(X_target)
            return np.expand_dims(proba, axis=0).astype(np.float32)
    return TabPFNv1EmbeddingWrapper

def _make_dummy_classes():
    class DummyTabPFNClassifier:
        def __init__(self, n_estimators=1, device="cpu", **kwargs): pass
        def fit(self, X, y): pass
        def predict_proba(self, X): return np.zeros((len(X), 2), dtype=np.float32)

    class DummyTabPFNEmbedding:
        def __init__(self, tabpfn_clf=None, n_fold=5): pass
        def fit(self, X, y): pass
        def get_embeddings(self, X_train, y_train, X_val, data_source="train"):
            X_in = X_train if data_source == "train" else X_val
            n_features = X_in.shape[1] if hasattr(X_in, 'shape') else len(X_in[0])
            d_out = max(256, n_features)
            X_out = np.zeros((1, len(X_in), d_out), dtype=np.float32)
            X_out[0, :, :n_features] = X_in[:, :n_features]
            return X_out
    return DummyTabPFNClassifier, DummyTabPFNEmbedding
