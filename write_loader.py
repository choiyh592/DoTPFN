import os

def write_file(path, content):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Wrote {path}")

loader_content = '''import logging
import numpy as np

logger = logging.getLogger("DoTPFN.TabPFN")

# Registry of supported TabPFN backends
TABPFN_BACKENDS = {
    "tabpfn_extensions": {
        "description": "TabPFN Extensions (PriorLabs extended API with embeddings)",
        "classifier_import": ("tabpfn_extensions", "TabPFNClassifier"),
        "embedding_import": ("tabpfn_extensions.embedding", "TabPFNEmbedding"),
    },
    "tabpfn": {
        "description": "TabPFN v2 (PriorLabs standard package)",
        "classifier_import": ("tabpfn", "TabPFNClassifier"),
        "embedding_import": ("tabpfn.embedding", "TabPFNEmbedding"),
    },
    "tabpfn_v1": {
        "description": "TabPFN v1 (original Hollmann et al. 2022)",
        "classifier_import": ("tabpfn", "TabPFNClassifier"),
        "embedding_import": None,  # v1 has no embedding API
    },
}

def get_tabpfn_classes(version="auto"):
    if version == "auto":
        for backend_name in ["tabpfn_extensions", "tabpfn"]:
            result = _try_load_backend(backend_name)
            if result is not None:
                return result
        logger.warning(
            "No TabPFN backend found. Install tabpfn_extensions or tabpfn. "
            "Falling back to dummy identity embeddings."
        )
        return _make_dummy_classes()

    if version not in TABPFN_BACKENDS:
        raise ValueError(
            f"Unknown TabPFN version '{version}'. Supported: {list(TABPFN_BACKENDS.keys()) + ['auto']}"
        )

    result = _try_load_backend(version)
    if result is not None:
        return result

    raise ImportError(
        f"TabPFN backend '{version}' could not be imported. Please install the corresponding package."
    )

def _try_load_backend(backend_name):
    backend = TABPFN_BACKENDS[backend_name]
    try:
        import importlib
        clf_module = importlib.import_module(backend["classifier_import"][0])
        TabPFNClassifier = getattr(clf_module, backend["classifier_import"][1])

        if backend["embedding_import"] is not None:
            emb_module = importlib.import_module(backend["embedding_import"][0])
            TabPFNEmbedding = getattr(emb_module, backend["embedding_import"][1])
        else:
            logger.info(
                f"Backend '{backend_name}' has no native embedding API. Using predict_proba-based fallback."
            )
            TabPFNEmbedding = _make_v1_embedding_wrapper(TabPFNClassifier)

        logger.info(f"Loaded TabPFN backend: {backend['description']}")
        return TabPFNClassifier, TabPFNEmbedding
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not load backend '{backend_name}': {e}")
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
        def __init__(self, n_estimators=1, device="cpu"):
            pass
        def fit(self, X, y):
            pass
        def predict_proba(self, X):
            return np.zeros((len(X), 2), dtype=np.float32)

    class DummyTabPFNEmbedding:
        def __init__(self, tabpfn_clf=None, n_fold=5):
            pass
        def fit(self, X, y):
            pass
        def get_embeddings(self, X_train, y_train, X_val, data_source="train"):
            X_in = X_train if data_source == "train" else X_val
            n_features = X_in.shape[1] if hasattr(X_in, 'shape') else len(X_in[0])
            d_out = max(256, n_features)
            X_out = np.zeros((1, len(X_in), d_out), dtype=np.float32)
            X_out[0, :, :n_features] = X_in[:, :n_features]
            return X_out

    return DummyTabPFNClassifier, DummyTabPFNEmbedding
'''
write_file('src/dotpfn/utils/tabpfn_loader.py', loader_content)
