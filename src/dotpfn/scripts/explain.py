import os
import cv2
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from PIL import Image
import torch.nn as nn
from dotpfn.utils.logging import setup_logger
from dotpfn.utils.config import ConfigNode
from dotpfn.data.dataset import load_all_document_embeddings, get_stratified_kfold_splits
from dotpfn.models.fusion import DoTPFN

# Dyn-grid size colpali dependencies
try:
    from colpali_engine.models import ColQwen2Processor
except ImportError:
    class ColQwen2Processor:
        @staticmethod
        def from_pretrained(path):
            class DummyProcessor:
                def process_images(self, images):
                    # Dummy object having grid property
                    class DummyGrid:
                        def __init__(self):
                            self.image_grid_thw = [[0, 36, 27]] # Mock LLM Grid
                    return DummyGrid()
            return DummyProcessor()

warnings.filterwarnings('ignore')
logger = setup_logger("DoTPFN.Explain")

# Dyn-imports helper
def get_tabpfn_classes():
    try:
        from tabpfn_extensions import TabPFNClassifier
        from tabpfn_extensions.embedding import TabPFNEmbedding
        return TabPFNClassifier, TabPFNEmbedding
    except ImportError:
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.embedding import TabPFNEmbedding
            return TabPFNClassifier, TabPFNEmbedding
        except ImportError:
            logger.warning("Neither tabpfn_extensions nor tabpfn could be imported. Explainer running on mock embeddings.")
            class DummyTabPFNClassifier:
                def __init__(self, n_estimators=1, device="cpu"): pass
                def fit(self, X, y): pass
            class DummyTabPFNEmbedding:
                def __init__(self, tabpfn_clf, n_fold=5): pass
                def fit(self, X, y): pass
                def get_embeddings(self, X_train, y_train, X_val, data_source="train"):
                    return np.zeros((1, len(X_val), 256), dtype=np.float32)
            return DummyTabPFNClassifier, DummyTabPFNEmbedding


# =====================================================================
# Hierarchical Explainer Pipeline Core
# =====================================================================
class HierarchicalMultimodalExplainer:
    def __init__(self, model: nn.Module, feature_names: list, colqwen_model_path: str = "vidore/colqwen2-v0.1", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.feature_names = feature_names + ["Clinical_Document"]
        self.processor = ColQwen2Processor.from_pretrained(colqwen_model_path)
        
        self.supp_tab = None
        self.supp_doc = None
        self.supp_y = None
        self.baseline_doc = None
        
        self.X_train_raw = None
        self.y_train_raw = None
        self.tab_embedder = None
        
        self.TabPFNClassifier, self.TabPFNEmbedding = get_tabpfn_classes()

    def setup_support_set(self, X_train_raw, y_train_raw, supp_tab_emb, supp_doc_emb):
        logger.info("Setting up Explainer Support Set and Baselines...")
        self.X_train_raw = X_train_raw
        self.y_train_raw = y_train_raw
        
        clf = self.TabPFNClassifier(n_estimators=1, device=self.device)
        clf.fit(X_train_raw, y_train_raw)
        self.tab_embedder = self.TabPFNEmbedding(tabpfn_clf=clf, n_fold=5)
        self.tab_embedder.fit(X_train_raw, y_train_raw)

        self.supp_tab = supp_tab_emb.to(self.device)
        self.supp_doc = supp_doc_emb.to(self.device)
        self.supp_y = torch.tensor(y_train_raw, dtype=torch.float32, device=self.device)
        self.baseline_doc = torch.zeros_like(self.supp_doc[0]).to(self.device)

    # --- STAGE 1: MACRO SHAP ---
    def stage1_macro_shap(self, query_tab_raw: np.ndarray, query_doc_emb: torch.Tensor, background_tab_raw: np.ndarray, save_path="macro_shap.png"):
        logger.info("\n--- Running Stage 1: Macro Modality SHAP ---")
        background_docs = np.zeros((background_tab_raw.shape[0], 1))
        background_data = np.hstack([background_tab_raw, background_docs])
        
        query_doc_flag = np.array([[1]])
        query_data = np.hstack([query_tab_raw.reshape(1, -1), query_doc_flag])
        
        def _shap_predict_wrapper(Z_matrix):
            X_tab_perms = Z_matrix[:, :-1]
            doc_flags = Z_matrix[:, -1]
            batch_size = 64
            all_probs = []
            
            for i in range(0, len(Z_matrix), batch_size):
                x_batch = X_tab_perms[i:i+batch_size]
                flags_batch = doc_flags[i:i+batch_size]
                
                tab_embs = self.tab_embedder.get_embeddings(self.X_train_raw, self.y_train_raw, x_batch, data_source="test")
                tab_embs = torch.tensor(tab_embs, dtype=torch.float32, device=self.device)
                if tab_embs.dim() == 3:
                    tab_embs = tab_embs.squeeze(0) if tab_embs.size(0) == 1 else tab_embs.squeeze(1)
                
                batch_docs = []
                for flag in flags_batch:
                    batch_docs.append(query_doc_emb.squeeze(0).to(self.device) if flag > 0.5 else self.baseline_doc)
                batch_docs = torch.stack(batch_docs)
                
                with torch.no_grad():
                    logits, _ = self.model(self.supp_tab, self.supp_doc, self.supp_y, tab_embs, batch_docs)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.extend(probs)
                    
            return np.array(all_probs)

        logger.info(f"Executing KernelExplainer with {len(background_data)} background samples...")
        explainer = shap.KernelExplainer(_shap_predict_wrapper, background_data, feature_names=self.feature_names)
        shap_values = explainer.shap_values(query_data, nsamples=500)
        
        exp = shap.Explanation(
            values=shap_values[0], base_values=explainer.expected_value, 
            data=query_data[0], feature_names=self.feature_names
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(exp, show=False)
        plt.title("Stage 1: Multimodal Driver Analysis (Tabular vs. Document)")
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved Macro SHAP Waterfall plot to: {save_path}")
        plt.close()

    # --- STAGE 2: MICRO SALIENCY (Direct Cross-Attention Maps) ---
    def _get_grid_shape(self, image: Image.Image) -> tuple:
        batch_doc = self.processor.process_images([image])
        if hasattr(batch_doc, 'image_grid_thw'):
            return int(batch_doc.image_grid_thw[0][1]), int(batch_doc.image_grid_thw[0][2])
        raise ValueError("Processor did not return spatial grid sizes.")

    def stage2_micro_saliency(self, query_tab_emb: torch.Tensor, query_doc_emb: torch.Tensor, image_path: str, save_path="micro_saliency.png"):
        logger.info("\n--- Running Stage 2: Micro Document Saliency (Cross-Attention Map) ---")
        
        # Saliency plot requires actual hypnogram PNG
        if not os.path.exists(image_path):
            logger.warning(f"Raw PNG file not found at {image_path}. Generating mock hypnogram image.")
            os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
            # Make random mock RGB image representing a clinical chart
            Image.fromarray(np.uint8(np.random.rand(800, 600, 3)*255)).save(image_path)
            
        image = Image.open(image_path).convert("RGB")
        vit_grid_h, vit_grid_w = self._get_grid_shape(image)
        llm_grid_h, llm_grid_w = vit_grid_h // 2, vit_grid_w // 2
        expected_image_tokens = llm_grid_h * llm_grid_w

        q_tab = query_tab_emb.unsqueeze(0).to(self.device, dtype=torch.float32)
        q_doc = query_doc_emb.clone().detach().unsqueeze(0).to(self.device, dtype=torch.float32)
        
        # Forward pass to capture attention weights
        with torch.no_grad():
            _ = self.model(self.supp_tab, self.supp_doc, self.supp_y, q_tab, q_doc)
        
        # Slice raw attention weights from pooler
        attn_raw = self.model.doc_pooler.last_attn_weights.squeeze().cpu().numpy()
        
        # Identify spatial token slice offsets
        num_tokens = len(attn_raw)
        start_idx = 4
        if start_idx + expected_image_tokens > num_tokens:
            start_idx = max(0, num_tokens - expected_image_tokens)
            
        attn_img_tokens = attn_raw[start_idx : start_idx + expected_image_tokens]
        
        # Min-max normalization for blending
        attn_min, attn_max = attn_img_tokens.min(), attn_img_tokens.max()
        if attn_max > attn_min: 
            attn_img_tokens = (attn_img_tokens - attn_min) / (attn_max - attn_min)

        # Reshape to 2D grid and resize to match original dimensions
        attn_2d = attn_img_tokens.reshape(llm_grid_h, llm_grid_w)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_orig, w_orig, _ = img_cv.shape
        attn_resized = cv2.resize(attn_2d, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        attn_resized = np.clip(attn_resized, 0, 1)
        
        # Threshold low weights for pure transparency
        attn_resized[attn_resized < 0.05] = 0.01 
        
        # Overlay heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
        mask = (attn_resized > 0)[..., np.newaxis] 
        blended = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)
        overlay_rgb = cv2.cvtColor(np.where(mask, blended, img_cv), cv2.COLOR_BGR2RGB)
        
        # Draw side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(image)
        axes[0].set_title("Original Clinical Document")
        axes[0].axis('off')
        
        axes[1].imshow(overlay_rgb)
        axes[1].set_title("Stage 2: Direct Cross-Attention Saliency")
        axes[1].axis('off')
        
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved Attention overlay to: {save_path}")
        plt.close()


# =====================================================================
# Explanation Orchestrator entrypoint
# =====================================================================
def run_explanation(config: ConfigNode):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading clinical dataset from: {config.data.csv_path}")
    df = pd.read_csv(config.data.csv_path)

    target_label = config.explanation.target_label
    cond_on = getattr(config.explanation, 'condition_on', None)
    patient_id_col = config.data.patient_id_col
    feature_cols = config.data.feature_cols

    sub_df = df.dropna(subset=[target_label]).copy()
    if cond_on is not None:
        sub_df = sub_df[sub_df[cond_on] == 1].reset_index(drop=True)
        model_name = f"{target_label}_cond_on_{cond_on}"
    else:
        model_name = f"{target_label}"
        
    sub_df = sub_df[sub_df[target_label].isin([0, 1])].reset_index(drop=True)

    # Recreate CV folds split to load matching support set
    splits, sub_df_clean = get_stratified_kfold_splits(
        df=sub_df, 
        label_col=target_label, 
        group_col=patient_id_col, 
        n_splits=5
    )
    
    FOLD = config.explanation.fold
    train_idx, test_idx = splits[FOLD]
    
    train_df = sub_df_clean.iloc[train_idx]
    test_df = sub_df_clean.iloc[test_idx]

    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[target_label].values

    # Determine dynamic tabular dimension using embedder
    TabPFNClassifier, TabPFNEmbedding = get_tabpfn_classes()
    clf_dummy = TabPFNClassifier(n_estimators=1, device=device)
    clf_dummy.fit(X_train_raw, y_train_raw)
    tab_embedder = TabPFNEmbedding(tabpfn_clf=clf_dummy, n_fold=5)
    tab_embedder.fit(X_train_raw, y_train_raw)
    
    supp_tab_embs = tab_embedder.get_embeddings(X_train_raw, y_train_raw, X_train_raw, data_source="train")
    supp_tab_embs = torch.tensor(supp_tab_embs, dtype=torch.float32)
    if supp_tab_embs.dim() == 3:
        supp_tab_embs = supp_tab_embs.squeeze(0) if supp_tab_embs.size(0) == 1 else supp_tab_embs.squeeze(1)
        
    TAB_EMBED_DIM = supp_tab_embs.shape[1]
    logger.info(f"Detected Dynamic TabPFN Prior Dimension: {TAB_EMBED_DIM}")

    # Load Model Weights
    model_path = os.path.join(config.explanation.model_save_dir, f"best_{model_name}_fold_{FOLD}.pt")
    if not os.path.exists(model_path):
        # Create dummy state dict if missing (so we can compile/dry-run checks without crashing)
        logger.warning(f"Trained model not found at {model_path}. Creating a temporary model for explainability dry-runs.")
        model = DoTPFN(tab_embed_dim=TAB_EMBED_DIM, doc_input_dim=config.model.doc_input_dim, d_model=config.model.d_model, num_heads=config.model.num_heads)
    else:
        model = DoTPFN(tab_embed_dim=TAB_EMBED_DIM, doc_input_dim=config.model.doc_input_dim, d_model=config.model.d_model, num_heads=config.model.num_heads)
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Pre-loading support set documents
    supp_doc_embs = load_all_document_embeddings(train_df)

    # explainer setup
    explainer = HierarchicalMultimodalExplainer(
        model=model, 
        feature_names=feature_cols, 
        colqwen_model_path=config.explanation.colqwen_model_path,
        device=device
    )
    explainer.setup_support_set(X_train_raw, y_train_raw, supp_tab_embs, supp_doc_embs)
    
    # KernelSHAP summary
    background_tab_raw = shap.kmeans(X_train_raw, 10).data 

    # Explain query patient
    PATIENT_IDX = config.explanation.patient_idx
    query_patient = train_df.iloc[PATIENT_IDX]
    query_patient_id = query_patient[patient_id_col]
    logger.info(f"\n>>> Generating Explainability Report for Patient ID: {query_patient_id} <<<")
    
    query_tab_raw = query_patient[feature_cols].values
    query_doc_file = query_patient['embedding_file']
    
    # Extract test embeddings
    query_tab_emb = tab_embedder.get_embeddings(X_train_raw, y_train_raw, query_tab_raw.reshape(1, -1), data_source="test")
    query_tab_emb = torch.tensor(query_tab_emb, dtype=torch.float32).squeeze()
    
    q_e = torch.load(query_doc_file, map_location='cpu')
    query_doc_emb = q_e.squeeze(0).float() if q_e.dim() == 3 else q_e.float()
    
    # Resolve Raw image path
    id_for_path = os.path.basename(query_doc_file).replace('.pt', '')
    query_image_path = os.path.join(config.data.image_dir, f"{id_for_path}.png")
    
    # Stage 1: Macro SHAP
    shap_path = os.path.join(config.explanation.output_dir, f"SHAP_{model_name}_Patient_{query_patient_id}.png")
    explainer.stage1_macro_shap(
        query_tab_raw=query_tab_raw,
        query_doc_emb=query_doc_emb,
        background_tab_raw=background_tab_raw,
        save_path=shap_path
    )
    
    # Stage 2: Micro Saliency Map
    saliency_path = os.path.join(config.explanation.output_dir, f"GradCAM_{model_name}_Patient_{query_patient_id}.png")
    explainer.stage2_micro_saliency(
        query_tab_emb=query_tab_emb,
        query_doc_emb=query_doc_emb,
        image_path=query_image_path,
        save_path=saliency_path
    )
    logger.info("\nExplainability report generated successfully!")
