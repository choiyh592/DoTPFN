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
from sklearn.model_selection import StratifiedGroupKFold

# ColPali / Qwen2 Processor for dynamic grid sizing
from colpali_engine.models import ColQwen2Processor

# TabPFN Library
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

warnings.filterwarnings('ignore')

# ==========================================
# 1. Modality Projector (Document Embedder)
# ==========================================
class CrossAttentionWithLearnedQueries(nn.Module):
    """
    Acts as the Cross-Attention Pooler (CAP) from the MultiModalPFN paper.
    Compresses variable-length or high-dimensional document/image embeddings 
    into a compact, fixed-size representation using learned query tokens.
    """
    def __init__(self, input_dim=128, embed_dim=128, num_heads=8, num_queries=1):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        # Learnable queries to extract salient information from the document
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        self.proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # NEW: Store the attention weights for spatial visualization
        self.last_attn_weights = None

    def forward(self, x):
        # Ensure sequence dimension exists: (Batch, Seq_Len, Dim)
        if x.dim() == 2:
            x = x.unsqueeze(1) 
            
        x_proj = self.proj(x)
        batch_size = x_proj.size(0)
        
        # Expand queries for the batch
        q = self.queries.expand(batch_size, -1, -1)
        
        # Cross-attention: Queries attend to Document features
        # MHA returns (attn_output, attn_output_weights)
        attn_out, attn_weights = self.mha(query=q, key=x_proj, value=x_proj)
        
        # Store for the explainer (Shape: batch_size, num_queries, seq_len)
        self.last_attn_weights = attn_weights 
        
        # Flatten the pooled tokens (Batch, num_queries * embed_dim)
        return attn_out.reshape(batch_size, -1)

# ==========================================
# 2. Multimodal MMPFN Architecture (Explicit TabPFN In-Context Decoder)
# ==========================================
class TabPFNInContextDecoder(nn.Module):
    """
    Implements Bayesian-like inference on joint embeddings.
    Test queries cross-attend to the labeled support set (train instances) 
    to yield predictive posteriors conditionally independent of other queries.
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.label_embedding = nn.Embedding(2, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, support_x, support_y, query_x):
        """
        Args:
            support_x: (1, N_support, D) - Train joint embeddings
            support_y: (1, N_support)    - Train labels
            query_x:   (1, N_query, D)   - Test joint embeddings
        """
        # Embed train labels and fuse into support features
        supp_y_emb = self.label_embedding(support_y.long())
        support_features = support_x + supp_y_emb
        
        # Concatenate sequences for the transformer: (1, N_support + N_query, D)
        combined_x = torch.cat([support_features, query_x], dim=1)
        
        N_supp = support_x.size(1)
        N_query = query_x.size(1)
        N_tot = N_supp + N_query
        
        # Build strict Causal Mask for Bayesian Inference:
        # 1. Support attends to Support
        # 2. Query attends to Support
        # 3. Query CANNOT attend to other Queries (conditional independence)
        mask = torch.zeros(N_tot, N_tot, dtype=torch.bool, device=support_x.device)
        mask[:, N_supp:] = True

        mask.diagonal().fill_(False) # Queries can still attend to themselves
        
        # Contextual inference
        out = self.transformer(combined_x, mask=mask)
        
        # Extract contextualized query outputs and map to logits
        query_out = out[:, N_supp:, :]
        return self.classifier(query_out).squeeze(-1)


class DoTPFN(nn.Module):
    """
    End-to-End MMPFN: Projects image/doc data into the TabPFN domain, 
    fuses them, and executes Bayesian-like in-context decoding.
    """
    def __init__(self, tab_embed_dim, doc_input_dim=128, doc_embed_dim=128, doc_num_heads=8, d_model=256, num_heads=4, dropout=0.2):
        super().__init__()
        
        # Project arbitrary TabPFN embeddings to a clean, divisible dimension (d_model)
        self.tab_proj = nn.Linear(tab_embed_dim, d_model)
        
        # Modality Projector: map doc embeddings to matching pre-trained dim (doc_embed_dim)
        self.doc_pooler = CrossAttentionWithLearnedQueries(
            input_dim=doc_input_dim, 
            embed_dim=doc_embed_dim, 
            num_heads=doc_num_heads,
            num_queries=1
        )
        
        # Align joint dimensions (d_model from tab + doc_embed_dim from image) -> d_model
        self.fusion = nn.Sequential(
            nn.Linear(d_model + doc_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.decoder = TabPFNInContextDecoder(embed_dim=d_model, num_heads=num_heads)

    def forward_features(self, tab_emb, doc_emb):
        tab_proj = self.tab_proj(tab_emb)
        doc_pooled = self.doc_pooler(doc_emb)
        joint_embs = self.fusion(torch.cat([tab_proj, doc_pooled], dim=-1))
        return joint_embs

    def forward(self, support_tab, support_doc, support_y, query_tab, query_doc):
        # Extract fused token embeddings
        support_embs = self.forward_features(support_tab, support_doc).unsqueeze(0) # (1, N_supp, D)
        query_embs = self.forward_features(query_tab, query_doc).unsqueeze(0)       # (1, N_query, D)
        support_y = support_y.unsqueeze(0)                                          # (1, N_supp)
        
        # TabPFN Bayesian Inference
        logits = self.decoder(support_embs, support_y, query_embs)
        return logits.squeeze(0) # (N_query)

# ==========================================
# 2. Hierarchical Explainer Pipeline
# ==========================================
class HierarchicalMultimodalExplainer:
    def __init__(self, model: nn.Module, feature_names: list, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.feature_names = feature_names + ["Clinical_Document"]
        self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
        
        # State variables
        self.supp_tab = None
        self.supp_doc = None
        self.supp_y = None
        self.baseline_doc = None
        
        self.X_train_raw = None
        self.y_train_raw = None
        self.tab_embedder = None

    def setup_support_set(self, X_train_raw, y_train_raw, supp_tab_emb, supp_doc_emb):
        """Initializes the frozen support set ('memory') and document baselines."""
        print("Setting up Explainer Support Set and Baselines...")
        self.X_train_raw = X_train_raw
        self.y_train_raw = y_train_raw
        
        # Fit the TabPFN Embedder once
        clf = TabPFNClassifier(n_estimators=1, device=self.device)
        clf.fit(X_train_raw, y_train_raw)
        self.tab_embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=5)
        self.tab_embedder.fit(X_train_raw, y_train_raw)

        # Freeze the Support Set
        self.supp_tab = supp_tab_emb.to(self.device)
        self.supp_doc = supp_doc_emb.to(self.device)
        self.supp_y = torch.tensor(y_train_raw, dtype=torch.float32, device=self.device)
        
        # Calculate the baseline "average" document for SHAP permutations
        # self.baseline_doc = self.supp_doc.mean(dim=0)
        self.baseline_doc = torch.zeros_like(self.supp_doc[0]).to(self.device)

    # --- STAGE 1: MACRO SHAP ---
    def stage1_macro_shap(self, query_tab_raw: np.ndarray, query_doc_emb: torch.Tensor, background_tab_raw: np.ndarray, save_path="macro_shap.png"):
        print("\n--- Running Stage 1: Macro Modality SHAP ---")
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
                    logits = self.model(self.supp_tab, self.supp_doc, self.supp_y, tab_embs, batch_docs)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.extend(probs)
                    
            return np.array(all_probs)

        print(f"Executing KernelExplainer with {len(background_data)} background samples...")
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
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved Macro SHAP Waterfall plot to {save_path}")
        plt.close()

    # --- STAGE 2: MICRO SALIENCY (GRAD-CAM) ---
    def _get_grid_shape(self, image: Image.Image) -> tuple:
        batch_doc = self.processor.process_images([image])
        if hasattr(batch_doc, 'image_grid_thw'):
            return int(batch_doc.image_grid_thw[0][1]), int(batch_doc.image_grid_thw[0][2])
        raise ValueError("Processor did not return spatial grid sizes.")

    def stage2_micro_saliency(self, query_tab_emb: torch.Tensor, query_doc_emb: torch.Tensor, image_path: str, save_path="micro_saliency.png"):
        print("\n--- Running Stage 2: Micro Document Saliency (Cross-Attention Map) ---")
        image = Image.open(image_path).convert("RGB")
        vit_grid_h, vit_grid_w = self._get_grid_shape(image)
        llm_grid_h, llm_grid_w = vit_grid_h // 2, vit_grid_w // 2
        expected_image_tokens = llm_grid_h * llm_grid_w

        q_tab = query_tab_emb.unsqueeze(0).to(self.device, dtype=torch.float32)
        q_doc = query_doc_emb.clone().detach().unsqueeze(0).to(self.device, dtype=torch.float32)
        
        # Forward pass (No gradients needed)
        with torch.no_grad():
            _ = self.model(self.supp_tab, self.supp_doc, self.supp_y, q_tab, q_doc)
        
        # Extract attention weights directly from the Modality Projector
        # Shape: (batch=1, num_queries=1, seq_len) -> (seq_len,)
        attn_raw = self.model.doc_pooler.last_attn_weights.squeeze().cpu().numpy()
        
        # 1. Define total tokens and adjust start_idx FIRST
        num_tokens = len(attn_raw)
        start_idx = 4
        if start_idx + expected_image_tokens > num_tokens:
            start_idx = max(0, num_tokens - expected_image_tokens)
            
        # 2. Slice the spatial tokens safely
        attn_img_tokens = attn_raw[start_idx : start_idx + expected_image_tokens]
        
        # 3. Normalize only the image tokens to [0, 1] range for visualization
        attn_min, attn_max = attn_img_tokens.min(), attn_img_tokens.max()
        if attn_max > attn_min: 
            attn_img_tokens = (attn_img_tokens - attn_min) / (attn_max - attn_min)

        # 4. Reshape into the 2D LLM grid
        attn_2d = attn_img_tokens.reshape(llm_grid_h, llm_grid_w)
        
        # Resize attention map to original image dimensions
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_orig, w_orig, _ = img_cv.shape
        attn_resized = cv2.resize(attn_2d, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        attn_resized = np.clip(attn_resized, 0, 1) # Prevent uint8 underflow
        
        # Optional: I removed the harsh 0.25 threshold from your Grad-CAM 
        # because pure attention maps usually look much better when smoothly blended.
        # However, we set values close to 0 to be transparent.
        attn_resized[attn_resized < 0.05] = 0.01 
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
        
        # Blend with original image
        mask = (attn_resized > 0)[..., np.newaxis] 
        blended = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)
        overlay_rgb = cv2.cvtColor(np.where(mask, blended, img_cv), cv2.COLOR_BGR2RGB)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(image)
        axes[0].set_title("Original Clinical Document")
        axes[0].axis('off')
        
        axes[1].imshow(overlay_rgb)
        axes[1].set_title("Stage 2: Direct Cross-Attention Weights")
        axes[1].axis('off')
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved Attention map to {save_path}")
        plt.close()


# ==========================================
# End-to-End Configuration & Execution
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    CSV_PATH = "/home/won_ju_kim/yhchoi/PSG_260408/final_combined_data.csv"
    MODEL_SAVE_DIR = "/home/won_ju_kim/yhchoi/PSG_260408/pfn_exp_pt_260521"
    
    # 1. Feature Defs
    feature_cols = [
        'SEX', 'AGE', 'Height', 'Weight', 'BMI', 
        'PSQI', 'BDIII', 'ISI', 'ESS', 'SSS', 'STOPBANG', 'BQ', 'HTN', 'DM', 'DL', 
        'Liver', 'Kidney', 'Lung', 'IHD', 'CHF', 'Arrythmia', 'Cancer', 'Thyroid', 
        'Rhinitis', 'CTD', 'Epilepsy', 'CVA', 'Dementia', 'PSY', 'OSA_op', 
        'REM_Episodes', 'TST', 'Total_Stage_N1_ratio', 'Total_Stage_N2_ratio', 
        'Total_Stage_N3_ratio', 'Total_Stage_R_ratio', 'WASO', 'Sleep_latency', 
        'REM_sleep_latency', 'Sleep_Efficiency', 'Arousal_index', 'AHI', 
        'O2_satu_nadir', 'snoring_index', 'PLMs_index', 'PLMar_index', 'AHI_supine', 
        'AHI_REM', 'Apnea_NREM_avg', 'Apnea_REM_avg', 'Hypopnea_NREM_avg', 
        'Hypopnea_REM_avg', 'No_of_desatu', 'Min_O2_NREM', 'Min_O2_REM', 
        'HRV_NREM', 'HRV_REM', 'CCI'
    ]
    
    # LABELS_TO_PROCESS = [
    #     "adherence_1yr", "adherence_3yr", "adherence_3yr", "adherence_5yr",
    #     "adherence_5yr", "adherence_5yr", "adherence_5yr", "adherence_4yr",
    #     "adherence_4yr", "adherence_4yr", "adherence_4yr"
    # ]
    
    LABELS_TO_PROCESS = [
        "adherence_3m", 
        "adherence_6m", 
        "adherence_9m", 
        "adherence_1yr", 
        "adherence_2yr",
        "adherence_3yr",
        "adherence_4yr",
        "adherence_5yr",
    ]

    # COND_ONS = [
    #     "adherence_3m", "adherence_1yr", "adherence_3m", "adherence_1yr",
    #     "adherence_3m", "adherence_6m", "adherence_9m", "adherence_3m", 
    #     "adherence_6m", "adherence_9m", "adherence_1yr" 
    # ]
    COND_ONS = None
    PATIENT_ID_COL = "ID"
    
    # Setup dimension sizes
    DOC_EMBED_DIM = 128 # Standard for ColQwen2/ColPali
    
    # ---------------------------------------------------------
    # AUTOMATED INFERENCE SETUP (Explaining a specific patient)
    # ---------------------------------------------------------
    
    # We will pick the first label configuration as the target model to explain
    target_label = LABELS_TO_PROCESS[7] # "adherence_5yr"
    # cond_on = COND_ONS[0]               # "adherence_3m"
    cond_on = None
    
    # Load Real Data
    print(f"Loading real data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Filter dataset exactly like training pipeline
    sub_df = df.dropna(subset=[target_label]).copy()
    if cond_on is not None:
        sub_df = sub_df[sub_df[cond_on] == 1].reset_index(drop=True)
        model_name = f"{target_label}_cond_on_{cond_on}"
    else:
        model_name = f"{target_label}"
    print(f"\nInitializing Explanation Pipeline for Model: {model_name}")
    sub_df = sub_df[sub_df[target_label].isin([0, 1])].reset_index(drop=True)
    
    # Recreate Fold 0 split to get exact same support/query set
    skf = StratifiedGroupKFold(n_splits=5)
    splits = list(skf.split(sub_df, sub_df[target_label], groups=sub_df[PATIENT_ID_COL]))
    FOLD = 1
    train_idx, test_idx = splits[FOLD] # Target Fold 0
    
    train_df = sub_df.iloc[train_idx]
    test_df = sub_df.iloc[test_idx]
    
    # 1. Generate Support Set Tabular Embeddings FIRST to dynamically find the correct shape
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[target_label].values
    
    print("Fitting TabPFN Embedder to dynamically determine tabular embedding dimension...")
    clf_dummy = TabPFNClassifier(n_estimators=1, device=device)
    clf_dummy.fit(X_train_raw, y_train_raw)
    tab_embedder = TabPFNEmbedding(tabpfn_clf=clf_dummy, n_fold=5)
    tab_embedder.fit(X_train_raw, y_train_raw)
    
    supp_tab_embs = tab_embedder.get_embeddings(X_train_raw, y_train_raw, X_train_raw, data_source="train")
    supp_tab_embs = torch.tensor(supp_tab_embs, dtype=torch.float32)
    if supp_tab_embs.dim() == 3:
        supp_tab_embs = supp_tab_embs.squeeze(0) if supp_tab_embs.size(0) == 1 else supp_tab_embs.squeeze(1)
        
    TAB_EMBED_DIM = supp_tab_embs.shape[1]
    print(f"Detected TabPFN Embedding Dimension: {TAB_EMBED_DIM}")
    
    # 2. Load Model Weights using the exact dynamic embedding dimension
    model_path = os.path.join(MODEL_SAVE_DIR, f"best_{model_name}_fold_{FOLD}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please ensure training is complete.")
        
    model = DoTPFN(tab_embed_dim=TAB_EMBED_DIM, doc_input_dim=DOC_EMBED_DIM, d_model=256, num_heads=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 3. Helper to load Document Embeddings
    def load_docs(dataframe):
        embs = []
        for f in dataframe['embedding_file']:
            e = torch.load(f, map_location='cpu') # Load safely
            if e.dim() == 3 and e.size(0) == 1: e = e.squeeze(0)
            embs.append(e.float())
        return torch.stack(embs)
        
    print("Pre-loading Support Set Document Embeddings...")
    supp_doc_embs = load_docs(train_df)

    # ---------------------------------------------------------
    # INITIALIZE EXPLAINER
    # ---------------------------------------------------------
    explainer = HierarchicalMultimodalExplainer(model=model, feature_names=feature_cols, device=device)
    explainer.setup_support_set(X_train_raw, y_train_raw, supp_tab_embs, supp_doc_embs)
    
    # Background data for KernelSHAP (KMeans summary to keep it fast)
    background_tab_raw = shap.kmeans(X_train_raw, 10).data 
    
    # ---------------------------------------------------------
    # EXPLAIN QUERY PATIENT (Pick Patient #0 from Validation Set)
    # ---------------------------------------------------------
    PATIENT_IDX = 8
    query_patient = train_df.iloc[PATIENT_IDX]
    query_patient_id = query_patient[PATIENT_ID_COL]
    print(f"\n>>> Generating Explainability Report for Test Patient ID: {query_patient_id} <<<")
    
    query_tab_raw = query_patient[feature_cols].values
    query_doc_file = query_patient['embedding_file']
    
    # Calculate exact TabPFN Prior for the specific query
    query_tab_emb = tab_embedder.get_embeddings(X_train_raw, y_train_raw, query_tab_raw.reshape(1, -1), data_source="test")
    query_tab_emb = torch.tensor(query_tab_emb, dtype=torch.float32).squeeze()
    
    # Load Query Document Embedding
    q_e = torch.load(query_doc_file, map_location='cpu')
    query_doc_emb = q_e.squeeze(0).float() if q_e.dim() == 3 else q_e.float()
    
    # Reconstruct Image Path based on standard embedding extension (Update if paths differ!)
    # E.g., /path/to/embeds/123.pt -> /path/to/images/123.png
    # Replace the below path logic with your specific raw image directory if needed
    id_for_path = query_doc_file.split('/')[-1].replace('.pt', '')
    query_image_path = f"/home/won_ju_kim/yhchoi/PSG_260408/PSG_Lastpage_Images_HypnogramMatch/{id_for_path}.png"
    
    if not os.path.exists(query_image_path):
        print(f"WARNING: Image not found at {query_image_path}. Please provide valid image path for Saliency Map.")
        # Create a dummy image just so the script doesn't hard-crash during testing
        query_image_path = "temp_dummy_document.png"
        Image.fromarray(np.uint8(np.random.rand(800, 600, 3)*255)).save(query_image_path)
    
    # --- EXECUTE STAGE 1 ---
    explainer.stage1_macro_shap(
        query_tab_raw=query_tab_raw, 
        query_doc_emb=query_doc_emb, 
        background_tab_raw=background_tab_raw,
        save_path=f"SHAP_{model_name}_Patient_{query_patient_id}.png"
    )
    
    # --- EXECUTE STAGE 2 ---
    explainer.stage2_micro_saliency(
        query_tab_emb=query_tab_emb,
        query_doc_emb=query_doc_emb,
        image_path=query_image_path,
        save_path=f"GradCAM_{model_name}_Patient_{query_patient_id}.png"
    )
    
    print("\n✅ Explainability Pipeline Complete!")