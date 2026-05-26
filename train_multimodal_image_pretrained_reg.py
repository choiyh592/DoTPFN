import os
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    precision_score, recall_score, f1_score
)

# TabPFN Extensions Library
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

    def forward(self, x, return_attn=False):
        # Ensure sequence dimension exists: (Batch, Seq_Len, Dim)
        if x.dim() == 2:
            x = x.unsqueeze(1) 
            
        x_proj = self.proj(x)
        batch_size = x_proj.size(0)
        
        # Expand queries for the batch
        q = self.queries.expand(batch_size, -1, -1)
        
        # Cross-attention: Queries attend to Document features
        attn_out, attn_weights = self.mha(query=q, key=x_proj, value=x_proj)
        
        # Flatten the pooled tokens (Batch, num_queries * embed_dim)
        out = attn_out.reshape(batch_size, -1)
        
        if return_attn:
            return out, attn_weights
        return out

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
        
        # Explanability Anchor: populated dynamically during training to regularize feature drift
        self.frozen_doc_pooler = None
        
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
        
        # If an explainability anchor is present, track attention drift
        if self.frozen_doc_pooler is not None:
            doc_pooled, attn_after = self.doc_pooler(doc_emb, return_attn=True)
            with torch.no_grad():
                _, attn_before = self.frozen_doc_pooler(doc_emb, return_attn=True)
            # Alignment regularization loss
            reg_loss = nn.functional.mse_loss(attn_after, attn_before)
        else:
            doc_pooled = self.doc_pooler(doc_emb)
            reg_loss = torch.tensor(0.0, device=tab_emb.device)
            
        joint_embs = self.fusion(torch.cat([tab_proj, doc_pooled], dim=-1))
        return joint_embs, reg_loss

    def forward(self, support_tab, support_doc, support_y, query_tab, query_doc):
        # Extract fused token embeddings and track attention penalties
        support_embs, supp_reg = self.forward_features(support_tab, support_doc)
        query_embs, query_reg = self.forward_features(query_tab, query_doc)
        
        support_embs = support_embs.unsqueeze(0) # (1, N_supp, D)
        query_embs = query_embs.unsqueeze(0)       # (1, N_query, D)
        support_y = support_y.unsqueeze(0)                                          # (1, N_supp)
        
        # TabPFN Bayesian Inference
        logits = self.decoder(support_embs, support_y, query_embs)
        
        total_reg_loss = supp_reg + query_reg
        return logits.squeeze(0), total_reg_loss # (N_query)


# ==========================================
# 3. Pipeline Engine (Episodic Meta-Learning)
# ==========================================
class ClinicalMultimodalPipeline:
    def __init__(self, feature_cols, device, model_save_dir, pretrained_image_dir=None):
        self.feature_cols = feature_cols
        self.device = device
        self.model_save_dir = model_save_dir
        self.pretrained_image_dir = pretrained_image_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

    def extract_tabpfn_embeddings(self, X_train, y_train, X_val):
        """Extracts structural tabular priors using the TabPFN Foundation Encoder."""
        print(f"    -> Fitting TabPFN Prior on {len(X_train)} samples...")
        clf = TabPFNClassifier(n_estimators=1, device=self.device)
        clf.fit(X_train, y_train)
        
        embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=5)
        embedder.fit(X_train, y_train)
        
        # Use the updated API: get_embeddings instead of transform
        train_tab_emb = embedder.get_embeddings(X_train, y_train, X_val, data_source="train")
        val_tab_emb = embedder.get_embeddings(X_train, y_train, X_val, data_source="test")
        
        return train_tab_emb, val_tab_emb

    def evaluate_model(self, model, supp_tab, supp_doc, supp_y, query_tab, query_doc, query_y):
        """ Evaluates using the entire train set as the Bayesian Support Set. """
        model.eval()
        with torch.no_grad():
            out = model(
                supp_tab.to(self.device), supp_doc.to(self.device), supp_y.to(self.device),
                query_tab.to(self.device), query_doc.to(self.device)
            )
            # Unpack if the model returns a regularization loss tuple
            if isinstance(out, tuple):
                logits, _ = out
            else:
                logits = out
            probs = torch.sigmoid(logits).cpu().numpy()
        
        all_labels = query_y.numpy()
        preds = (probs >= 0.5).astype(int)
        
        if len(np.unique(all_labels)) == 1:
            return {"AUROC": 0.0, "AUPRC": 0.0, "Acc": accuracy_score(all_labels, preds), "F1": 0.0}
            
        return {
            "AUROC": roc_auc_score(all_labels, probs),
            "AUPRC": average_precision_score(all_labels, probs),
            "Acc": accuracy_score(all_labels, preds),
            "F1": f1_score(all_labels, preds, zero_division=0)
        }

    def train_one_fold(self, train_df, val_df, label_col, fold, save_name):
        """
        Executes Episodic TabPFN training: Fuses modalities and trains the In-Context Decoder.
        """
        X_train = train_df[self.feature_cols].values
        y_train = train_df[label_col].values
        X_val = val_df[self.feature_cols].values
        y_val = val_df[label_col].values
        
        # Extract Tabular PFN representations
        train_tab_emb, val_tab_emb = self.extract_tabpfn_embeddings(X_train, y_train, X_val)
        train_tab_emb = torch.tensor(train_tab_emb, dtype=torch.float32)
        val_tab_emb = torch.tensor(val_tab_emb, dtype=torch.float32)
        
        # Remove dummy batch dimension returned by TabPFN (e.g. (1, N, D) -> (N, D))
        if train_tab_emb.dim() == 3:
            train_tab_emb = train_tab_emb.squeeze(0) if train_tab_emb.size(0) == 1 else train_tab_emb.squeeze(1)
        if val_tab_emb.dim() == 3:
            val_tab_emb = val_tab_emb.squeeze(0) if val_tab_emb.size(0) == 1 else val_tab_emb.squeeze(1)
            
        # Preload Document Embeddings to memory
        def load_docs(df):
            embs = []
            for f in df['embedding_file']:
                e = torch.load(f)
                if e.dim() == 3 and e.size(0) == 1: e = e.squeeze(0)
                embs.append(e.float()) # Ensure casting to float32 to prevent dtype mismatches
            return torch.stack(embs)
            
        train_doc_embs = load_docs(train_df)
        val_doc_embs = load_docs(val_df)
        
        train_y = torch.tensor(y_train, dtype=torch.float32)
        val_y = torch.tensor(y_val, dtype=torch.float32)
        
        tab_dim = train_tab_emb.shape[1]
        doc_dim = train_doc_embs.size(-1)
        
        # Use dimensions matching the pre-trained image model
        model = DoTPFN(
            tab_embed_dim=tab_dim, 
            doc_input_dim=doc_dim, 
            doc_embed_dim=128,   # Matched to image-only default embed_dim
            doc_num_heads=8,     # Matched to image-only default num_heads
            d_model=256, 
            num_heads=4
        ).to(self.device)

        # ----------------------------------------------------
        # Load Pre-Trained Image Weights 
        # ----------------------------------------------------
        if self.pretrained_image_dir:
            pretrained_path = os.path.join(self.pretrained_image_dir, f"best_model_label_{save_name}_fold_{fold}.pt")
            if os.path.exists(pretrained_path):
                print(f"    -> Loading pretrained document pooler weights from: {pretrained_path}")
                pretrained_state = torch.load(pretrained_path, map_location=self.device)
                
                # Map keys from image-only model ('pooler.*') to multimodal model ('doc_pooler.*')
                mapped_state = {}
                for k, v in pretrained_state.items():
                    if k.startswith('pooler.'):
                        new_key = k.replace('pooler.', 'doc_pooler.')
                        mapped_state[new_key] = v
                
                # Load strictly the mapped pooler weights
                model.load_state_dict(mapped_state, strict=False)
                print(f"    -> Successfully loaded {len(mapped_state)} pre-trained tensors.")
                
                # === STRATEGY INTEGRATION: Unfreeze image encoder but configure alignment anchor ===
                print("    -> Unfreezing document pooler weights for regularized end-to-end training.")
                for param in model.doc_pooler.parameters():
                    param.requires_grad = True
                
                # Create a frozen copy of the pristine pooler state to act as our baseline grounding anchor
                model.frozen_doc_pooler = copy.deepcopy(model.doc_pooler)
                model.frozen_doc_pooler.eval()
                for param in model.frozen_doc_pooler.parameters():
                    param.requires_grad = False
            else:
                print(f"    -> WARNING: Pretrained weights not found at {pretrained_path}. Training from scratch.")

        criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer filters out frozen parameters automatically (i.e. model.frozen_doc_pooler is skipped)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-3)
        
        epochs = 100
        steps_per_epoch = 25 # Emulate meta-learning episodes
        patience = 15
        best_auroc = 0.0
        best_weights = None
        patience_counter = 0
        
        # Regularization Coefficient: Higher values prioritize preservation of original CAM mappings
        lambda_reg = 0.75
        
        for epoch in range(epochs):
            model.train()
            for step in range(steps_per_epoch):
                optimizer.zero_grad()
                
                # Episodic Context Sampling (e.g. 70% support, 30% query per episode)
                indices = torch.randperm(len(train_y))
                supp_size = max(1, int(len(indices) * 0.7))
                supp_idx, query_idx = indices[:supp_size], indices[supp_size:]
                
                if len(query_idx) == 0: 
                    continue
                
                out = model(
                    train_tab_emb[supp_idx].to(self.device), train_doc_embs[supp_idx].to(self.device), train_y[supp_idx].to(self.device),
                    train_tab_emb[query_idx].to(self.device), train_doc_embs[query_idx].to(self.device)
                )
                
                if isinstance(out, tuple):
                    logits, reg_loss = out
                else:
                    logits, reg_loss = out, torch.tensor(0.0, device=self.device)
                
                # Unified training target balance
                loss = criterion(logits, train_y[query_idx].to(self.device)) + lambda_reg * reg_loss
                loss.backward()
                optimizer.step()
                
            # Cross-validate using entire train set as the Bayesian Support 
            metrics = self.evaluate_model(
                model, train_tab_emb, train_doc_embs, train_y, 
                val_tab_emb, val_doc_embs, val_y
            )
            
            if metrics["AUROC"] > best_auroc:
                best_auroc = metrics["AUROC"]
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience: 
                    break
                    
        if best_weights:
            # Clean up the checkpoint to drop the anchor submodule before storing weights
            clean_weights = {k: v for k, v in best_weights.items() if not k.startswith('frozen_doc_pooler.')}
            model.load_state_dict(best_weights)
            torch.save(clean_weights, f"{self.model_save_dir}/best_{save_name}_fold_{fold}.pt")
            
        return self.evaluate_model(model, train_tab_emb, train_doc_embs, train_y, val_tab_emb, val_doc_embs, val_y)

    def run_cv_experiment(self, df, labels, group_col=None, condition_ons=None, cond_label=1):
        summary_results = []

        for i, label in enumerate(labels):
            sub_df = df.dropna(subset=[label]).copy()

            if condition_ons is not None:
                model_save_name = f"{label}_cond_on_{condition_ons[i]}"
                sub_df = sub_df[sub_df[condition_ons[i]] == cond_label].reset_index(drop=True)
            else:
                model_save_name = label

            print(f"\n>>> Starting 5-Fold CV for Label: {model_save_name} (N={len(sub_df)})")
            sub_df = sub_df[sub_df[label].isin([0, 1])].reset_index(drop=True)
            
            if len(sub_df) < 10:
                print(f"Skipping {model_save_name} due to insufficient samples.")
                continue

            if group_col and group_col in sub_df.columns:
                skf = StratifiedGroupKFold(n_splits=5)
                splits = skf.split(sub_df, sub_df[label], groups=sub_df[group_col])
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                splits = skf.split(sub_df, sub_df[label])

            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(splits):
                print(f"  --- Fold {fold+1}/5 ---")
                train_df, val_df = sub_df.iloc[train_idx], sub_df.iloc[val_idx]
                
                if len(train_df[label].unique()) < 2:
                    print("  Skipping fold: missing positive or negative class in train.")
                    continue

                metrics = self.train_one_fold(train_df, val_df, label, fold, model_save_name)
                fold_metrics.append(metrics)
                print(f"      Validation AUROC: {metrics['AUROC']:.4f} | F1: {metrics['F1']:.4f}")

            if fold_metrics:
                avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
                std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
                
                res = {"Label": model_save_name}
                for k in avg_metrics:
                    res[f"{k}_mean"] = avg_metrics[k]
                    res[f"{k}_std"] = std_metrics[k]
                summary_results.append(res)

        print("\n" + "="*50)
        print("FINAL CONSOLIDATED MULTIMODAL RESULTS (5-FOLD CV)")
        print("="*50)
        report_df = pd.DataFrame(summary_results)
        if not report_df.empty:
            for m in ["AUROC", "AUPRC", "Acc", "F1"]:
                report_df[m] = report_df.apply(lambda x: f"{x[m+'_mean']:.3f} ± {x[m+'_std']:.3f}", axis=1)
            print(report_df[["Label", "AUROC", "AUPRC", "Acc", "F1"]].to_string(index=False))
        
        return report_df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    CSV_PATH = "/home/won_ju_kim/yhchoi/PSG_260408/final_combined_data.csv"
    MODEL_SAVE_DIR = "/home/won_ju_kim/yhchoi/PSG_260408/psg_exp_1_260526_condons"
    LOG_SAVE_PATH = "/home/won_ju_kim/yhchoi/PSG_260408/psg_exp_1_260526_condons/logs/results.csv"
    PRETRAINED_IMAGE_DIR = "/home/won_ju_kim/yhchoi/PSG_260408/psg_exp_1_260526_imageonly_condons"
    
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
    #     "adherence_3m", "adherence_6m", "adherence_9m", "adherence_1yr", 
    #     "adherence_2yr", "adherence_3yr", "adherence_4yr", 
    #     "adherence_5yr",
    # ]

    # COND_ONS = None

    LABELS_TO_PROCESS = [
        "adherence_1yr",
        "adherence_3yr",
        "adherence_3yr",
        "adherence_5yr",
        "adherence_5yr",
        "adherence_5yr",
        "adherence_5yr",
        "adherence_4yr",
        "adherence_4yr",
        "adherence_4yr",
        "adherence_4yr"
    ]

    COND_ONS = [
        "adherence_3m",  
        "adherence_1yr", 
        "adherence_3m",  
        "adherence_1yr",
        "adherence_3m", 
        "adherence_6m", 
        "adherence_9m", 
        "adherence_3m", 
        "adherence_6m", 
        "adherence_9m",
        "adherence_1yr" 
    ]
    
    PATIENT_ID_COL = "ID"
    
    print(f"\nLoading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    pipeline = ClinicalMultimodalPipeline(
        feature_cols=feature_cols,
        device=device,
        model_save_dir=MODEL_SAVE_DIR,
        pretrained_image_dir=PRETRAINED_IMAGE_DIR
    )
    
    results_df = pipeline.run_cv_experiment(
        df=df,
        labels=LABELS_TO_PROCESS,
        group_col=PATIENT_ID_COL,
        condition_ons=COND_ONS
    )
    
    os.makedirs(os.path.dirname(LOG_SAVE_PATH), exist_ok=True)
    results_df.to_csv(LOG_SAVE_PATH, index=False)
    print(f"\nPipeline execution complete. Logs saved to {LOG_SAVE_PATH}")