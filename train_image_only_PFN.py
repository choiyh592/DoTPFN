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

warnings.filterwarnings('ignore')

# ==========================================
# 1. Modality Projector (Document/Image Embedder)
# ==========================================
class CrossAttentionWithLearnedQueries(nn.Module):
    """
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

    def forward(self, x):
        # Ensure sequence dimension exists: (Batch, Seq_Len, Dim)
        if x.dim() == 2:
            x = x.unsqueeze(1) 
            
        x_proj = self.proj(x)
        batch_size = x_proj.size(0)
        
        # Expand queries for the batch
        q = self.queries.expand(batch_size, -1, -1)
        
        # Cross-attention: Queries attend to Document features
        attn_out, _ = self.mha(query=q, key=x_proj, value=x_proj)
        
        # Flatten the pooled tokens (Batch, num_queries * embed_dim)
        return attn_out.reshape(batch_size, -1)

# ==========================================
# 2. Image/Document-Only Architecture
# ==========================================
class InContextDecoder(nn.Module):
    """
    Implements Bayesian-like inference on document embeddings.
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
            support_x: (1, N_support, D) - Train embeddings
            support_y: (1, N_support)    - Train labels
            query_x:   (1, N_query, D)   - Test embeddings
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


class DocOnlyPFN(nn.Module):
    """
    End-to-End Image/Doc Modality Model: Projects image/doc data into a shared 
    d_model domain and executes Bayesian-like in-context decoding.
    """
    def __init__(self, doc_input_dim=128, d_model=256, num_heads=4, dropout=0.2):
        super().__init__()
        
        # Modality Projector: map doc embeddings to the shared d_model domain
        self.doc_pooler = CrossAttentionWithLearnedQueries(
            input_dim=doc_input_dim, 
            embed_dim=d_model, 
            num_heads=num_heads,
            num_queries=1
        )
        
        # Non-linear projection to finalize the embeddings representation
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.decoder = InContextDecoder(embed_dim=d_model, num_heads=num_heads)

    def forward_features(self, doc_emb):
        doc_pooled = self.doc_pooler(doc_emb)
        embs = self.proj(doc_pooled)
        return embs

    def forward(self, support_doc, support_y, query_doc):
        # Extract modality token embeddings
        support_embs = self.forward_features(support_doc).unsqueeze(0) # (1, N_supp, D)
        query_embs = self.forward_features(query_doc).unsqueeze(0)     # (1, N_query, D)
        support_y = support_y.unsqueeze(0)                             # (1, N_supp)
        
        # Bayesian Inference
        logits = self.decoder(support_embs, support_y, query_embs)
        return logits.squeeze(0) # (N_query)


# ==========================================
# 3. Pipeline Engine (Episodic Meta-Learning)
# ==========================================
class ClinicalDocPipeline:
    def __init__(self, device, model_save_dir):
        self.device = device
        self.model_save_dir = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

    def evaluate_model(self, model, supp_doc, supp_y, query_doc, query_y):
        """ Evaluates using the entire train set as the Bayesian Support Set. """
        model.eval()
        with torch.no_grad():
            logits = model(
                supp_doc.to(self.device), supp_y.to(self.device),
                query_doc.to(self.device)
            )
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
        Executes Episodic training: Processes doc embeddings and trains the In-Context Decoder.
        """
        y_train = train_df[label_col].values
        y_val = val_df[label_col].values
            
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
        
        doc_dim = train_doc_embs.size(-1)
        
        # Use a fixed d_model (e.g., 256) that is guaranteed to be divisible by num_heads (4)
        model = DocOnlyPFN(
            doc_input_dim=doc_dim, 
            d_model=256, 
            num_heads=4
        ).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        
        epochs = 100
        steps_per_epoch = 15 # Emulate meta-learning episodes
        patience = 15
        best_auroc = 0.0
        best_weights = None
        patience_counter = 0
        
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
                
                logits = model(
                    train_doc_embs[supp_idx].to(self.device), train_y[supp_idx].to(self.device),
                    train_doc_embs[query_idx].to(self.device)
                )
                
                loss = criterion(logits, train_y[query_idx].to(self.device))
                loss.backward()
                optimizer.step()
                
            # Cross-validate using entire train set as the Bayesian Support 
            metrics = self.evaluate_model(
                model, train_doc_embs, train_y, 
                val_doc_embs, val_y
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
            model.load_state_dict(best_weights)
            torch.save(best_weights, f"{self.model_save_dir}/best_{save_name}_fold_{fold}.pt")
            
        return self.evaluate_model(model, train_doc_embs, train_y, val_doc_embs, val_y)

    def run_cv_experiment(self, df, labels, group_col=None, condition_ons=None, cond_label=1):
        summary_results = []

        for i, label in enumerate(labels):
            # Clean data
            sub_df = df.dropna(subset=[label]).copy()

            # Optional Conditional filtering
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

            # Stratification
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
                
                # Check class representation
                if len(train_df[label].unique()) < 2:
                    print("  Skipping fold: missing positive or negative class in train.")
                    continue

                metrics = self.train_one_fold(train_df, val_df, label, fold, model_save_name)
                fold_metrics.append(metrics)
                print(f"      Validation AUROC: {metrics['AUROC']:.4f} | F1: {metrics['F1']:.4f}")

            # Aggregate
            if fold_metrics:
                avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
                std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
                
                res = {"Label": model_save_name}
                for k in avg_metrics:
                    res[f"{k}_mean"] = avg_metrics[k]
                    res[f"{k}_std"] = std_metrics[k]
                summary_results.append(res)

        # Final Consolidated Reporting
        print("\n" + "="*50)
        print("FINAL CONSOLIDATED DOC-ONLY RESULTS (5-FOLD CV)")
        print("="*50)
        report_df = pd.DataFrame(summary_results)
        if not report_df.empty:
            for m in ["AUROC", "AUPRC", "Acc", "F1"]:
                report_df[m] = report_df.apply(lambda x: f"{x[m+'_mean']:.3f} ± {x[m+'_std']:.3f}", axis=1)
            print(report_df[["Label", "AUROC", "AUPRC", "Acc", "F1"]].to_string(index=False))
        
        return report_df


# ==========================================
# End-to-End Usage with Real Data
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    CSV_PATH = "/home/yhchoi/PSG_DocParse_260501/final_combined_data_67.csv"
    MODEL_SAVE_DIR = "/home/yhchoi/PSG_DocParse_260501/exp_doconly_260511"
    LOG_SAVE_PATH = "/home/yhchoi/PSG_DocParse_260501/exp_doconly_260511/logs/results.csv"
    
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

    COND_ONS = None
    PATIENT_ID_COL = "ID"
    
    # 2. Load Real Data
    print(f"\nLoading real data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # 3. Initialize and Run Pipeline
    pipeline = ClinicalDocPipeline(
        device=device,
        model_save_dir=MODEL_SAVE_DIR
    )
    
    results_df = pipeline.run_cv_experiment(
        df=df,
        labels=LABELS_TO_PROCESS,
        group_col=PATIENT_ID_COL,
        condition_ons=COND_ONS
    )
    
    # Save logs
    os.makedirs(os.path.dirname(LOG_SAVE_PATH), exist_ok=True)
    results_df.to_csv(LOG_SAVE_PATH, index=False)
    print(f"\nPipeline execution complete. Logs saved to {LOG_SAVE_PATH}")