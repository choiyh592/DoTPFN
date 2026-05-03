import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    precision_score, recall_score, f1_score
)
from torch.utils.data import Dataset, DataLoader
import copy
from src.crossattentionpooler import CrossAttentionWithLearnedQueries

# ==========================================
# 1. Dataset & Model (Unchanged logic)
# ==========================================
class EmbeddingDataset(Dataset):
    def __init__(self, df, label_col):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = torch.load(row['embedding_file']) 
        if emb.dim() == 3 and emb.size(0) == 1:
            emb = emb.squeeze(0)
        return emb.float(), torch.tensor(row[self.label_col], dtype=torch.float32)

class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        super().__init__()
        self.pooler = CrossAttentionWithLearnedQueries(
            input_dim=embed_dim, embed_dim=embed_dim, num_heads=num_heads
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        pooled_features = self.pooler(x)
        logits = self.classifier(pooled_features)
        return logits.squeeze(-1)

# ==========================================
# 2. Evaluation & Single Fold Training
# ==========================================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for embeddings, labels in dataloader:
            logits = model(embeddings.to(device))
            all_labels.extend(labels.numpy())
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
    
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    
    return {
        "AUROC": roc_auc_score(all_labels, all_probs),
        "AUPRC": average_precision_score(all_labels, all_probs),
        "Acc": accuracy_score(all_labels, preds),
        "F1": f1_score(all_labels, preds, zero_division=0)
    }

def train_one_fold(train_df, val_df, label_col, embed_dim, device, model_save_dir, fold, epochs=500, patience=30):
    print("Starting Fold..")
    train_loader = DataLoader(EmbeddingDataset(train_df, label_col), batch_size=32, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(val_df, label_col), batch_size=32, shuffle=False)
    
    model = AttentionClassifier(embed_dim=embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    best_auroc = 0.0
    best_weights = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(embeddings.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            
        metrics = evaluate_model(model, val_loader, device)
        if metrics["AUROC"] > best_auroc:
            best_auroc = metrics["AUROC"]
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience: break
                
    model.load_state_dict(best_weights)
    torch.save(best_weights, f"{model_save_dir}/best_model_label_{label_col}_fold_{fold}.pt")
    return evaluate_model(model, val_loader, device)

# ==========================================
# 3. 5-Fold CV over Multiple Labels
# ==========================================
def run_cv_experiment(csv_path, labels, model_save_dir, group_col=None, embed_dim=128, condition_ons=None, cond_label=1):
    """
    Runs single experiment.
    TODO: add comment
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_df = pd.read_csv(csv_path)
    summary_results = []

    for i, label in enumerate(labels):
        print(f"\n>>> Starting 5-Fold CV for Label: {label}")
        # Clean data for this specific label
        df = full_df.dropna(subset=[label]).copy()

        if condition_ons != None:
            assert len(condition_ons) == len(labels), "labels and condition_ons should be of same length"
            df = df[df[condition_ons[i]].isin([cond_label])].reset_index(drop=True)

        df = df[df[label].isin([0, 1])].reset_index(drop=True)
        
        # Use GroupKFold if patient_id is provided, else StratifiedKFold
        if group_col and group_col in df.columns:
            skf = StratifiedGroupKFold(n_splits=5)
            splits = skf.split(df, df[label], groups=df[group_col])
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splits = skf.split(df, df[label])

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            metrics = train_one_fold(train_df, val_df, label, embed_dim, device, model_save_dir, fold)
            fold_metrics.append(metrics)
            print(f"  Fold {fold+1} | AUROC: {metrics['AUROC']:.4f} | F1: {metrics['F1']:.4f}")

        # Aggregate Fold Results
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
        std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
        
        res = {"Label": label}
        for k in avg_metrics:
            res[f"{k}_mean"] = avg_metrics[k]
            res[f"{k}_std"] = std_metrics[k]
        summary_results.append(res)

    # Final Logging
    print("\n" + "="*50)
    print("FINAL CONSOLIDATED RESULTS (5-FOLD CV)")
    print("="*50)
    report_df = pd.DataFrame(summary_results)
    # Displaying mean ± std for readability
    for m in ["AUROC", "AUPRC", "Acc", "F1"]:
        report_df[m] = report_df.apply(lambda x: f"{x[m+'_mean']:.3f} ± {x[m+'_std']:.3f}", axis=1)
    
    print(report_df[["Label", "AUROC", "AUPRC", "Acc", "F1"]].to_string(index=False))
    return report_df

if __name__ == "__main__":
    CSV_PATH = "/home/won_ju_kim/yhchoi/PSG_260408/final_combined_data.csv"
    
    # Define your list of labels here
    # LABELS_TO_PROCESS = [
    #     "adherence_3m", 
    #     "adherence_6m", 
    #     "adherence_9m", 
    #     "adherence_1yr", 
    #     "adherence_2yr",
    #     "adherence_3yr",
    #     "adherence_4yr",
    #     "adherence_5yr",
    # ]

    LABELS_TO_PROCESS = [
        "adherence_5yr",
        # "adherence_5yr",
        # "adherence_5yr",
        # "adherence_4yr",
        # "adherence_4yr",
        # "adherence_4yr",
        # "adherence_4yr"
    ]

    # COND_ONS = [
    #     "adherence_3m",
    #     "adherence_6m",
    #     "adherence_9m",
    #     "adherence_3m",
    #     "adherence_6m",
    #     "adherence_9m",
    #     "adherence_1yr"
    # ]

    COND_ONS = None
    
    # If you have a patient ID column (e.g., 'PID'), put it here to keep 
    # the same patient's multiple records in the same fold.
    PATIENT_ID_COL = "ID"

    MODEL_SAVE_DIR = "/home/won_ju_kim/yhchoi/PSG_260408/logs"
    
    df = run_cv_experiment(CSV_PATH, LABELS_TO_PROCESS, MODEL_SAVE_DIR, group_col=PATIENT_ID_COL, condition_ons=COND_ONS)

    df.to_csv("/home/won_ju_kim/yhchoi/PSG_260408/logs/results.csv")