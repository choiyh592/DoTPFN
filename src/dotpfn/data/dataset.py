import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

class EmbeddingDataset(Dataset):
    """PyTorch Dataset that loads document/image embeddings dynamically from disk."""
    def __init__(self, df: pd.DataFrame, label_col: str):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = torch.load(row['embedding_file'], map_location='cpu')
        
        # If ColQwen2 output has batch dimension, squeeze it (1, L, D) -> (L, D)
        if emb.dim() == 3 and emb.size(0) == 1:
            emb = emb.squeeze(0)
            
        return emb.float(), torch.tensor(row[self.label_col], dtype=torch.float32)


def load_all_document_embeddings(df: pd.DataFrame) -> torch.Tensor:
    """Helper to preload all document/image embeddings into a single memory block."""
    embeddings = []
    for filepath in df['embedding_file']:
        emb = torch.load(filepath, map_location='cpu')
        if emb.dim() == 3 and emb.size(0) == 1:
            emb = emb.squeeze(0)
        embeddings.append(emb.float())
    return torch.stack(embeddings)


def get_stratified_kfold_splits(df: pd.DataFrame, label_col: str, group_col: str = None, n_splits: int = 5, random_state: int = 42):
    """Generates K-Fold splits. Supports patient-level grouping (StratifiedGroupKFold) or normal StratifiedKFold."""
    # Ensure there are no NaNs in label
    df_clean = df.dropna(subset=[label_col]).copy()
    df_clean = df_clean[df_clean[label_col].isin([0, 1])].reset_index(drop=True)
    
    if group_col and group_col in df_clean.columns:
        skf = StratifiedGroupKFold(n_splits=n_splits)
        splits = list(skf.split(df_clean, df_clean[label_col], groups=df_clean[group_col]))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(df_clean, df_clean[label_col]))
        
    return splits, df_clean
