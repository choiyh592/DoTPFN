import os
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotpfn.utils.logging import setup_logger
from dotpfn.utils.metrics import compute_metrics
from dotpfn.utils.config import ConfigNode
from dotpfn.data.dataset import EmbeddingDataset, load_all_document_embeddings, get_stratified_kfold_splits
from dotpfn.models.classifiers import AttentionClassifier, ImageOnlyInContextDecoder
from dotpfn.models.fusion import DoTPFN

warnings.filterwarnings('ignore')
logger = setup_logger("DoTPFN.Train")

from dotpfn.utils.tabpfn_loader import get_tabpfn_classes


# =====================================================================
# 1. Pipeline Engine: Clinical Multimodal Pipeline (DoTPFN)
# =====================================================================
class ClinicalMultimodalPipeline:
    def __init__(self, config: ConfigNode):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model_save_dir = config.training.model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        tabpfn_version = getattr(config.tabpfn, "version", "auto") if hasattr(config, "tabpfn") else "auto"
        self.TabPFNClassifier, self.TabPFNEmbedding = get_tabpfn_classes(version=tabpfn_version)

    def extract_tabpfn_embeddings(self, X_train, y_train, X_val):
        logger.info(f"    -> Fitting TabPFN Prior on {len(X_train)} samples...")
        tabpfn_kwargs = {"n_estimators": 1, "device": self.device}
        if hasattr(self.config, "tabpfn") and hasattr(self.config.tabpfn, "model_path"):
            tabpfn_kwargs["model_path"] = self.config.tabpfn.model_path
        clf = self.TabPFNClassifier(**tabpfn_kwargs)
        clf.fit(X_train, y_train)
        
        embedder = self.TabPFNEmbedding(tabpfn_clf=clf, n_fold=5)
        embedder.fit(X_train, y_train)
        
        train_tab_emb = embedder.get_embeddings(X_train, y_train, X_val, data_source="train")
        val_tab_emb = embedder.get_embeddings(X_train, y_train, X_val, data_source="test")
        
        return train_tab_emb, val_tab_emb

    def evaluate_model(self, model, supp_tab, supp_doc, supp_y, query_tab, query_doc, query_y):
        model.eval()
        with torch.no_grad():
            out = model(
                supp_tab.to(self.device), supp_doc.to(self.device), supp_y.to(self.device),
                query_tab.to(self.device), query_doc.to(self.device)
            )
            logits, _ = out
            probs = torch.sigmoid(logits).cpu().numpy()
        
        return compute_metrics(query_y.numpy(), probs)

    def train_one_fold(self, train_df, val_df, label_col, fold, save_name):
        X_train = train_df[self.config.data.feature_cols].values
        y_train = train_df[label_col].values
        X_val = val_df[self.config.data.feature_cols].values
        y_val = val_df[label_col].values
        
        # 1. Tabular prior extraction
        train_tab_emb, val_tab_emb = self.extract_tabpfn_embeddings(X_train, y_train, X_val)
        train_tab_emb = torch.tensor(train_tab_emb, dtype=torch.float32)
        val_tab_emb = torch.tensor(val_tab_emb, dtype=torch.float32)
        
        # Squeeze dim-3 batches (Handle out-of-fold embeddings or dummy batch)
        if train_tab_emb.dim() == 3:
            train_tab_emb = train_tab_emb.mean(dim=0)
        if val_tab_emb.dim() == 3:
            val_tab_emb = val_tab_emb.mean(dim=0)
            
        # 2. Preload Document Embeddings
        train_doc_embs = load_all_document_embeddings(train_df)
        val_doc_embs = load_all_document_embeddings(val_df)
        
        train_y = torch.tensor(y_train, dtype=torch.float32)
        val_y = torch.tensor(y_val, dtype=torch.float32)
        
        tab_dim = train_tab_emb.shape[-1]
        doc_dim = train_doc_embs.size(-1)
        
        # 3. Model construction
        model = DoTPFN(
            tab_embed_dim=tab_dim, 
            doc_input_dim=doc_dim, 
            doc_embed_dim=self.config.model.doc_embed_dim,
            doc_num_heads=self.config.model.doc_num_heads,
            d_model=self.config.model.d_model,
            num_heads=self.config.model.num_heads,
            dropout=self.config.model.dropout
        ).to(self.device)

        # Pretrained alignment initialization
        if hasattr(self.config.training, 'pretrained_image_dir') and self.config.training.pretrained_image_dir:
            pretrained_path = os.path.join(
                self.config.training.pretrained_image_dir, 
                f"best_model_label_{save_name}_fold_{fold}.pt"
            )
            if os.path.exists(pretrained_path):
                logger.info(f"    -> Loading pretrained document weights from: {pretrained_path}")
                pretrained_state = torch.load(pretrained_path, map_location=self.device)
                
                # Map keys pooler -> doc_pooler
                mapped_state = {}
                for k, v in pretrained_state.items():
                    if k.startswith('pooler.'):
                        mapped_state[k.replace('pooler.', 'doc_pooler.')] = v
                
                model.load_state_dict(mapped_state, strict=False)
                logger.info(f"    -> Successfully loaded {len(mapped_state)} pre-trained tensors.")
                
                # Setup anchor copy for regularization
                for param in model.doc_pooler.parameters():
                    param.requires_grad = True
                
                model.frozen_doc_pooler = copy.deepcopy(model.doc_pooler)
                model.frozen_doc_pooler.eval()
                for param in model.frozen_doc_pooler.parameters():
                    param.requires_grad = False
            else:
                logger.warning(f"    -> Anchor weights not found at {pretrained_path}. Training from scratch.")

        criterion = nn.BCEWithLogitsLoss()
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.config.training.lr, 
            weight_decay=self.config.training.weight_decay
        )
        
        best_auroc = 0.0
        best_weights = None
        patience_counter = 0
        lambda_reg = getattr(self.config.training, 'lambda_reg', 0.75)
        
        # 4. Episodic episodic optimization loops
        for epoch in range(self.config.training.epochs):
            model.train()
            for step in range(self.config.training.steps_per_epoch):
                optimizer.zero_grad()
                
                # Episodic support vs query sets
                indices = torch.randperm(len(train_y))
                supp_size = max(1, int(len(indices) * 0.7))
                supp_idx, query_idx = indices[:supp_size], indices[supp_size:]
                
                if len(query_idx) == 0: 
                    continue
                
                out = model(
                    train_tab_emb[supp_idx].to(self.device), train_doc_embs[supp_idx].to(self.device), train_y[supp_idx].to(self.device),
                    train_tab_emb[query_idx].to(self.device), train_doc_embs[query_idx].to(self.device)
                )
                
                logits, reg_loss = out
                loss = criterion(logits, train_y[query_idx].to(self.device)) + lambda_reg * reg_loss
                loss.backward()
                optimizer.step()
                
            # Compute epoch metrics
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
                if patience_counter >= self.config.training.patience: 
                    break
                    
        if best_weights:
            # Clean up checkpoint: drop frozen anchor submodule before saving weights
            clean_weights = {k: v for k, v in best_weights.items() if not k.startswith('frozen_doc_pooler.')}
            model.load_state_dict(best_weights)
            torch.save(clean_weights, f"{self.model_save_dir}/best_{save_name}_fold_{fold}.pt")
            
        return self.evaluate_model(
            model, train_tab_emb, train_doc_embs, train_y, 
            val_tab_emb, val_doc_embs, val_y
        )


# =====================================================================
# 2. Pipeline Engine: Image-Only Attention Classifier
# =====================================================================
class ImageOnlyClassifierPipeline:
    def __init__(self, config: ConfigNode):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model_save_dir = config.training.model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

    def evaluate_model(self, model, dataloader):
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for embeddings, labels in dataloader:
                logits = model(embeddings.to(self.device))
                all_labels.extend(labels.numpy())
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        
        return compute_metrics(all_labels, all_probs)

    def train_one_fold(self, train_df, val_df, label_col, fold, save_name):
        train_loader = DataLoader(EmbeddingDataset(train_df, label_col), batch_size=self.config.training.batch_size, shuffle=True)
        val_loader = DataLoader(EmbeddingDataset(val_df, label_col), batch_size=self.config.training.batch_size, shuffle=False)
        
        # Load sample to dynamically detect document dimension
        sample_emb, _ = EmbeddingDataset(train_df, label_col)[0]
        doc_dim = sample_emb.shape[-1]
        
        model = AttentionClassifier(
            input_dim=doc_dim,
            embed_dim=self.config.model.embed_dim,
            num_heads=self.config.model.num_heads
        ).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.training.lr)
        
        best_auroc = 0.0
        best_weights = None
        patience_counter = 0
        
        for epoch in range(self.config.training.epochs):
            model.train()
            for embeddings, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(embeddings.to(self.device)), labels.to(device=self.device))
                loss.backward()
                optimizer.step()
                
            metrics = self.evaluate_model(model, val_loader)
            if metrics["AUROC"] > best_auroc:
                best_auroc = metrics["AUROC"]
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.patience: 
                    break
                    
        if best_weights:
            model.load_state_dict(best_weights)
            torch.save(best_weights, f"{self.model_save_dir}/best_model_label_{save_name}_fold_{fold}.pt")
        return self.evaluate_model(model, val_loader)


# =====================================================================
# 3. Pipeline Engine: Image-Only In-Context Decoder (Image-PFN)
# =====================================================================
class ImagePFNDecoderPipeline:
    def __init__(self, config: ConfigNode):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model_save_dir = config.training.model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)

    def evaluate_model(self, model, supp_doc, supp_y, query_doc, query_y):
        model.eval()
        with torch.no_grad():
            logits = model(supp_doc.to(self.device), supp_y.to(self.device), query_doc.to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return compute_metrics(query_y.numpy(), probs)

    def train_one_fold(self, train_df, val_df, label_col, fold, save_name):
        train_doc_embs = load_all_document_embeddings(train_df)
        val_doc_embs = load_all_document_embeddings(val_df)
        
        train_y = torch.tensor(train_df[label_col].values, dtype=torch.float32)
        val_y = torch.tensor(val_df[label_col].values, dtype=torch.float32)
        
        doc_dim = train_doc_embs.size(-1)
        
        model = ImageOnlyInContextDecoder(
            doc_input_dim=doc_dim,
            embed_dim=self.config.model.embed_dim,
            num_heads=self.config.model.num_heads,
            num_layers=self.config.model.num_layers,
            dropout=self.config.model.dropout
        ).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.training.lr, 
            weight_decay=self.config.training.weight_decay
        )
        
        best_auroc = 0.0
        best_weights = None
        patience_counter = 0
        
        for epoch in range(self.config.training.epochs):
            model.train()
            for step in range(self.config.training.steps_per_epoch):
                optimizer.zero_grad()
                
                # Episodic support vs query sets
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
                
            metrics = self.evaluate_model(model, train_doc_embs, train_y, val_doc_embs, val_y)
            if metrics["AUROC"] > best_auroc:
                best_auroc = metrics["AUROC"]
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.patience: 
                    break
                    
        if best_weights:
            model.load_state_dict(best_weights)
            torch.save(best_weights, f"{self.model_save_dir}/best_model_label_{save_name}_fold_{fold}.pt")
        return self.evaluate_model(model, train_doc_embs, train_y, val_doc_embs, val_y)


# =====================================================================
# Main Orchestrator Entrypoint
# =====================================================================
def run_training(config: ConfigNode):
    logger.info(f"Loading clinical metadata dataset from: {config.data.csv_path}")
    df = pd.read_csv(config.data.csv_path)
    
    # Instantiate the corresponding pipeline
    if config.model_type == "multimodal":
        pipeline = ClinicalMultimodalPipeline(config)
    elif config.model_type == "image_only":
        pipeline = ImageOnlyClassifierPipeline(config)
    elif config.model_type == "image_pfn":
        pipeline = ImagePFNDecoderPipeline(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    summary_results = []
    condition_ons = getattr(config.data, 'condition_ons', None)
    patient_id_col = getattr(config.data, 'patient_id_col', None)

    for i, label in enumerate(config.data.labels):
        sub_df = df.dropna(subset=[label]).copy()

        # Conditional label stratification (applied BEFORE binary filter, matching original)
        if condition_ons is not None and condition_ons[i] is not None:
            save_name = f"{label}_cond_on_{condition_ons[i]}"
            sub_df = sub_df[sub_df[condition_ons[i]] == 1].reset_index(drop=True)
        else:
            save_name = label

        # Binary label filter
        sub_df = sub_df[sub_df[label].isin([0, 1])].reset_index(drop=True)

        logger.info(f"\n>>> Starting 5-Fold CV for Label: {save_name} (N={len(sub_df)})")
        
        if len(sub_df) < 10:
            logger.warning(f"Skipping {save_name} due to insufficient samples.")
            continue

        # Data splitting (patient id grouping vs standard stratification)
        splits, sub_df_clean = get_stratified_kfold_splits(
            df=sub_df, 
            label_col=label, 
            group_col=patient_id_col, 
            n_splits=config.training.k_folds
        )

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"  --- Fold {fold+1}/{config.training.k_folds} ---")
            train_df, val_df = sub_df_clean.iloc[train_idx], sub_df_clean.iloc[val_idx]
            
            if len(train_df[label].unique()) < 2:
                logger.warning("  Skipping fold due to single-class representation in train set.")
                continue

            metrics = pipeline.train_one_fold(train_df, val_df, label, fold, save_name)
            fold_metrics.append(metrics)
            logger.info(f"      Val AUROC: {metrics['AUROC']:.4f} | F1: {metrics['F1']:.4f} | Acc: {metrics['Acc']:.4f}")

        # Consolidate results for target label
        if fold_metrics:
            avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
            std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
            
            res = {"Label": save_name}
            for k in avg_metrics:
                res[f"{k}_mean"] = avg_metrics[k]
                res[f"{k}_std"] = std_metrics[k]
            summary_results.append(res)

    # Consolidated output table logging
    logger.info("\n" + "="*70)
    logger.info("FINAL CONSOLIDATED 5-FOLD CV PIPELINE REPORT")
    logger.info("="*70)
    report_df = pd.DataFrame(summary_results)
    if not report_df.empty:
        for m in ["AUROC", "AUPRC", "Acc", "F1"]:
            report_df[m] = report_df.apply(lambda x: f"{x[m+'_mean']:.3f} ± {x[m+'_std']:.3f}", axis=1)
        
        logger.info("\n" + report_df[["Label", "AUROC", "AUPRC", "Acc", "F1"]].to_string(index=False))
        
        # Save validation results
        log_save_path = getattr(config.training, 'log_save_path', None)
        if log_save_path:
            os.makedirs(os.path.dirname(log_save_path), exist_ok=True)
            report_df.to_csv(log_save_path, index=False)
            logger.info(f"\nConsolidated logs saved to: {log_save_path}")
    else:
        logger.warning("No labels were successfully processed.")
