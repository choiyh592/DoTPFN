import os
import torch
import numpy as np
import pandas as pd
from dotpfn.utils.logging import setup_logger
from dotpfn.utils.config import ConfigNode
from dotpfn.data.dataset import load_all_document_embeddings
from dotpfn.models.fusion import DoTPFN
from dotpfn.models.classifiers import AttentionClassifier, ImageOnlyInContextDecoder
from dotpfn.utils.tabpfn_loader import get_tabpfn_classes

logger = setup_logger("DoTPFN.Infer")

def run_inference(config: ConfigNode):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model_type = config.model_type
    
    logger.info(f"Starting inference pipeline for model_type: {model_type}")
    
    # Load query set (the new patients to predict on)
    logger.info(f"Loading query CSV from {config.data.query_csv_path}")
    query_df = pd.read_csv(config.data.query_csv_path)
    if len(query_df) == 0:
        logger.error("Query CSV is empty!")
        return
        
    query_doc_embs = load_all_document_embeddings(query_df)

    if model_type in ["multimodal", "image_pfn"]:
        # Need support set for in-context learning
        logger.info(f"Loading support CSV from {config.data.support_csv_path}")
        support_df = pd.read_csv(config.data.support_csv_path)
        support_df = support_df.dropna(subset=[config.data.target_label])
        support_df = support_df[support_df[config.data.target_label].isin([0, 1])].reset_index(drop=True)
        
        support_doc_embs = load_all_document_embeddings(support_df)
        support_y = torch.tensor(support_df[config.data.target_label].values, dtype=torch.float32)
        
    # Load model weights
    weights_path = config.inference.model_weights_path
    logger.info(f"Loading model weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location=device) if os.path.exists(weights_path) else None
    if state_dict is None:
        logger.warning(f"Weights file not found at {weights_path}. Initializing with random weights.")

    probs = []

    if model_type == "multimodal":
        X_train = support_df[config.data.feature_cols].values
        y_train = support_df[config.data.target_label].values
        X_query = query_df[config.data.feature_cols].values
        
        tabpfn_version = getattr(config.tabpfn, "version", "auto") if hasattr(config, "tabpfn") else "auto"
        TabPFNClassifier, TabPFNEmbedding = get_tabpfn_classes(version=tabpfn_version)
        
        logger.info("Extracting TabPFN embeddings...")
        clf = TabPFNClassifier(n_estimators=getattr(config.tabpfn, "n_estimators", 1), device=device)
        clf.fit(X_train, y_train)
        
        embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=5)
        embedder.fit(X_train, y_train)
        
        support_tab_emb = embedder.get_embeddings(X_train, y_train, X_train, data_source="train")
        query_tab_emb = embedder.get_embeddings(X_train, y_train, X_query, data_source="test")
        
        support_tab_emb = torch.tensor(support_tab_emb, dtype=torch.float32)
        query_tab_emb = torch.tensor(query_tab_emb, dtype=torch.float32)
        
        if support_tab_emb.dim() == 3:
            support_tab_emb = support_tab_emb.squeeze(0) if support_tab_emb.size(0) == 1 else support_tab_emb.squeeze(1)
            query_tab_emb = query_tab_emb.squeeze(0) if query_tab_emb.size(0) == 1 else query_tab_emb.squeeze(1)

        tab_dim = support_tab_emb.shape[1]
        doc_dim = support_doc_embs.shape[-1]
        
        model = DoTPFN(
            tab_embed_dim=tab_dim,
            doc_input_dim=doc_dim,
            doc_embed_dim=config.model.doc_embed_dim,
            doc_num_heads=config.model.doc_num_heads,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout
        ).to(device)
        
        if state_dict:
            model.load_state_dict(state_dict)
        model.eval()
        
        with torch.no_grad():
            logits, _ = model(
                support_tab_emb.to(device), support_doc_embs.to(device), support_y.to(device),
                query_tab_emb.to(device), query_doc_embs.to(device)
            )
            probs = torch.sigmoid(logits).cpu().numpy()

    elif model_type == "image_only":
        doc_dim = query_doc_embs.shape[-1]
        model = AttentionClassifier(
            input_dim=doc_dim,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads
        ).to(device)
        
        if state_dict:
            model.load_state_dict(state_dict)
        model.eval()
        
        all_probs = []
        batch_size = getattr(config.inference, "batch_size", 32)
        with torch.no_grad():
            for i in range(0, len(query_doc_embs), batch_size):
                batch_docs = query_doc_embs[i:i+batch_size].to(device)
                logits = model(batch_docs)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        probs = np.array(all_probs)

    elif model_type == "image_pfn":
        doc_dim = query_doc_embs.shape[-1]
        model = ImageOnlyInContextDecoder(
            doc_input_dim=doc_dim,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        ).to(device)
        
        if state_dict:
            model.load_state_dict(state_dict)
        model.eval()
        
        with torch.no_grad():
            logits = model(support_doc_embs.to(device), support_y.to(device), query_doc_embs.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()

    else:
        logger.error(f"Unknown model type: {model_type}")
        return

    # Ensure scalar prediction returns match length
    if np.ndim(probs) == 0:
        probs = np.array([probs])

    query_df["probability"] = probs
    query_df["prediction"] = (query_df["probability"] >= getattr(config.inference, "threshold", 0.5)).astype(int)
    
    out_path = config.inference.output_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    # Save a clean output with just ID, probability, and prediction
    out_cols = [config.data.patient_id_col, "probability", "prediction"]
    query_df[out_cols].to_csv(out_path, index=False)
    logger.info(f"Inference complete! Results saved to {out_path}")
