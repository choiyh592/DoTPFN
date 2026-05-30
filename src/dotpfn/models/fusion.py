import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .cross_attention import CrossAttentionWithLearnedQueries
from .in_context import TabPFNInContextDecoder

class DoTPFN(nn.Module):
    """End-to-End MMPFN / DoTPFN model: Projects image/doc data and TabPFN embeddings
    into a shared domain, fuses them, and executes Bayesian-like in-context decoding.
    """
    def __init__(
        self,
        tab_embed_dim: int,
        doc_input_dim: int = 128,
        doc_embed_dim: int = 128,
        doc_num_heads: int = 8,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
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
        
        # Explainability Anchor: populated dynamically during training to regularize feature drift
        self.frozen_doc_pooler = None
        
        # Align joint dimensions (d_model from tab + doc_embed_dim from image) -> d_model
        self.fusion = nn.Sequential(
            nn.Linear(d_model + doc_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.decoder = TabPFNInContextDecoder(embed_dim=d_model, num_heads=num_heads)

    def forward_features(self, tab_emb: torch.Tensor, doc_emb: torch.Tensor):
        tab_proj = self.tab_proj(tab_emb)
        
        # If an explainability anchor is present, track attention drift
        if self.frozen_doc_pooler is not None:
            doc_pooled, attn_after = self.doc_pooler(doc_emb, return_attn=True)
            with torch.no_grad():
                _, attn_before = self.frozen_doc_pooler(doc_emb, return_attn=True)
            # Alignment regularization loss
            reg_loss = F.mse_loss(attn_after, attn_before)
        else:
            doc_pooled = self.doc_pooler(doc_emb)
            reg_loss = torch.tensor(0.0, device=tab_emb.device)
            
        joint_embs = self.fusion(torch.cat([tab_proj, doc_pooled], dim=-1))
        return joint_embs, reg_loss

    def forward(self, support_tab: torch.Tensor, support_doc: torch.Tensor, support_y: torch.Tensor, query_tab: torch.Tensor, query_doc: torch.Tensor):
        # Extract fused token embeddings and track attention penalties
        support_embs, supp_reg = self.forward_features(support_tab, support_doc)
        query_embs, query_reg = self.forward_features(query_tab, query_doc)
        
        support_embs = support_embs.unsqueeze(0)  # (1, N_supp, D)
        query_embs = query_embs.unsqueeze(0)      # (1, N_query, D)
        support_y = support_y.unsqueeze(0)        # (1, N_supp)
        
        # TabPFN Bayesian Inference
        logits = self.decoder(support_embs, support_y, query_embs)
        
        total_reg_loss = supp_reg + query_reg
        return logits.squeeze(0), total_reg_loss  # (N_query), scalar loss tensor
