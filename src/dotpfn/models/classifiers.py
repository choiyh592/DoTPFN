import torch
import torch.nn as nn
from .cross_attention import CrossAttentionWithLearnedQueries
from .in_context import TabPFNInContextDecoder

class AttentionClassifier(nn.Module):
    """Image/Document-only classifier that pools embeddings via cross-attention 
    and applies a linear classification head.
    """
    def __init__(self, input_dim: int = 128, embed_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.pooler = CrossAttentionWithLearnedQueries(
            input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, num_queries=1
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_features = self.pooler(x)
        logits = self.classifier(pooled_features)
        return logits.squeeze(-1)


class ImageOnlyInContextDecoder(nn.Module):
    """Image/Document-only classifier that uses episodic causal Bayesian in-context learning 
    over support set embeddings.
    """
    def __init__(
        self,
        doc_input_dim: int = 128,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.pooler = CrossAttentionWithLearnedQueries(
            input_dim=doc_input_dim, embed_dim=embed_dim, num_heads=num_heads, num_queries=1
        )
        self.decoder = TabPFNInContextDecoder(
            embed_dim=embed_dim, num_heads=min(4, num_heads), num_layers=num_layers, dropout=dropout
        )

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        support_pooled = self.pooler(support_x).unsqueeze(0)  # (1, N_supp, D)
        query_pooled = self.pooler(query_x).unsqueeze(0)      # (1, N_query, D)
        support_y = support_y.unsqueeze(0)                    # (1, N_supp)
        
        logits = self.decoder(support_pooled, support_y, query_pooled)
        return logits.squeeze(0)  # (N_query)
