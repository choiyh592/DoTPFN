import torch
import torch.nn as nn

class TabPFNInContextDecoder(nn.Module):
    """Implements Bayesian-like in-context episodic inference on joint embeddings.
    Test queries cross-attend to the labeled support set (train instances) 
    to yield predictive posteriors conditionally independent of other queries.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
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

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_x: (B, N_support, D) - Train joint embeddings
            support_y: (B, N_support)    - Train labels
            query_x:   (B, N_query, D)   - Test joint embeddings
        Returns:
            Logits: (B, N_query)
        """
        # Embed train labels and fuse into support features
        supp_y_emb = self.label_embedding(support_y.long())
        support_features = support_x + supp_y_emb
        
        # Concatenate sequences for the transformer: (B, N_support + N_query, D)
        combined_x = torch.cat([support_features, query_x], dim=1)
        
        B, N_supp, D = support_x.shape
        N_query = query_x.size(1)
        N_tot = N_supp + N_query
        
        # Build strict Causal Mask for Bayesian Inference:
        # 1. Support attends to Support
        # 2. Query attends to Support
        # 3. Query CANNOT attend to other Queries (conditional independence)
        mask = torch.zeros(N_tot, N_tot, dtype=torch.bool, device=support_x.device)
        mask[:, N_supp:] = True
        mask.diagonal().fill_(False) # Queries can still attend to themselves
        
        # Contextual inference through the transformer stack
        out = self.transformer(combined_x, mask=mask)
        
        # Extract contextualized query outputs and map to logits
        query_out = out[:, N_supp:, :]
        return self.classifier(query_out).squeeze(-1)
