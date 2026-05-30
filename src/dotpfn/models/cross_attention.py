import torch
import torch.nn as nn

class MLP(nn.Module):
    """Native PyTorch implementation of a Multi-Layer Perceptron (timm equivalent)."""
    def __init__(self, in_features, hidden_features, out_features, drop=0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CrossAttentionWithLearnedQueries(nn.Module):
    """Acts as the Cross-Attention Pooler (CAP).
    Compresses variable-length or high-dimensional document/image embeddings 
    into a compact, fixed-size representation using learned query tokens.
    """
    def __init__(
        self,
        input_dim: int = 128,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_queries: int = 1
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = int(embed_dim)
        
        # Learnable queries to extract salient information from the document
        self.queries = nn.Parameter(torch.randn(1, num_queries, self.embed_dim))
        
        self.proj = nn.Linear(input_dim, self.embed_dim) if input_dim != self.embed_dim else nn.Identity()
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True)
        
        # Store the attention weights for spatial explanations and Grad-CAM
        self.last_attn_weights = None

    def forward(self, x, return_attn=False):
        """
        Args:
            x: [B, seq_len, input_dim] or [B, input_dim] (which gets unsqueezed to sequence size of 1)
            return_attn: If True, returns both pooled representation and raw attention weights.
        Returns:
            Pooled feature tensor: [B, num_queries * embed_dim]
        """
        # Ensure sequence dimension exists: (Batch, Seq_Len, Dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x_proj = self.proj(x)
        batch_size = x_proj.size(0)
        
        # Expand learned query tokens for the entire batch
        q = self.queries.expand(batch_size, -1, -1)
        
        # Cross-attention: Queries attend to Document key-value pairs
        attn_out, attn_weights = self.mha(query=q, key=x_proj, value=x_proj)
        
        # Store attention weights for explainers (Shape: batch_size, num_queries, seq_len)
        self.last_attn_weights = attn_weights
        
        # Flatten pooled tokens: (Batch, num_queries * embed_dim)
        out = attn_out.reshape(batch_size, -1)
        
        if return_attn:
            return out, attn_weights
        return out
