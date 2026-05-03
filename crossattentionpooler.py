import torch
import torch.nn as nn

class MLP(nn.Module):
    """Native PyTorch implementation of the timm Mlp"""
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
    def __init__(
        self,
        input_dim: int = 128,
        embed_dim: int = 128,
        num_heads: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = int(embed_dim) 
        self.decoder_attn = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True, dropout=0.15)

        self.decoder_ffn = MLP(
            in_features=self.embed_dim, 
            hidden_features=int(self.embed_dim * 4), 
            out_features=self.embed_dim, 
            drop=0.15
        )
        
        self.learned_agg = nn.Parameter(torch.randn(1, 1, self.embed_dim), requires_grad=True)
    
    def forward(self, x):
        """
        Args:
            x: [B, num_tokens, embed_dim]
        Returns:
            [B, embed_dim] pooled representation
        """
        decoder_queries = self.learned_agg.repeat(x.shape[0], 1, 1)

        attn_out, _ = self.decoder_attn(query=decoder_queries, key=x, value=x)
        
        x = attn_out[:, 0, :]
        x = self.decoder_ffn(x)
        
        return x