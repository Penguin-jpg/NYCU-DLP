import torch.nn as nn
import torch
import math


# TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.model_dim = dim
        self.head_dim = dim // num_heads

        # weight matrix for Q, K, V
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        # scale and softmax for calculate self-attention
        self.scale = 1 / math.sqrt(self.head_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_weights_dropout = nn.Dropout(attn_drop)

        # weight matrix for output
        self.W_o = nn.Linear(dim, dim)

    def forward(self, x):
        """Hint: input x tensor shape is (batch_size, num_image_tokens, dim),
        because the bidirectional transformer first will embed each token to dim dimension,
        and then pass to n_layers of encoders consist of Multi-Head Attention and MLP.
        # of head set 16
        Total d_k , d_v set to 768
        d_k , d_v for one head will be 768//16.
        """
        # batch_size and num_tokens
        B, N = x.shape[:2]

        # apply transformations (output shape: [batch_size, num_tokens, dim])
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # split the heads (reshape to [batch_size, num_heads, num_tokens, head_dim])
        q = q.view(B, self.num_heads, N, self.head_dim)
        k = k.view(B, self.num_heads, N, self.head_dim)
        v = v.view(B, self.num_heads, N, self.head_dim)

        # self-attention can be decomposed into:
        # 1. scores = qk^T / sqrt{head_dim}
        # 2. weights = softmax(scores)
        # 3. weights = dropout(scores)
        # 4. attention = weights * v
        # (note that everything is matrix multiplication)

        # tranpose the last two dimensions of k to perform matrix multiplication
        # output shape: [batch_size, num_heads, num_tokens, num_tokens]
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale

        # use softmax to get probabilities as weights and apply dropout
        # output shape: [batch_size, num_heads, num_tokens, num_tokens]
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.attention_weights_dropout(attention_weights)

        # reshape to [batch_size, num_tokens, num_heads, head_dim]
        attention = torch.matmul(attention_weights, v).view(
            B, N, self.num_heads, self.head_dim
        )

        # concate results from all heads (output shape: [batch_size, num_tokens, dim])
        attention = attention.view(B, N, self.model_dim)

        # project the output
        out = self.W_o(attention)

        return out


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1),
        )

    def forward(self, input):
        return super().forward(input)


class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12),
        )

    def forward(self, input):
        return super().forward(input)


class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)

        x = x + attn
        x = self.LayerNorm1(x)

        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)


# if __name__ == "__main__":
#     mh = MultiHeadAttention(768, 16, 0.0)
#     x = torch.randn([1, 16, 768])
#     print(f"before attention: {x.shape}")
#     out = mh(x)
#     print(f"after attention: {out.shape}")
