import torch
from einops import pack, rearrange, repeat, unpack
from torch import nn


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

        self.input_dim = num_numerical_types
        self.output_dim = int(dim * num_numerical_types)

    def forward(self, x):
        x = nn.LayerNorm(self.input_dim)(x)
        x = rearrange(x, "b n -> b n 1")
        x = x * self.weights + self.biases
        x = x.flatten(start_dim=1, end_dim=2)
        return nn.LayerNorm(self.output_dim)(x)


def exists(val):
    return val is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta**-freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        emb = torch.einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        attn_dim,
        num_register_tokens,
        num_classes,
        theta,
        depth,
        heads,
        dim_head,
        mlp_dim,
    ) -> None:
        super().__init__()

        self.to_embedding = NumericalEmbedder(
            dim=input_dim, num_numerical_types=attn_dim
        )

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, attn_dim))

        self.pe = ScaledSinusoidalEmbedding(dim=attn_dim, theta=theta)

        self.transformer = Transformer(
            dim=attn_dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim
        )

        self.pool = TODO
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(attn_dim, num_classes)

    def forward(self, x):
        x = self.to_embedding(x)
        x = x + self.pe(x)

        r = repeat(self.register_tokens, "n d -> b n d", b=batch)
        x, ps = pack([x, r], "b * d")

        x = self.transformer(x)
        x, _ = unpack(x, ps, "b * d")

        x = self.pool(x)
        # x = x.mean(dim=1)

        x = self.to_latent(x)
        x = self.linear_head(x)

        # out = torch.cumprod(torch.sigmoid(x), dim=-1)
        return x
