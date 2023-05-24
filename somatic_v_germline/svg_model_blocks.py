from collections import namedtuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

Config = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def exists(val):
    return val is not None


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


class Attend(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        return self.flash_attn(q, k, v)


# class RMSNorm(nn.Module):
#     def __init__(self, dim, eps=1e-08) -> None:
#         super().__init__()
#         self.scale = dim**-0.5
#         self.eps = eps
#         self.g = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
#         return x / norm.clamp(min=self.eps) * self.g


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        # self.norm = RMSNorm(dim)

        self.attend = Attend()

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
    ) -> None:
        super().__init__()
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
        return x


class SimpleViT(nn.Module):
    def __init__(
        self, *, input_dim, attn_dim, num_classes, depth, heads, dim_head, mlp_dim
    ) -> None:
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, attn_dim, bias=True),
            nn.LayerNorm(attn_dim),
        )
        self.pe = ScaledSinusoidalEmbedding(dim=attn_dim)

        self.transformer = Transformer(
            dim=attn_dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim
        )

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, num_classes),
        )

    def forward(self, x):
        x = self.to_embedding(x)
        x = x + self.pe(x)

        x = self.transformer(x)
        x = self.to_latent(x)

        return self.linear_head(x)
