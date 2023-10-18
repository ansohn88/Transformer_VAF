from typing import Optional

import torch
from einops import pack, repeat, unpack
from einops.layers.torch import Rearrange
from torch import nn

from attend import Attend
from simpool import SimPool


def exists(val):
    return val is not None


# class ScaledSinusoidalEmbedding(nn.Module):
#     def __init__(self, dim, theta=10000):
#         super().__init__()
#         assert (dim % 2) == 0
#         self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)

#         half_dim = dim // 2
#         freq_seq = torch.arange(half_dim).float() / half_dim
#         inv_freq = theta**-freq_seq
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#     def forward(self, x, pos=None):
#         seq_len, device = x.shape[1], x.device

#         if not exists(pos):
#             pos = torch.arange(seq_len, device=device)

#         emb = torch.einsum("i, j -> i j", pos, self.inv_freq)
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb * self.scale


def FeedForward(dim, mult=4, dropout=0.0):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim),
    )


class Attention(nn.Module):
    def __init__(self, dim, dim_head, heads, dropout=0.0, flash=True):
        super().__init__()
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=heads),
        )
        self.attend = Attend(flash=flash, dropout=dropout)

        self.to_out = nn.Sequential(
            Rearrange("b h n d -> b n (h d)"),
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x)
        out = self.attend(q, k, v)
        return self.to_out(out)


class iTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_variates: int,
        depth: int,
        dim: int,
        num_tokens_per_variate: int,
        num_register_tokens: int,
        dim_head: int,
        heads: int,
        attn_dropout: float,
        ff_mult: int,
        ff_dropout: float,
        out_dim: int = 2,
        flash_attn: bool = True,
        simpool: bool = False,
        gamma: Optional[float] = None,
        use_beta: bool = False
    ) -> None:
        super().__init__()

        self.mlp_in = nn.Sequential(
            nn.Linear(num_variates, dim * num_tokens_per_variate),
            nn.LayerNorm(dim),
            Rearrange("b v (n d) -> b (v n) d", n=num_tokens_per_variate),
            nn.LayerNorm(dim),
        )

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            flash=flash_attn,
                        ),
                        nn.LayerNorm(dim),
                        FeedForward(dim, mult=ff_mult, dropout=ff_dropout),
                        nn.LayerNorm(dim),
                    ]
                )
            )

        self.simpool = simpool
        if simpool:
            self.pool = SimPool(
                dim=dim,
                num_heads=heads,
                qkv_bias=False,
                qk_scale=None,
                gamma=gamma,
                use_beta=use_beta,
            )
        self.to_latent = nn.Identity()

        self.to_out = nn.Linear(dim, out_dim)

    def forward(self, x):
        # batch, device = x.shape[0], x.device
        batch = x.shape[0]

        x = self.mlp_in(x)

        r = repeat(self.register_tokens, "n d -> b n d", b=batch)

        x, ps = pack([x, r], "b * d")

        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)

        x, _ = unpack(x, ps, "b * d")

        if self.simpool:
            x = self.pool(x)
        else:
            x = x.mean(dim=1)

        x = self.to_latent(x)

        return self.to_out(x)
