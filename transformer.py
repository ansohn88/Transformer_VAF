from typing import Optional

import torch
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from attend import Attend
from simpool import SimPool


def exists(val):
    return val is not None

class RevIN(nn.Module):
    def __init__(self, num_variates, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        self.gamma = nn.Parameter(torch.ones(num_variates, 1))
        self.beta = nn.Parameter(torch.zeros(num_variates, 1))

    def forward(self, x, return_statistics=False):
        assert x.shape[1] == self.num_variates

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        var_rsqrt = var.clamp(min=self.eps).rsqrt()
        instance_normalized = (x - mean) * var_rsqrt
        rescaled = instance_normalized * self.gamma + self.beta

        def reverse_fn(scaled_output):
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(
                min=self.eps
            )
            unscaled_output = (scaled_output - self.beta) / clamped_gamma
            return unscaled_output * var.sqrt() + mean

        if not return_statistics:
            return rescaled, reverse_fn

        statistics = Statistics(mean, var, self.gamma, self.beta)

        return rescaled, reverse_fn, statistics


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = rearrange(x, "... (r d) -> r ... d", r=2)
        return x * F.gelu(gate)


def FeedForward(dim, mult=4, dropout=0.0):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GEGLU(),
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

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, dim_inner, bias=False),
            nn.SiLU(),
            Rearrange("b n (h d) -> b h n d", h=heads),
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
        out = out * self.to_v_gates(x
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
        use_reversible_instance_norm: bool = False,
        simpool: bool = False,
        gamma: Optional[float] = None,
        use_beta: bool = False
    ) -> None:
        super().__init__()

        self.mlp_in = nn.Sequential(
            nn.Linear(num_variates, dim * num_tokens_per_variate),
            Rearrange("b v (n d) -> b (v n) d", n=num_tokens_per_variate),
            nn.LayerNorm(dim),
        )

        self.register_tokens = (
            nn.Parameter(torch.randn(num_register_tokens, dim))
            if num_register_tokens > 0
            else None
        )

        self.reversible_instance_norm = (
            RevIN(num_variates) if use_reversible_instance_norm else None
        )

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

        has_mem = exists(self.register_tokens)

        if exists(self.reversible_instance_norm):
            x, reverse_fn = self.reversible_instance_norm(x)

        x = self.mlp_in(x)

        if has_mem:
            r = repeat(self.register_tokens, "n d -> b n d", b=batch)
            x, ps = pack([x, r], "b * d")

        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)

        if has_mem:
            x, _ = unpack(x, ps, "b * d")

        if exists(self.reversible_instance_norm):
            x = reverse_fn(x)

        if self.simpool:
            x = self.pool(x)
        else:
            x = x.mean(dim=1)

        x = self.to_latent(x)

        return self.to_out(x)
