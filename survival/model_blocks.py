import math
from collections import namedtuple

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

####################################################################################

############################# Attention Modules ####################################

####################################################################################

Config = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def exists(val):
    return val is not None


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
        # self.scale = dim_head**-0.5
        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)

        self.attend = Attend()

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.pre_norm(x)

        # project for queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # # scale
        # q = q * self.scale

        out = self.attend(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.post_norm(out)


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(
            rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1")
        )
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer("bias", bias, persistent=False)

        return self.bias


class LearnedAlibiPositionalBias(AlibiPositionalBias):
    def __init__(self, heads, total_heads):
        super().__init__(heads, total_heads)
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def forward(self, i, j):
        h, device = self.heads, self.device

        def get_slopes(param):
            return pad_at_dim(param.exp(), (0, h - param.shape[0]), dim=-2)

        if exists(self.bias) and self.bias.shape[-1] >= j:
            bias = self.bias[..., :i, :j]
        else:
            bias = self.get_bias(i, j, device)
            self.register_buffer("bias", bias, persistent=False)

        slopes = get_slopes(self.learned_logslopes)
        bias = bias * slopes

        return bias


####################################################################################

############################# Conformer Modules ####################################

####################################################################################


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class ConformerConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.2
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


####################################################################################

############################# Position Encodings ###################################

####################################################################################


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


####################################################################################

################################ FeedForward #######################################

####################################################################################


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.net(x)


####################################################################################

################################ Simple ViT #######################################

####################################################################################


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
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)


####################################################################################

################################ Conformer #######################################

####################################################################################


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head,
        heads,
        mlp_dim,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        ff_dropout=0.2,
        conv_dropout=0.2,
        conv_causal=False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=ff_dropout)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head)
        self.conv = ConformerConvModule(
            dim=dim,
            causal=conv_causal,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=ff_dropout)

    def forward(self, x):
        x = self.ff1(x) + x
        x = self.attn(x) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        return x


class Conformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head,
        heads,
        mlp_dim,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        conv_causal=False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ConformerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    conv_causal=conv_causal,
                )
            )

    def forward(self, x):
        for block in self.layers:
            x = block(x)

        return x


class ConformerModel(nn.Module):
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

        self.conformer = Conformer(
            dim=attn_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
        )

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, num_classes),
        )

    def forward(self, x):
        x = self.to_embedding(x)
        x = x + self.pe(x)

        x = self.conformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
