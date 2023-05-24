import torch
from svg_model_blocks import SimpleViT
from timm.models.layers import trunc_normal_
from torch import nn

###############################################################################

# ViT-Based Model #

##############################################################################


class AttnModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        attn_dim: int,
        num_heads: int,
        depth: int,
    ) -> None:
        super().__init__()

        mlp_dim = int(4 * attn_dim)
        self.model = SimpleViT(
            input_dim=input_dim,
            attn_dim=attn_dim,
            num_classes=out_dim,
            depth=depth,
            heads=num_heads,
            dim_head=int(attn_dim // num_heads),
            mlp_dim=mlp_dim,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
