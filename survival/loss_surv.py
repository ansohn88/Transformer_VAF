from typing import Optional

import torch
import torch.nn as nn


def loss_reg_l1(model, coef):
    # l1_reg = coef * sum([torch.abs(W).sum() for W in model.parameters()])
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            # torch.abs(W).sum() is equivalent to W.norm(1)
            l1_reg = l1_reg + torch.abs(W).sum()

    l1_reg = coef * l1_reg
    return l1_reg


class SurvLoss(nn.Module):

    def __init__(
            self,
            alpha: float = 0.0,
            eps: float = 1e-7,
    ) -> None:
        super(SurvLoss, self).__init__()

        self.alpha = alpha
        self.eps = eps

    def forward(
            self,
            hazards: torch.Tensor,
            survival: torch.Tensor,
            d_lbl: torch.Tensor,
            event: int,
            cur_alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        c = 0 for uncensored samples (with event, event = 1), 
        c = 1 for censored samples (without event, event = 0).
        """
        d_lbl = d_lbl.long()
        event = event.squeeze(0).long()

        if event == 1:
            c = event - 1
        elif event == 0:
            c = event + 1
        else:
            raise ValueError("Event in ranges [0, 1]")

        S_padded = torch.cat(
            [torch.ones_like(c), survival],
            dim=1
        )

        s_prev = torch.gather(S_padded, dim=1, index=d_lbl).clamp(min=self.eps)
        h_this = torch.gather(hazards, dim=1, index=d_lbl).clamp(min=self.eps)
        s_this = torch.gather(
            S_padded, dim=1, index=d_lbl+1).clamp(min=self.eps)

        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = - c * torch.log(s_this)

        neg_l = censored_loss + uncensored_loss
        alpha = self.alpha if cur_alpha is None else cur_alpha
        loss = ((1.0 - alpha) * neg_l) + (alpha * uncensored_loss)
        loss = loss.mean()
        return loss
