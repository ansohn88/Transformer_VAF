import numpy as np
import torch
from torch import Tensor

# Helper Functions


def _calculate_distribution(
    loc: torch.Tensor,
    var: torch.Tensor,
    tmax: int,
    distribution_fn,
    return_prob: bool = False,
) -> torch.Tensor:
    """
    Helper function to get either log probabilities or probabilities.

    Args:
        loc (Tensor): Location parameter tensor.
        var (Tensor): Variance parameter tensor.
        tmax (int)  : Maximum time.
        distribution_fn (function): Distribution function to use.
        return_prob (bool, optional): Flag to return probabilities.
                                     Defaults to False.

    Returns:
        Tensor: Log probabilities or probabilities.
    """
    loc = loc.view(-1, 1)
    var = var.view(-1, 1)
    return distribution_fn(loc, var, tmax, return_prob)


# Distribution Functions


def discretized_logistic(
    loc: Tensor, var: Tensor, tmax: int, return_prob: bool = False
) -> Tensor:
    """
    Calculate discretized logistic distribution.

    Args:
        loc (Tensor): Location parameter tensor, shape (batch_size, 1).
        var (Tensor): Variance parameter tensor, shape (batch_size, 1).
        tmax (int): Maximum time.
        return_prob (bool, optional): Flag to return probabilities.
                                      Defaults to False.

    Returns:
        Tensor: Log probabilities or probabilities.
    """
    scale = torch.sqrt(var)
    trange = torch.arange(1, float(tmax) + 1, device=loc.device)
    probs = torch.sigmoid((trange + 0.5 - loc) / scale) - torch.sigmoid(
        (trange - 0.5 - loc) / scale
    )
    probs /= probs.sum(dim=1, keepdim=True)
    return torch.log(probs + 1e-8) if not return_prob else probs


def discretized_gaussian(
    loc: Tensor, var: Tensor, tmax: int, return_prob: bool = False
) -> Tensor:
    """
    Calculate discretized Gaussian distribution.

    Args:
        loc (Tensor): Location parameter tensor, shape (batch_size, 1).
        var (Tensor): Variance parameter tensor, shape (batch_size, 1).
        tmax (int): Maximum time.
        return_prob (bool, optional): Flag to return probabilities.
                                      Defaults to False.

    Returns:
        Tensor: Log probabilities or probabilities.
    """
    trange = torch.arange(1, float(tmax) + 1, device=loc.device)
    log_probs = -((trange - loc) ** 2) / (2 * var + 1e-8)
    log_probs -= torch.logsumexp(log_probs, dim=1, keepdim=True)
    return log_probs if not return_prob else torch.exp(log_probs)


# Utility Functions


def get_survival_function(prob: Tensor) -> np.ndarray:
    """
    Calculate survival function.

    Args:
        p (Tensor): Probability tensor, shape (batch_size, tmax).

    Returns:
        np.ndarray: Survival function values.
    """
    prob = prob.data.cpu().numpy()
    return np.cumsum(prob[:, ::-1], axis=1)[:, ::-1]


def get_mean_prediction(prob: Tensor, tmax: int) -> Tensor:
    """
    Calculate mean prediction.

    Args:
        p (Tensor): Probability tensor, shape (batch_size, tmax).
        tmax (int): Maximum time.

    Returns:
        Tensor: Mean prediction values.
    """
    time_range = torch.arange(1, tmax + 1, device=prob.device)
    return (time_range * prob).sum(dim=1).data


def get_probs(
    loc: Tensor,
    variance: Tensor,
    tmax: int,
    distribution: str = "discretized_gaussian",
) -> Tensor:
    """
    Get probabilities based on the distribution.

    Args:
        loc (Tensor): loc parameter tensor, shape (batch_size, 1).
        variance (Tensor): Variance parameter tensor, shape (batch_size, 1).
        tmax (int): Maximum time.
        distribution (str, optional): Type of distribution to use.
                                      Defaults to 'discretized_gaussian'.

    Returns:
        Tensor: Probabilities.
    """
    distribution_fn = (
        discretized_gaussian
        if distribution == "discretized_gaussian"
        else discretized_logistic
    )
    return _calculate_distribution(
        loc, variance, tmax, distribution_fn, return_prob=True
    )


# Loss functions


def centime_loss(
    loc: Tensor,
    var: Tensor,
    delta: Tensor,
    time: Tensor,
    tmax: int,
    distribution: str = "discretized_gaussian",
) -> (Tensor, Tensor):
    """
    Compute the centime loss for survival analysis.

    Args:
        loc (Tensor): Location parameter tensor, shape (batch_size, 1).
        var (Tensor): Variance parameter tensor, shape (batch_size, 1).
        delta (Tensor): Event indicator tensor, shape (batch_size, 1).
        time (Tensor): Time tensor, shape (batch_size, 1). Event time if uncensored,
                                    censoring time if censored.
        tmax (int): Maximum time.
        distribution (str, optional): Distribution to use.
                                    Defaults to 'discretized_gaussian'.

    Returns:
        Tuple[Tensor, Tensor]: Loss for censored and uncensored data.
    """
    time = time.view(-1)
    delta = delta.view(-1).type(torch.bool)
    uncens_time = time[delta].view(-1, 1)
    cens_time = time[~delta].view(-1, 1)

    logp = _calculate_distribution(
        loc,
        var,
        tmax,
        discretized_gaussian
        if distribution == "discretized_gaussian"
        else discretized_logistic,
    )
    logp_cens = logp[~delta]
    logp_uncens = logp[delta]
    loss_uncens = logp_uncens.gather(
        1, uncens_time - 1
    ).sum()  # Adjust for 0-based indexing

    loss_cens = 0
    for c_time, _logp in zip(cens_time, logp_cens):
        factor = torch.arange(c_time[0], float(tmax), device=loc.device).log()
        loss_cens += torch.logsumexp(_logp[c_time:] - factor, 0)
    loss_cens = -loss_cens / len(time)

    loss_uncens = -loss_uncens / len(time)
    return loss_cens, loss_uncens
