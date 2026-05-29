import torch
from torch import Tensor


def stable_softmax(logits: Tensor) -> Tensor:
    """
    Numerically stable softmax along the last dimension.
    Args:
        logits: Tensor of shape (batch_size, num_classes)
    Returns:
        Softmax probabilities, same shape as logits.
    """
    # Max for numerical stability; shape: (batch_size, 1)
    max_logits, _ = logits.max(dim=-1, keepdim=True)
    # Shift logits; shape: (batch_size, num_classes)
    shifted = logits - max_logits
    # Exponentiate and sum; shapes: (batch_size, num_classes), (batch_size, 1)
    exp_shifted = torch.exp(shifted)
    sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
    # Normalize
    return exp_shifted / sum_exp


def label_smoothing_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    epsilon: float = 0.1,
) -> Tensor:
    """
    Cross-entropy loss with label smoothing, implemented from scratch.
    Args:
        logits: Tensor of shape (batch_size, num_classes)
        targets: Tensor of shape (batch_size,) with integer class indices
        epsilon: Label smoothing factor in [0, 1)
    Returns:
        Scalar loss (mean over batch).
    """
    batch_size, num_classes = logits.shape

    # Compute log softmax stably: logits - logsumexp(logits)
    # Max for stability; shape: (batch_size, 1)
    max_logits, _ = logits.max(dim=-1, keepdim=True)
    shifted = logits - max_logits  # (batch_size, num_classes)
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))  # (batch_size, 1)
    log_softmax = shifted - log_sum_exp  # (batch_size, num_classes)

    # One-hot encoding of targets; shape: (batch_size, num_classes)
    one_hot = torch.zeros(batch_size, num_classes, dtype=logits.dtype, device=logits.device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

    # Smoothed target distribution; shape: (batch_size, num_classes)
    smoothed = (1.0 - epsilon) * one_hot + epsilon / num_classes

    # Cross-entropy: -sum(smoothed * log_softmax) over classes
    # shape: (batch_size,)
    loss_per_sample = -torch.sum(smoothed * log_softmax, dim=-1)

    # Mean over batch
    return loss_per_sample.mean()