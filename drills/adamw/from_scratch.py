"""AdamW optimizer implemented from scratch in pure PyTorch.

Adam (Kingma & Ba, 2014) with *decoupled weight decay* (Loshchilov & Hutter, 2017).

Key equations per parameter p with gradient g at step t:

    m_t  = β₁ · m_{t-1} + (1 − β₁) · g
    v_t  = β₂ · v_{t-1} + (1 − β₂) · g²
    m̂_t  = m_t / (1 − β₁ᵗ)                  ← bias-corrected first moment
    v̂_t  = v_t / (1 − β₂ᵗ)                  ← bias-corrected second moment
    p    = p − α · m̂_t / (√v̂_t + ε)  −  α · λ · p
                                         ^^^^^^^^^^^
                              decoupled weight decay (direct on params,
                              NOT through gradient as in Adam + L2 reg)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class AdamWFromScratch(Optimizer):
    """AdamW with explicit bias-corrected moments and decoupled weight decay."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults: dict[str, Any] = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, maximize=maximize,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        """Perform a single optimisation step.

        Returns the loss if *closure* is provided (mirrors PyTorch API).
        """
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1: float = group["betas"][0]
            beta2: float = group["betas"][1]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            maximize: bool = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # grad: (*p.shape)  — negate if maximising
                grad: Tensor = p.grad if not maximize else -p.grad

                # Lazy state initialisation (first time we see this param)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # exp_avg  – EMA of gradients (first moment)
                    # exp_avg2 – EMA of squared gradients (second moment)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg: Tensor = state["exp_avg"]       # m_{t-1}, (*p.shape)
                exp_avg_sq: Tensor = state["exp_avg_sq"] # v_{t-1}, (*p.shape)

                state["step"] += 1
                t: int = state["step"]                   # 1-indexed step counter

                # ---- 1. Update biased moments ----
                # m_t = β₁ · m_{t-1} + (1 − β₁) · g
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)        # (*p.shape)
                # v_t = β₂ · v_{t-1} + (1 − β₂) · g²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)  # (*p.shape)

                # ---- 2. Bias correction ----
                bias_correction1: float = 1.0 - beta1 ** t    # scalar
                bias_correction2: float = 1.0 - beta2 ** t    # scalar
                # √(v̂_t) + ε   (denominator, broadcastable to p.shape)
                denom: Tensor = (
                    (exp_avg_sq / bias_correction2).sqrt().add_(eps)  # (*p.shape)
                )
                # α · m̂_t / (√v̂_t + ε)  — bias-corrected Adam update
                step_size: float = lr / bias_correction1          # scalar

                # ---- 3. Update parameters ----
                # Pure gradient-based Adam update (no weight decay baked in)
                p.addcdiv_(exp_avg, denom, value=-step_size)       # (*p.shape)

                # ---- 4. Decoupled weight decay ----
                # p = p − α · λ · p   (directly on params, independent of grad)
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)            # (*p.shape)

        return loss
