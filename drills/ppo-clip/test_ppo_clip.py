import torch
import torch.nn as nn
import pytest
from from_scratch import compute_ppo_clipped_surrogate_objective, gae


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _reference_ppo_loss(
    action_logprobs: torch.Tensor,
    old_action_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """Pure-reference PPO clipped surrogate (independent re-implementation)."""
    ratio = torch.exp(action_logprobs - old_action_logprobs)
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.min(ratio * advantages, ratio_clipped * advantages)
    return -surrogate.mean()


def _reference_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Independent GAE reference using the same loop logic but written separately."""
    T = rewards.shape[0]
    advs = torch.zeros(T, dtype=rewards.dtype)
    gae_val = torch.tensor(0.0, dtype=rewards.dtype)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae_val = delta + gamma * lam * (1.0 - dones[t]) * gae_val
        advs[t] = gae_val
    return advs


# ===========================================================================
#  PPO clipped surrogate tests
# ===========================================================================

class TestPPOClippedSurrogate:

    # ---- analytic: ratio == 1 everywhere ----------------------------------
    def test_ratio_one_analytic(self):
        """When logprobs match, ratio=1, loss = -mean(advantages)."""
        torch.manual_seed(0)
        B = 32
        old_lp = torch.randn(B)
        adv = torch.randn(B)
        cur_lp = old_lp.clone()

        loss, metrics = compute_ppo_clipped_surrogate_objective(
            cur_lp, old_lp, adv, clip_epsilon=0.2
        )
        expected = -adv.mean()
        assert torch.allclose(loss, expected, atol=1e-6), (
            f"Expected {expected.item():.6f}, got {loss.item():.6f}"
        )

    # ---- analytic: positive advantages, large ratio → clip active ---------
    def test_clip_active_positive_adv(self):
        """With large ratio and positive advantages, clipping caps the gain."""
        B = 16
        eps = 0.2
        old_lp = torch.zeros(B)
        adv = torch.ones(B)  # all positive
        # Make ratio = exp(2) ≈ 7.39, well above 1+eps
        cur_lp = torch.full((B,), 2.0)

        loss, _ = compute_ppo_clipped_surrogate_objective(
            cur_lp, old_lp, adv, clip_epsilon=eps
        )
        # ratio_clipped = 1+eps = 1.2 for all; min(r*adv, clip*r*adv) = 1.2 * 1 = 1.2
        expected = torch.tensor(-1.2)
        assert torch.allclose(loss, expected, atol=1e-6)

    # ---- analytic: negative advantages, tiny ratio → clip active ----------
    def test_clip_active_negative_adv(self):
        """With small ratio and negative advantages, clipping caps the loss."""
        B = 16
        eps = 0.2
        old_lp = torch.zeros(B)
        adv = -torch.ones(B)  # all negative
        # Make ratio = exp(-3) ≈ 0.05, well below 1-eps=0.8
        cur_lp = torch.full((B,), -3.0)

        loss, _ = compute_ppo_clipped_surrogate_objective(
            cur_lp, old_lp, adv, clip_epsilon=eps
        )
        # ratio_clipped = 1-eps = 0.8; unclipped ratio*adv = 0.05*(-1) = -0.05
        # min(-0.05, 0.8*(-1)) = min(-0.05, -0.8) = -0.8
        expected = torch.tensor(0.8)  # -(-0.8)
        assert torch.allclose(loss, expected, atol=1e-6)

    # ---- exact comparison with independent reference ----------------------
    def test_against_reference(self):
        """Match independent reference implementation across random inputs."""
        torch.manual_seed(123)
        B = 64
        old_lp = torch.randn(B)
        cur_lp = old_lp + torch.randn(B) * 0.3
        adv = torch.randn(B)
        eps = 0.15

        loss, _ = compute_ppo_clipped_surrogate_objective(
            cur_lp, old_lp, adv, clip_epsilon=eps
        )
        ref_loss = _reference_ppo_loss(cur_lp, old_lp, adv, eps)
        assert torch.allclose(loss, ref_loss, atol=1e-6), (
            f"loss={loss.item():.6f}, ref={ref_loss.item():.6f}"
        )

    # ---- shape check ------------------------------------------------------
    def test_output_shapes(self):
        B = 20
        old_lp = torch.randn(B)
        cur_lp = torch.randn(B)
        adv = torch.randn(B)

        loss, metrics = compute_ppo_clipped_surrogate_objective(
            cur_lp, old_lp, adv
        )
        assert loss.shape == (), f"loss shape {loss.shape}, expected scalar"
        for key in [
            "ratio_mean", "ratio_std", "surrogate_unclipped_mean",
            "surrogate_clipped_mean", "loss", "approx_kl", "clip_fraction",
        ]:
            assert key in metrics, f"missing metric: {key}"
            assert metrics[key].shape == ()

    # ---- gradient flow & finiteness ---------------------------------------
    def test_gradient_flow_and_finite(self):
        torch.manual_seed(7)
        B = 48
        old_lp = torch.randn(B)
        adv = (torch.randn(B) - torch.randn(B))  # roughly zero-mean
        cur_lp = old_lp.clone().detach().requires_grad_(True)

        loss, metrics = compute_ppo_clipped_surrogate_objective(
            cur_lp, old_lp, adv, clip_epsilon=0.2
        )
        loss.backward()

        assert cur_lp.grad is not None, "No gradient reached cur_logprobs"
        assert torch.isfinite(cur_lp.grad).all(), "Non-finite gradient detected"
        assert torch.isfinite(loss), "Loss is not finite"
        for k, v in metrics.items():
            assert torch.isfinite(v), f"Metric '{k}' is not finite: {v}"

    # ---- gradient magnitude sanity ----------------------------------------
    def test_gradient_magnitude(self):
        """Larger epsilon → more clipping → smaller gradient magnitude
        when ratio is far from 1 and advantage is positive."""
        torch.manual_seed(99)
        B = 32
        old_lp = torch.zeros(B)
        adv = torch.ones(B)
        grad_norms = []
        for eps in [0.1, 0.3, 0.5]:
            cur_lp = torch.full((B,), 1.0, requires_grad=True)
            loss, _ = compute_ppo_clipped_surrogate_objective(
                cur_lp, old_lp, adv, clip_epsilon=eps
            )
            loss.backward()
            grad_norms.append(cur_lp.grad.norm().item())
        # With ratio = e ≈ 2.718, larger epsilon clips more aggressively
        # so gradient should shrink with larger epsilon
        assert grad_norms[0] > grad_norms[1] > grad_norms[2], (
            f"Expected decreasing grad norms with larger epsilon, got {grad_norms}"
        )

    # ---- clip fraction metric correctness ---------------------------------
    def test_clip_fraction_metric(self):
        """When ratio is uniform at 1.0, clip_fraction should be 0."""
        B = 50
        lp = torch.randn(B)
        adv = torch.randn(B)
        _, metrics = compute_ppo_clipped_surrogate_objective(
            lp, lp, adv, clip_epsilon=0.2
        )
        assert torch.allclose(
            metrics["clip_fraction"], torch.tensor(0.0), atol=1e-7
        )

    # ---- multiple epsilon values ------------------------------------------
    def test_various_epsilons(self):
        torch.manual_seed(42)
        B = 32
        old_lp = torch.randn(B)
        cur_lp = old_lp + torch.randn(B) * 0.1
        adv = torch.randn(B)
        for eps in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            loss, _ = compute_ppo_clipped_surrogate_objective(
                cur_lp, old_lp, adv, clip_epsilon=eps
            )
            ref = _reference_ppo_loss(cur_lp, old_lp, adv, eps)
            assert torch.allclose(loss, ref, atol=1e-6), (
                f"eps={eps}: loss={loss.item():.6f}, ref={ref.item():.6f}"
            )


# ===========================================================================
#  GAE tests
# ===========================================================================

class TestGAE:

    # ---- all-done analytic: A_t = r_t - V(s_t) ---------------------------
    def test_all_done_analytic(self):
        """When every step is terminal, advantages = rewards - values[:-1]."""
        T = 5
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        dones = torch.ones(T)

        adv = gae(rewards, values, dones, gamma=0.99, lam=0.95)
        expected = rewards - values[:T]
        assert torch.allclose(adv, expected, atol=1e-6), (
            f"Expected {expected}, got {adv}"
        )

    # ---- single-step analytic ---------------------------------------------
    def test_single_step(self):
        """T=1: A_0 = r_0 + γ·V(1)·(1-done) - V(0)."""
        rewards = torch.tensor([3.0])
        values = torch.tensor([1.0, 2.0])
        dones = torch.tensor([0.0])
        gamma, lam = 0.99, 0.95

        adv = gae(rewards, values, dones, gamma, lam)
        expected = 3.0 + 0.99 * 2.0 - 1.0
        assert torch.allclose(adv, torch.tensor([expected]), atol=1e-6)

    # ---- manual multi-step (no dones) -------------------------------------
    def test_manual_no_dones(self):
        """Hand-computed GAE for T=3, no termination."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5, 2.0])
        dones = torch.tensor([0.0, 0.0, 0.0])
        gamma, lam = 0.99, 0.95

        # Manual computation
        # t=2: δ=3+0.99*2.0-1.5=3.48; A2=3.48
        # t=1: δ=2+0.99*1.5-1.0=2.485; A1=2.485+0.99*0.95*3.48
        # t=0: δ=1+0.99*1.0-0.5=1.49; A0=1.49+0.99*0.95*A1
        A2 = 3.0 + 0.99 * 2.0 - 1.5
        A1 = (2.0 + 0.99 * 1.5 - 1.0) + 0.99 * 0.95 * A2
        A0 = (1.0 + 0.99 * 1.0 - 0.5) + 0.99 * 0.95 * A1
        expected = torch.tensor([A0, A1, A2])

        adv = gae(rewards, values, dones, gamma, lam)
        assert torch.allclose(adv, expected, atol=1e-6), (
            f"Expected {expected}, got {adv}"
        )

    # ---- manual with mid-episode done -------------------------------------
    def test_manual_with_done(self):
        """Hand-computed GAE with a done at t=2 (resets accumulation)."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5, 2.0])
        dones = torch.tensor([0.0, 1.0, 0.0])  # episode ends at t=1
        gamma, lam = 0.99, 0.95

        # t=2: δ=3+0.99*2.0-1.5=3.48; A2=3.48 (no done at t=2, gae_accum was 0)
        # t=1: δ=2+0.99*1.5*0-1.0=1.0; A1=1.0+0.99*0.95*0*gae_accum=1.0 (done=1, resets)
        #   Wait: done=1 at t=1, so:
        #   δ_1 = 2.0 + 0.99*1.5*(1-1) - 1.0 = 2.0 - 1.0 = 1.0
        #   A1 = 1.0 + 0.99*0.95*(1-1)*0 = 1.0
        # t=0: δ=1+0.99*1.0*1-0.5=1.49; A0=1.49+0.99*0.95*1*A1
        A2 = 3.0 + 0.99 * 2.0 * 1.0 - 1.5  # done[2]=0
        A1 = (2.0 + 0.99 * 1.5 * 0.0 - 1.0) + 0.99 * 0.95 * 0.0 * 0.0
        A0 = (1.0 + 0.99 * 1.0 * 1.0 - 0.5) + 0.99 * 0.95 * 1.0 * A1
        expected = torch.tensor([A0, A1, A2])

        adv = gae(rewards, values, dones, gamma, lam)
        assert torch.allclose(adv, expected, atol=1e-6), (
            f"Expected {expected}, got {adv}"
        )

    # ---- against independent reference ------------------------------------
    def test_against_reference(self):
        """Random inputs compared against independent GAE implementation."""
        torch.manual_seed(77)
        T = 50
        rewards = torch.randn(T)
        values = torch.randn(T + 1)
        dones = (torch.rand(T) > 0.85).float()  # ~15% done rate
        gamma, lam = 0.99, 0.95

        adv = gae(rewards, values, dones, gamma, lam)
        ref = _reference_gae(rewards, values, dones, gamma, lam)
        assert torch.allclose(adv, ref, atol=1e-6), (
            f"Max diff: {(adv - ref).abs().max().item():.8f}"
        )

    # ---- shape check ------------------------------------------------------
    def test_output_shape(self):
        T = 13
        rewards = torch.randn(T)
        values = torch.randn(T + 1)
        dones = torch.zeros(T)

        adv = gae(rewards, values, dones)
        assert adv.shape == (T,), f"Expected shape ({T},), got {adv.shape}"

    # ---- dtype preservation -----------------------------------------------
    def test_dtype_float64(self):
        T = 10
        rewards = torch.randn(T, dtype=torch.float64)
        values = torch.randn(T + 1, dtype=torch.float64)
        dones = torch.zeros(T, dtype=torch.float64)

        adv = gae(rewards, values, dones)
        assert adv.dtype == torch.float64

    # ---- finiteness -------------------------------------------------------
    def test_finite(self):
        torch.manual_seed(0)
        T = 100
        rewards = torch.randn(T) * 5
        values = torch.randn(T + 1) * 5
        dones = (torch.rand(T) > 0.8).float()

        adv = gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert torch.isfinite(adv).all(), "Non-finite values in GAE output"


# ===========================================================================
#  Integration: GAE → PPO pipeline
# ===========================================================================

class TestIntegration:
    def test_gae_then_ppo(self):
        """Run GAE, feed advantages into PPO loss, check shapes and grads."""
        torch.manual_seed(42)
        T = 32
        B = T  # one trajectory
        rewards = torch.randn(T)
        values = torch.randn(T + 1, requires_grad=True)
        dones = torch.zeros(T)
        old_lp = torch.randn(T)
        cur_lp = old_lp.clone().detach().requires_grad_(True)

        adv = gae(rewards, values.detach(), dones)
        assert adv.shape == (T,)
        assert torch.isfinite(adv).all()

        loss, metrics = compute_ppo_clipped_surrogate_objective(
            cur_lp, old_lp, adv
        )
        loss.backward()

        assert cur_lp.grad is not None
        assert torch.isfinite(cur_lp.grad).all()
        assert torch.isfinite(loss)

    def test_full_pipeline_end_to_end(self):
        """Synthetic mini-loop: 3 gradient steps should change the loss."""
        torch.manual_seed(0)
        T = 64
        rewards = torch.randn(T)
        values = torch.randn(T + 1)
        dones = (torch.rand(T) > 0.9).float()
        old_lp = torch.randn(T)

        adv = gae(rewards, values, dones, gamma=0.99, lam=0.95)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        cur_lp = old_lp.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([cur_lp], lr=0.01)

        losses = []
        for _ in range(5):
            opt.zero_grad()
            loss, _ = compute_ppo_clipped_surrogate_objective(
                cur_lp, old_lp, adv
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should change over optimization steps
        assert losses[0] != losses[-1], "Loss did not change during optimization"


# ===========================================================================
#  Main
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
