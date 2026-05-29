import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from from_scratch import (
    compute_log_probs_from_logits,
    dpo_loss,
    DPOLossModule,
)


def test_compute_log_probs_from_logits():
    """Test shape, masking, and numerical correctness."""
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 8, 10
    ignore_index = -100

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # mask last 2 positions
    labels[:, -2:] = ignore_index

    log_probs = compute_log_probs_from_logits(logits, labels, ignore_index)
    assert log_probs.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {log_probs.shape}"

    # Compute reference using PyTorch's built-in cross-entropy
    # Shift logits and labels for next-token prediction
    ref_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
    ref_labels = labels[:, 1:].contiguous().view(-1)
    ref_log_probs = -F.cross_entropy(ref_logits, ref_labels, ignore_index=ignore_index, reduction='none')
    # Reshape to (batch, seq_len-1), mask, and sum
    ref_log_probs = ref_log_probs.view(batch_size, -1)
    mask = (labels[:, 1:] != ignore_index).float()
    ref_seq_log_probs = (ref_log_probs * mask).sum(dim=-1)

    assert torch.allclose(log_probs, ref_seq_log_probs, atol=1e-5), \
        f"compute_log_probs_from_logits mismatch: max diff = {(log_probs - ref_seq_log_probs).abs().max()}"

    # Test gradient flow
    loss = log_probs.sum()
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), \
        "Gradients through compute_log_probs_from_logits are not finite"


def test_dpo_loss_analytic():
    """Test DPO loss against analytic ground truth."""
    torch.manual_seed(1)
    batch_size = 4

    # Fixed values for reproducibility
    policy_chosen_logps = torch.randn(batch_size, requires_grad=True)
    policy_rejected_logps = torch.randn(batch_size, requires_grad=True)
    reference_chosen_logps = torch.randn(batch_size, requires_grad=True)
    reference_rejected_logps = torch.randn(batch_size, requires_grad=True)

    for beta in [0.1, 0.5, 1.0]:
        for label_smoothing in [0.0, 0.1, 0.2]:
            loss, chosen_rewards, rejected_rewards, reward_margin = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=beta,
                label_smoothing=label_smoothing,
            )

            # Analytic reference
            chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
            rejected_log_ratio = policy_rejected_logps - reference_rejected_logps
            logits_diff = beta * (chosen_log_ratio - rejected_log_ratio)

            if label_smoothing == 0.0:
                expected_loss = -F.logsigmoid(logits_diff).mean()
            else:
                expected_loss = (
                    -(1.0 - label_smoothing) * F.logsigmoid(logits_diff)
                    - label_smoothing * F.logsigmoid(-logits_diff)
                ).mean()

            assert torch.allclose(loss, expected_loss, atol=1e-5), \
                f"DPO loss mismatch for beta={beta}, ls={label_smoothing}: {loss} vs {expected_loss}"

            # Check reward diagnostics
            expected_chosen_rewards = beta * chosen_log_ratio
            expected_rejected_rewards = beta * rejected_log_ratio
            expected_margin = expected_chosen_rewards - expected_rejected_rewards

            assert torch.allclose(chosen_rewards, expected_chosen_rewards, atol=1e-5), \
                f"Chosen rewards mismatch for beta={beta}, ls={label_smoothing}"
            assert torch.allclose(rejected_rewards, expected_rejected_rewards, atol=1e-5), \
                f"Rejected rewards mismatch for beta={beta}, ls={label_smoothing}"
            assert torch.allclose(reward_margin, expected_margin, atol=1e-5), \
                f"Reward margin mismatch for beta={beta}, ls={label_smoothing}"


def test_dpo_loss_shapes_and_gradients():
    """Test shapes and gradient flow through dpo_loss."""
    torch.manual_seed(2)
    batch_size = 3

    policy_chosen_logps = torch.randn(batch_size, requires_grad=True)
    policy_rejected_logps = torch.randn(batch_size, requires_grad=True)
    reference_chosen_logps = torch.randn(batch_size, requires_grad=True)
    reference_rejected_logps = torch.randn(batch_size, requires_grad=True)

    loss, chosen_rewards, rejected_rewards, reward_margin = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=0.1,
    )

    assert loss.shape == torch.Size([]), f"Expected scalar loss, got shape {loss.shape}"
    assert chosen_rewards.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {chosen_rewards.shape}"
    assert rejected_rewards.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {rejected_rewards.shape}"
    assert reward_margin.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {reward_margin.shape}"

    # Check all outputs are finite
    for name, tensor in [("loss", loss), ("chosen_rewards", chosen_rewards),
                          ("rejected_rewards", rejected_rewards), ("reward_margin", reward_margin)]:
        assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"

    # Test gradient flow
    loss.backward()
    for param, name in [(policy_chosen_logps, "policy_chosen_logps"),
                        (policy_rejected_logps, "policy_rejected_logps"),
                        (reference_chosen_logps, "reference_chosen_logps"),
                        (reference_rejected_logps, "reference_rejected_logps")]:
        assert param.grad is not None and torch.isfinite(param.grad).all(), \
            f"Gradients through {name} are not finite or None"


def test_dpo_loss_module():
    """Test DPOLossModule end-to-end with logits and labels."""
    torch.manual_seed(3)
    batch_size, seq_len, vocab_size = 2, 12, 20

    # Create inputs with requires_grad=True to test gradient flow
    policy_chosen_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    policy_rejected_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    reference_chosen_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    reference_rejected_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Add some padding
    chosen_labels[:, -3:] = -100
    rejected_labels[:, -3:] = -100

    criterion = DPOLossModule(beta=0.1, label_smoothing=0.0)
    loss, chosen_rewards, rejected_rewards, reward_margin = criterion(
        policy_chosen_logits,
        policy_rejected_logits,
        reference_chosen_logits,
        reference_rejected_logits,
        chosen_labels,
        rejected_labels,
    )

    # Check shapes
    assert loss.shape == torch.Size([]), f"Expected scalar loss, got shape {loss.shape}"
    assert chosen_rewards.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {chosen_rewards.shape}"
    assert rejected_rewards.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {rejected_rewards.shape}"
    assert reward_margin.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {reward_margin.shape}"

    # Check all outputs are finite
    for name, tensor in [("loss", loss), ("chosen_rewards", chosen_rewards),
                          ("rejected_rewards", rejected_rewards), ("reward_margin", reward_margin)]:
        assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"

    # Test gradient flow through all logits
    loss.backward()
    for param, name in [(policy_chosen_logits, "policy_chosen_logits"),
                        (policy_rejected_logits, "policy_rejected_logits"),
                        (reference_chosen_logits, "reference_chosen_logits"),
                        (reference_rejected_logits, "reference_rejected_logits")]:
        assert param.grad is not None and torch.isfinite(param.grad).all(), \
            f"Gradients through {name} are not finite or None"

    # Test with label smoothing
    criterion_ls = DPOLossModule(beta=0.5, label_smoothing=0.1)
    policy_chosen_logits_grad = policy_chosen_logits.detach().requires_grad_(True)
    policy_rejected_logits_grad = policy_rejected_logits.detach().requires_grad_(True)
    reference_chosen_logits_grad = reference_chosen_logits.detach().requires_grad_(True)
    reference_rejected_logits_grad = reference_rejected_logits.detach().requires_grad_(True)

    loss_ls, _, _, _ = criterion_ls(
        policy_chosen_logits_grad,
        policy_rejected_logits_grad,
        reference_chosen_logits_grad,
        reference_rejected_logits_grad,
        chosen_labels,
        rejected_labels,
    )
    loss_ls.backward()
    assert torch.isfinite(policy_chosen_logits_grad.grad).all(), \
        "Gradients with label smoothing are not finite"


def test_reference_implementation_agreement():
    """Verify against a manual PyTorch implementation of DPO loss."""
    torch.manual_seed(4)
    batch_size = 3

    policy_chosen_logps = torch.randn(batch_size, requires_grad=True)
    policy_rejected_logps = torch.randn(batch_size, requires_grad=True)
    reference_chosen_logps = torch.randn(batch_size, requires_grad=True)
    reference_rejected_logps = torch.randn(batch_size, requires_grad=True)

    beta = 0.2

    # Manual reference implementation
    chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratio = policy_rejected_logps - reference_rejected_logps
    logits = beta * (chosen_log_ratio - rejected_log_ratio)
    ref_loss = -F.logsigmoid(logits).mean()
    ref_chosen_rewards = beta * chosen_log_ratio
    ref_rejected_rewards = beta * rejected_log_ratio
    ref_margin = ref_chosen_rewards - ref_rejected_rewards

    # Our implementation
    loss, chosen_rewards, rejected_rewards, margin = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=beta,
    )

    assert torch.allclose(loss, ref_loss, atol=1e-5), \
        f"Loss mismatch: {loss} vs {ref_loss}"
    assert torch.allclose(chosen_rewards, ref_chosen_rewards, atol=1e-5), \
        f"Chosen rewards mismatch: {chosen_rewards} vs {ref_chosen_rewards}"
    assert torch.allclose(rejected_rewards, ref_rejected_rewards, atol=1e-5), \
        f"Rejected rewards mismatch: {rejected_rewards} vs {ref_rejected_rewards}"
    assert torch.allclose(margin, ref_margin, atol=1e-5), \
        f"Margin mismatch: {margin} vs {ref_margin}"


if __name__ == "__main__":
    print("Running test_compute_log_probs_from_logits...")
    test_compute_log_probs_from_logits()
    print("✓ PASSED\n")

    print("Running test_dpo_loss_analytic...")
    test_dpo_loss_analytic()
    print("✓ PASSED\n")

    print("Running test_dpo_loss_shapes_and_gradients...")
    test_dpo_loss_shapes_and_gradients()
    print("✓ PASSED\n")

    print("Running test_dpo_loss_module...")
    test_dpo_loss_module()
    print("✓ PASSED\n")

    print("Running test_reference_implementation_agreement...")
    test_reference_implementation_agreement()
    print("✓ PASSED\n")

    print("All tests passed! ✓")