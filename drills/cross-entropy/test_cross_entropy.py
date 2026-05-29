import torch
import torch.nn.functional as F
from from_scratch import stable_softmax, label_smoothing_cross_entropy


def test_stable_softmax_shape():
    logits = torch.randn(4, 10)
    result = stable_softmax(logits)
    assert result.shape == logits.shape, f"Shape mismatch: {result.shape} vs {logits.shape}"
    print("PASSED: test_stable_softmax_shape")


def test_stable_softmax_properties():
    logits = torch.randn(8, 20)
    probs = stable_softmax(logits)
    assert (probs >= 0).all(), "Softmax output contains negative values"
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5), \
        f"Softmax probabilities don't sum to 1: {sums}"
    print("PASSED: test_stable_softmax_properties")


def test_stable_softmax_vs_pytorch():
    torch.manual_seed(42)
    for _ in range(10):
        logits = torch.randn(16, 32)
        custom = stable_softmax(logits)
        reference = F.softmax(logits, dim=-1)
        assert torch.allclose(custom, reference, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(custom - reference).abs().max().item()}"
    print("PASSED: test_stable_softmax_vs_pytorch")


def test_stable_softmax_analytic():
    logits = torch.tensor([[0.0, 0.0]])
    probs = stable_softmax(logits)
    expected = torch.tensor([[0.5, 0.5]])
    assert torch.allclose(probs, expected, atol=1e-6), f"Expected {expected}, got {probs}"

    logits = torch.tensor([[1000.0, 1000.0]])
    probs = stable_softmax(logits)
    expected = torch.tensor([[0.5, 0.5]])
    assert torch.allclose(probs, expected, atol=1e-6), f"Expected {expected}, got {probs}"

    logits = torch.tensor([[0.0, 0.0, 0.0]])
    probs = stable_softmax(logits)
    expected = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    assert torch.allclose(probs, expected, atol=1e-6), f"Expected {expected}, got {probs}"

    print("PASSED: test_stable_softmax_analytic")


def test_stable_softmax_numerical_stability():
    logits = torch.tensor([[10000.0, 10001.0, 10002.0]])
    probs = stable_softmax(logits)
    assert torch.isfinite(probs).all(), f"Non-finite values: {probs}"
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    logits = torch.tensor([[-10000.0, -10001.0, -10002.0]])
    probs = stable_softmax(logits)
    assert torch.isfinite(probs).all(), f"Non-finite values: {probs}"
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    logits = torch.tensor([[10000.0, -10000.0]])
    probs = stable_softmax(logits)
    assert torch.isfinite(probs).all(), f"Non-finite values: {probs}"
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    print("PASSED: test_stable_softmax_numerical_stability")


def test_stable_softmax_gradient():
    logits = torch.randn(4, 10, requires_grad=True)
    probs = stable_softmax(logits)
    loss = probs.sum()
    loss.backward()
    assert logits.grad is not None, "Gradient is None"
    assert torch.isfinite(logits.grad).all(), "Non-finite gradients"
    print("PASSED: test_stable_softmax_gradient")


def test_label_smoothing_shape():
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    loss = label_smoothing_cross_entropy(logits, targets)
    assert loss.shape == (), f"Loss shape is not scalar: {loss.shape}"
    print("PASSED: test_label_smoothing_shape")


def test_label_smoothing_finite():
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    loss = label_smoothing_cross_entropy(logits, targets)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    print("PASSED: test_label_smoothing_finite")


def test_label_smoothing_vs_pytorch():
    torch.manual_seed(42)
    for epsilon in [0.0, 0.1, 0.2, 0.5]:
        for _ in range(5):
            logits = torch.randn(32, 20)
            targets = torch.randint(0, 20, (32,))
            custom = label_smoothing_cross_entropy(logits, targets, epsilon=epsilon)
            reference = F.cross_entropy(logits, targets, label_smoothing=epsilon)
            assert torch.allclose(custom, reference, atol=1e-5, rtol=1e-5), \
                f"epsilon={epsilon}, diff={abs(custom.item() - reference.item()):.8f}"
    print("PASSED: test_label_smoothing_vs_pytorch")


def test_label_smoothing_no_smoothing():
    torch.manual_seed(123)
    logits = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))
    custom = label_smoothing_cross_entropy(logits, targets, epsilon=0.0)
    reference = F.cross_entropy(logits, targets)
    assert torch.allclose(custom, reference, atol=1e-5, rtol=1e-5), \
        f"Diff: {abs(custom.item() - reference.item()):.8f}"
    print("PASSED: test_label_smoothing_no_smoothing")


def test_label_smoothing_gradient():
    logits = torch.randn(8, 10, requires_grad=True)
    targets = torch.randint(0, 10, (8,))
    loss = label_smoothing_cross_entropy(logits, targets, epsilon=0.1)
    loss.backward()
    assert logits.grad is not None, "Gradient is None"
    assert torch.isfinite(logits.grad).all(), "Non-finite gradients"
    print("PASSED: test_label_smoothing_gradient")


def test_label_smoothing_gradient_correctness():
    torch.manual_seed(42)
    for epsilon in [0.0, 0.1, 0.3]:
        logits_custom = torch.randn(8, 10, requires_grad=True)
        targets = torch.randint(0, 10, (8,))
        logits_ref = logits_custom.detach().clone().requires_grad_(True)

        loss_custom = label_smoothing_cross_entropy(logits_custom, targets, epsilon=epsilon)
        loss_custom.backward()

        loss_ref = F.cross_entropy(logits_ref, targets, label_smoothing=epsilon)
        loss_ref.backward()

        assert torch.allclose(logits_custom.grad, logits_ref.grad, atol=1e-5, rtol=1e-5), \
            f"Grad mismatch epsilon={epsilon}, max diff: {(logits_custom.grad - logits_ref.grad).abs().max().item()}"
    print("PASSED: test_label_smoothing_gradient_correctness")


def test_label_smoothing_analytic():
    eps = 0.1
    logits = torch.tensor([[torch.log(torch.tensor(3.0)), 0.0]])
    targets = torch.tensor([0])
    custom = label_smoothing_cross_entropy(logits, targets, epsilon=eps)

    # softmax = [3/4, 1/4], log_softmax = [log(3/4), log(1/4)]
    # smoothed = [1 - eps + eps/2, eps/2] = [0.95, 0.05]
    log_softmax = torch.tensor([torch.log(torch.tensor(3.0 / 4.0)),
                                torch.log(torch.tensor(1.0 / 4.0))])
    smoothed = torch.tensor([1.0 - eps + eps / 2, eps / 2])
    expected = -torch.sum(smoothed * log_softmax)

    assert torch.allclose(custom, expected, atol=1e-5), \
        f"Expected {expected.item()}, got {custom.item()}"
    print("PASSED: test_label_smoothing_analytic")


def test_label_smoothing_range():
    logits = torch.randn(16, 10)
    targets = torch.randint(0, 10, (16,))
    for epsilon in [0.0, 0.05, 0.1, 0.2, 0.5, 0.9]:
        loss = label_smoothing_cross_entropy(logits, targets, epsilon=epsilon)
        assert torch.isfinite(loss), f"Non-finite loss for epsilon={epsilon}"
        assert loss.item() >= 0, f"Negative loss for epsilon={epsilon}: {loss.item()}"
    print("PASSED: test_label_smoothing_range")


if __name__ == "__main__":
    tests = [
        test_stable_softmax_shape,
        test_stable_softmax_properties,
        test_stable_softmax_vs_pytorch,
        test_stable_softmax_analytic,
        test_stable_softmax_numerical_stability,
        test_stable_softmax_gradient,
        test_label_smoothing_shape,
        test_label_smoothing_finite,
        test_label_smoothing_vs_pytorch,
        test_label_smoothing_no_smoothing,
        test_label_smoothing_gradient,
        test_label_smoothing_gradient_correctness,
        test_label_smoothing_analytic,
        test_label_smoothing_range,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
    else:
        print("Some tests failed!")
        exit(1)