"""CPU arithmetic checks for the proposed soft OLA objective; not a training experiment."""
import json
import math
from pathlib import Path

def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]

def soft_ola(q, x, correction):
    # Each hypothetical class c has its own adjusted normalization.
    probs = [softmax([v - a for v, a in zip(x, row)]) for row in correction]
    loss = -sum(q[c] * math.log(probs[c][c]) for c in range(len(q)))
    grad = [sum(q[c] * probs[c][j] for c in range(len(q))) - q[j]
            for j in range(len(q))]
    return loss, grad

def main():
    raw = [-0.3, 0.1, 0.7]
    bias = [0.4, -0.2, 1.1]
    x = [z + b for z, b in zip(raw, bias)]
    q = softmax(x)
    zeros = [[0.0] * 3 for _ in range(3)]
    loss0, grad0 = soft_ola(q, x, zeros)
    entropy = -sum(p * math.log(p) for p in q)
    assert abs(loss0 - entropy) < 1e-12
    assert max(map(abs, grad0)) < 1e-12

    # Toy assumption: class 0 is suppressed relative to a reference distribution.
    # The toy stipulates that correction; it does not establish a real target error.
    correction = [[0.1, 0.0], [0.0, 0.0]]
    q2, x2 = [0.5, 0.5], [0.0, 0.0]
    loss, grad = soft_ola(q2, x2, correction)
    eps = 1e-6
    numeric = []
    for j in range(2):
        plus, minus = x2.copy(), x2.copy()
        plus[j] += eps
        minus[j] -= eps
        numeric.append((soft_ola(q2, plus, correction)[0] -
                        soft_ola(q2, minus, correction)[0]) / (2 * eps))
    assert max(abs(a - b) for a, b in zip(grad, numeric)) < 1e-8
    assert grad[0] < 0 < grad[1]
    # A one-hot soft objective is the corresponding row's adjusted hard CE.
    onehot_loss, _ = soft_ola([1.0, 0.0], x2, correction)
    assert abs(onehot_loss + math.log(softmax([-0.1, 0.0])[0])) < 1e-12
    # Incorrectly adding bias again fails the identity control.
    _, bad_grad = soft_ola(q, [v + b for v, b in zip(x, bias)], zeros)
    assert max(map(abs, bad_grad)) > 1e-3

    out = {
        "status": "PASS",
        "scope": "CPU arithmetic only; no data, model training, GPU, or accuracy evaluation",
        "zero_strength_ce_minus_entropy": loss0 - entropy,
        "zero_strength_teacher_student_gradient_max": max(map(abs, grad0)),
        "toy_corrected_gradient": grad,
        "toy_finite_difference_gradient": numeric,
        "finite_difference_max_error": max(abs(a-b) for a,b in zip(grad,numeric)),
        "double_bias_gradient_max": max(map(abs,bad_grad)),
        "claims_not_tested": ["GMM implementation", "PyTorch autograd",
                              "H75 accuracy", "target-label correctness", "mIoU improvement"]
    }
    Path(__file__).with_name("soft_ola_math_result.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()

