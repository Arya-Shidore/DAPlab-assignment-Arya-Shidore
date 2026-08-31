"""
Automated validation for TransformedOptimizer.

Three checks, each for SGD and AdamW:

  A. Equivalence: stepping through the wrapper produces exactly the same
     parameters as manually setting grad = T(g) and stepping a plain optimizer.
     This proves the optimizer really consumed the transformed gradient.

  B. Restoration: after step(), p.grad equals the ORIGINAL gradient (bit-exact),
     proving requirement (4) — grads are restored.

  C. Closed form (SGD only): plain SGD with lr, no momentum, gives
     p_new = p_old - lr * T(g) exactly, which we can compute by hand.

Everything is deterministic (fixed seeds), so results are reproducible.
"""

import torch
from transformed_optimizer import TransformedOptimizer, signed_square


def build_model(seed: int = 0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )


def get_batch(seed: int = 100):
    torch.manual_seed(seed)
    return torch.randn(16, 4), torch.randn(16, 1)


def loss_of(model, x, y):
    return torch.nn.functional.mse_loss(model(x), y)


def snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


def max_diff(a_list, b_list):
    return max((a - b).abs().max().item() for a, b in zip(a_list, b_list))


def equivalence_and_restore(make_opt, name):
    x, y = get_batch()

    # Model A: transform happens INSIDE the wrapper
    mA = build_model()
    optA = TransformedOptimizer(make_opt(mA.parameters()))
    optA.zero_grad()
    loss_of(mA, x, y).backward()
    grads_before = [p.grad.detach().clone() for p in mA.parameters()]
    optA.step()
    grads_after = [p.grad.detach().clone() for p in mA.parameters()]
    paramsA = snapshot(mA)

    # Model B (reference): SAME init + data, transform grads by hand, plain step
    mB = build_model()
    optB = make_opt(mB.parameters())
    optB.zero_grad()
    loss_of(mB, x, y).backward()
    for p in mB.parameters():
        p.grad.copy_(signed_square(p.grad))
    optB.step()
    paramsB = snapshot(mB)

    param_err = max_diff(paramsA, paramsB)     # should ~ float epsilon
    restore_err = max_diff(grads_before, grads_after)  # should be exactly 0
    print(f"[{name}] param diff vs reference = {param_err:.2e} | grad restore error = {restore_err:.2e}")
    assert param_err < 1e-6,  f"{name}: optimizer did not consume the transformed gradient"
    assert restore_err == 0.0, f"{name}: gradients were not restored exactly"


def closed_form_sgd():
    x, y = get_batch()
    lr = 0.1
    m = build_model()
    p0 = snapshot(m)
    opt = TransformedOptimizer(torch.optim.SGD(m.parameters(), lr=lr, momentum=0.0))
    opt.zero_grad()
    loss_of(m, x, y).backward()
    g = [p.grad.detach().clone() for p in m.parameters()]
    opt.step()
    err = max(
        (p.detach() - (p_old - lr * signed_square(gi))).abs().max().item()
        for p, p_old, gi in zip(m.parameters(), p0, g)
    )
    print(f"[SGD closed-form] max error vs p - lr*T(g) = {err:.2e}")
    assert err < 1e-6


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)  # tighter tolerances for the checks
    closed_form_sgd()
    equivalence_and_restore(lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9), "SGD+momentum")
    equivalence_and_restore(lambda ps: torch.optim.AdamW(ps, lr=0.01), "AdamW")
    print("\nAll tests passed.")
