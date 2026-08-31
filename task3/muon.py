"""
Muon optimizer  (MomentUm Orthogonalized by Newton-Schulz).

Reference implementation: Keller Jordan, https://kellerjordan.github.io/posts/muon/
and https://github.com/KellerJordan/Muon/blob/master/muon.py

Idea (understand this before the interview):
    Muon keeps SGD-momentum, then REPLACES each 2D parameter's momentum update
    with the nearest orthogonal matrix before applying it. If M = U S V^T is the
    SVD of the momentum, the nearest orthogonal matrix (in Frobenius norm) is
    U V^T, i.e. "set every singular value to 1". Rather than an expensive SVD,
    a 5-step Newton-Schulz (NS) iteration approximates U V^T cheaply.

Why only 2D params? Orthogonalization is defined for matrices. Biases,
LayerNorm gains, and embeddings/heads are 1D or semantically special, so the
standard recipe routes them to AdamW instead (see run_experiment.py).
"""

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal factor U V^T of a 2D matrix G via Newton-Schulz.

    The quintic coefficients (a, b, c) are the tuned values from the reference
    implementation; they drive the singular values of the normalized matrix
    toward 1 without an SVD. Runs in float32 here for CPU portability (the
    reference uses bfloat16 on GPU).
    """
    assert G.ndim == 2, "Newton-Schulz orthogonalization is only defined for matrices"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)                      # normalize so spectral norm <= 1
    transposed = False
    if X.size(0) > X.size(1):                     # iterate on the smaller Gram matrix
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden weights. Route everything else to AdamW yourself.

    Args:
        lr:            base learning rate
        momentum:      SGD momentum coefficient
        nesterov:      use Nesterov-style lookahead (as in the reference impl)
        ns_steps:      Newton-Schulz iterations
        weight_decay:  decoupled weight decay
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            nesterov, ns_steps = group["nesterov"], group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, "Muon expects 2D params; send 1D params to AdamW"

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)                 # m <- mom*m + g
                update = g.add(buf, alpha=mom) if nesterov else buf

                # orthogonalize the update, then rescale so its RMS is comparable
                # to Adam's (lets Adam-like learning rates transfer). The
                # sqrt(max(shape)) factor follows "Muon is Scalable for LLM Training".
                o = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = max(1.0, g.size(0) / g.size(1)) ** 0.5

                if wd != 0.0:
                    p.mul_(1 - lr * wd)              # decoupled weight decay
                p.add_(o, alpha=-lr * scale)

        return loss
