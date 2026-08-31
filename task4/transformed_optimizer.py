"""
Optimizer-agnostic gradient transformation.

We want any existing torch.optim optimizer to perform its update on the
transformed gradient

        T(g) = sign(g) * g**2 = g * |g|

instead of the raw gradient g, WITHOUT modifying or rewriting the optimizer,
and with the original gradients restored on `.grad` after the step.

The mechanism is a thin wrapper that, on each `step()`:
    1. saves the original gradients,
    2. overwrites p.grad with T(p.grad),
    3. calls the wrapped optimizer's own step(),
    4. copies the originals back into p.grad.
"""

from __future__ import annotations
import torch


def signed_square(g: torch.Tensor) -> torch.Tensor:
    """T(g) = sign(g) * g^2 = g * |g|  (elementwise, sign-preserving)."""
    return g * g.abs()


class TransformedOptimizer:
    """Drop-in wrapper around any torch.optim.Optimizer.

    Usage:
        base = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt  = TransformedOptimizer(base)          # transform defaults to g*|g|
        ...
        opt.zero_grad(); loss.backward(); opt.step()
    """

    def __init__(self, optimizer: torch.optim.Optimizer, transform=signed_square):
        self.optimizer = optimizer
        self.transform = transform

    @torch.no_grad()
    def step(self, closure=None):
        # (0) If a closure is supplied, we evaluate it ourselves exactly once to
        #     populate .grad and obtain the loss, then step WITHOUT passing the
        #     closure down. That prevents the wrapped optimizer from calling the
        #     closure internally and re-running backward, which would overwrite
        #     our transformed gradients mid-step.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # (1) save originals  +  (2) replace grad with T(grad)
        saved = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                saved[p] = p.grad.detach().clone()          # exact copy of g
                p.grad.copy_(self.transform(p.grad))        # in-place: g -> T(g)

        # (3) let the real optimizer do its normal update on T(g)
        self.optimizer.step()

        # (4) restore the original gradients on .grad
        for p, g in saved.items():
            p.grad.copy_(g)

        return loss

    # --- delegate the rest of the optimizer API so this is a true drop-in ---
    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def add_param_group(self, group):
        self.optimizer.add_param_group(group)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return f"TransformedOptimizer(transform={self.transform.__name__}, {self.optimizer!r})"
