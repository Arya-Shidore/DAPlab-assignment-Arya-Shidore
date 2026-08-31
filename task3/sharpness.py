"""
Three complementary ways to estimate the sharpness of a minimum.

Why three? No single number defines "sharpness", and each has a known blind
spot. Agreeing across methods is what makes the flatness claim credible.

1. top_hessian_eigenvalue  -- lambda_max of the loss Hessian via power iteration
   with Hessian-vector products. Direct curvature; the classic sharpness proxy.
   Blind spot: reparameterization-sensitive, single direction, expensive.

2. perturbation_sharpness   -- average loss increase under random FILTER-NORMALIZED
   perturbations of radius r (Li et al., 2018, "Visualizing the Loss Landscape").
   Cheap, many directions, scale-normalized. Blind spot: random directions may
   miss the sharpest one.

3. adversarial_sharpness    -- worst-case loss increase in an L2 ball found by a
   few gradient-ascent steps (the SAM notion of sharpness). Captures the sharp
   direction the random probe misses. Blind spot: only a local estimate.
"""

import copy
import torch


def _flat_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def top_hessian_eigenvalue(model, loss_fn, batch, iters=20, tol=1e-4):
    """Largest Hessian eigenvalue via power iteration on Hessian-vector products."""
    model.zero_grad()
    params = _flat_params(model)
    loss = loss_fn(model, batch)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    v = torch.randn_like(flat_grad)
    v /= v.norm()
    eig_old = 0.0
    for _ in range(iters):
        dot = (flat_grad * v).sum()
        hv = torch.autograd.grad(dot, params, retain_graph=True)
        hv = torch.cat([h.reshape(-1) for h in hv]).detach()
        eig = (v * hv).sum().item()          # Rayleigh quotient
        v = hv / (hv.norm() + 1e-12)
        if abs(eig - eig_old) < tol * (abs(eig) + 1e-12):
            break
        eig_old = eig
    return eig


@torch.no_grad()
def _filter_normalized_direction(model):
    """Random direction with each parameter tensor scaled to that tensor's norm."""
    direction = []
    for p in _flat_params(model):
        d = torch.randn_like(p)
        d.mul_(p.norm() / (d.norm() + 1e-12))   # per-tensor ("filter") normalization
        direction.append(d)
    return direction


@torch.no_grad()
def perturbation_sharpness(model, loss_fn, batch, radius=0.05, n_dirs=10):
    """Mean loss increase under random filter-normalized perturbations."""
    base = loss_fn(model, batch).item()
    params = _flat_params(model)
    saved = [p.detach().clone() for p in params]
    increases = []
    for _ in range(n_dirs):
        direction = _filter_normalized_direction(model)
        for p, d in zip(params, direction):
            p.add_(d, alpha=radius)
        increases.append(loss_fn(model, batch).item() - base)
        for p, s in zip(params, saved):
            p.copy_(s)                          # restore exactly
    t = torch.tensor(increases)
    return {"mean_increase": t.mean().item(), "max_increase": t.max().item(),
            "base_loss": base, "radius": radius}


def adversarial_sharpness(model, loss_fn, batch, radius=0.05, steps=5, step_frac=0.3):
    """Worst-case loss increase in an L2 ball of given radius (SAM-style)."""
    params = _flat_params(model)
    saved = [p.detach().clone() for p in params]
    with torch.no_grad():
        base = loss_fn(model, batch).item()

    for _ in range(steps):                       # gradient ASCENT toward worst point
        model.zero_grad()
        loss = loss_fn(model, batch)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            gnorm = torch.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12
            for p, g in zip(params, grads):
                p.add_(g, alpha=step_frac * radius / gnorm)
            # project back into the radius-r ball around the saved point
            disp = torch.sqrt(sum(((p - s) ** 2).sum() for p, s in zip(params, saved)))
            if disp > radius:
                for p, s in zip(params, saved):
                    p.copy_(s + (p - s) * (radius / disp))

    with torch.no_grad():
        worst = loss_fn(model, batch).item()
        for p, s in zip(params, saved):
            p.copy_(s)                           # restore exactly
    return {"worst_increase": worst - base, "base_loss": base, "radius": radius}
