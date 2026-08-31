# Optimizer-Agnostic Gradient Transformation

Make any `torch.optim` optimizer step on the transformed gradient

```
T(g) = sign(g) * g**2 = g * |g|
```

without modifying the optimizer, and with the original gradients restored on
`.grad` after the step.

## Files
- `transformed_optimizer.py` — `TransformedOptimizer` wrapper + `signed_square` transform.
- `test_transform.py` — automated correctness tests (SGD and AdamW).

## Run
```bash
pip install torch
python test_transform.py
```
Expected: all checks report ~0 error and it prints `All tests passed.`

## Usage
```python
from transformed_optimizer import TransformedOptimizer
base = torch.optim.AdamW(model.parameters(), lr=1e-3)
opt  = TransformedOptimizer(base)          # transform defaults to g*|g|

opt.zero_grad()
loss = loss_fn(model(x), y)
loss.backward()
opt.step()                                  # optimizer sees T(g); .grad restored after
```

## What is tested
1. **Equivalence** — wrapper output == manually transforming grads then a plain step.
2. **Restoration** — after `step()`, `.grad` equals the original gradient (bit-exact).
3. **Closed form (SGD)** — plain SGD gives `p - lr * T(g)` exactly.
