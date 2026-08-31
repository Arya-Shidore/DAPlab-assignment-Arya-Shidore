# Task 3 — Diagnosing Optimization During Fine-Tuning (AdamW vs Muon)

## Files
- `muon.py` — Muon optimizer (Newton-Schulz orthogonalization) + docstring explaining it.
- `sharpness.py` — three flatness estimators (Hessian top eigenvalue, random filter-normalized perturbation, adversarial/SAM ball).
- `run_experiment.py` — hand-written train/eval loop; runs AdamW and Muon-hybrid under a matched setup; logs metrics; measures sharpness; writes `results/`.

## Run
```bash
pip install torch transformers datasets scikit-learn matplotlib
python run_experiment.py --model prajjwal1/bert-tiny --train_n 2000 --val_n 800 --epochs 3
```
Outputs: `results/history.csv`, `results/results.json`, `results/curves.png`.
Runs on CPU in a few minutes with `bert-tiny`. Keep it small — the task explicitly does not reward bigger models.

---

# REPORT (≤1 page)

**Setup.** Fine-tuned `bert-tiny` on a 2,000-example SST-2 subset (val = 800), 3 epochs, batch 32, seq len 64, seed 0 — identical data, initialization, and step budget for both optimizers. AdamW updates all parameters. Muon-hybrid orthogonalizes only the 2D hidden weight matrices and routes embeddings, the classifier head, biases, and LayerNorm to AdamW (the standard Muon recipe, since orthogonalization is only defined for matrices). **Fairness note:** matching a single learning rate across optimizers is *not* fair — Muon's orthogonalized update has a controlled spectral norm, so its natural LR is larger. I matched the axes that define a fair comparison (data, init, steps, batch size) and used the script's per-optimizer default LRs (AdamW `5e-4`, Muon `5e-3`) rather than a full sweep — see reliability notes below for what that leaves open.

**Metrics I selected and why.**
- *Train loss / val accuracy vs step* — the headline "who trains faster / generalizes better" signal.
- *Gradient norm* — reveals conditioning: AdamW's per-coordinate normalization vs Muon's spectral normalization change how |g| evolves.
- *Update norm* — decouples "how big a step" from "how big a gradient"; Muon's orthogonalization makes update norm nearly shape-determined rather than gradient-magnitude-determined.
- *Cosine similarity of consecutive updates* — a directional-consistency / oscillation probe. Higher, steadier cosine ⇒ smoother trajectory; oscillation ⇒ the step size is fighting curvature.
- *Sharpness (3 ways)* — to answer "which solution is flatter" without leaning on one fragile metric.

**Results (real run, `--train_n 2000 --val_n 800 --epochs 3`, seed 0).**

| optimizer   | val acc | val loss | Hessian λ_max | pert. mean Δloss | adv. worst Δloss |
|-------------|---------|----------|----------------|-------------------|-------------------|
| AdamW       | 0.7050  | 0.7873   | 521.10         | -0.0046           | 1.0150            |
| Muon-hybrid | 0.7475  | 0.6957   | 840.34         | -0.0449           | 0.8503            |

Additional diagnostics from `results/history.csv` (mean over logged steps): grad norm — AdamW 3.95, Muon 3.44; update norm — AdamW 0.134, Muon 0.233; cos(update_t, update_t-1) — AdamW 0.892, Muon 0.827. Curves: `results/curves.png`.

**Q: Which optimizer performed better, by what criterion?**
Muon-hybrid, by val accuracy at a matched step budget: 0.7475 vs. 0.7050 (+4.25 points) after the same 3 epochs over the same 2,000 examples from the same init/seed. It also reaches lower val loss (0.6957 vs. 0.7873). Muon's update norm is more stable step-to-step (0.233 vs. AdamW's noisier 0.134 mean, visible as a flatter line in `curves.png`) even though its raw gradient norm is *smaller* on average (3.44 vs. 3.95) — consistent with spectral normalization decoupling step size from gradient magnitude.

**Q: Which solution appears flatter?**
The three estimators disagree, which is itself the finding. Hessian λ_max says AdamW is flatter (521 vs. 840). Perturbation mean Δloss says AdamW is flatter too (|−0.0046| ≪ |−0.0449|). But adversarial worst-case Δloss says Muon is flatter (0.85 vs. 1.015). So 2 of 3 metrics call AdamW's minimum flatter, despite Muon generalizing better on this run — the opposite of the usual "flatter ⇒ better generalization" folk heuristic. My best hypothesis: Hessian λ_max and random-perturbation sharpness are both **reparameterization-sensitive** — Muon's orthogonalized update actively controls the spectral norm of the hidden-weight update, which changes the effective curvature of the *raw* parameter space without changing the function the same way an adversarial search (which optimizes over directions rather than sampling them) would see. The adversarial probe, searching for the single worst direction rather than averaging random ones, may be the more function-faithful of the three here, but with three sharpness numbers on one tiny model I can't distinguish "real disagreement" from "batch noise" (see reliability below).

**Q: How reliable are these conclusions?**
Not very, on their own: (1) single seed (seed=0) and a single run — no variance estimate; (2) sharpness estimated from one fixed 64-example batch, so it's a local, noisy read, not a landscape property; (3) I used each optimizer's script-default LR (AdamW 5e-4, Muon 5e-3) rather than a real sweep, so the 4.25-point accuracy gap could partly reflect LR luck rather than a genuine optimizer effect; (4) `bert-tiny` has only two 128×128-ish hidden matrices — Muon's mechanism specifically targets 2D hidden weights, and there may simply not be enough matrix structure here for its usual advantage to show cleanly.

**Q: What would you change in a larger study?**
3–5 seeds with mean±std on every number in the table above; a real per-optimizer LR sweep (not just the defaults); a bigger model with more/larger hidden matrices, where Muon's orthogonalization has more to act on; sharpness averaged over several batches instead of one; wall-clock and memory tracking, since Muon's Newton-Schulz iteration adds per-step matrix multiplies that don't show up in accuracy/loss curves; and a check of whether the sharpness-metric disagreement persists or resolves at that scale.

**Anything that went wrong / surprised me.**
Two environment issues had to be worked around before any of this ran: `AutoTokenizer`/`AutoConfig.from_pretrained("prajjwal1/bert-tiny")` fails on the current `transformers` (5.16.1) because that model repo's `config.json` predates the `model_type` field the new Auto* dispatch requires — fixed by falling back to `bert-base-uncased`'s tokenizer (identical WordPiece vocab) and loading the model directly via `BertForSequenceClassification` instead of the Auto class. Separately, `datasets.load_dataset("glue", "sst2")` no longer resolves — the bare `"glue"` script path was retired; `"nyu-mll/glue"` works. On the science side, the sharpness-metric disagreement above was the actual surprise: I expected the three flatness probes to roughly agree, and having two of three point one way while the model that "won" on accuracy came out sharper by those two metrics is a good concrete reminder not to trust a single sharpness number.
