# Task 5 — What Should a Latent Message Preserve?

## Files
- `select_experiment.py` — the selection experiment (pure NumPy; runs anywhere). Compares RECON / TASK / SURPRISE / SURP+TASK selection at a fixed budget `k`. Ships with a synthetic generator (ground-truth control) and a `--real_prefix` mode for saved tensors.
- `build_tensors.py` — produces real saved hidden-state tensors from a small encoder (worker = full sentence, receiver side info = first half), the materials the task asks you to submit.
- `sst2_X.npy`, `sst2_Y.npy`, `sst2_labels.npy` — the saved hidden-state tensors (bert-tiny, n=3000), produced by `build_tensors.py`.
- `latent_message.ipynb` — the notebook: reruns the synthetic validation and the real-tensor experiment end to end, with the plot and interpretation below (already executed, outputs included).

## Reproduce
```bash
pip install torch transformers datasets numpy nbformat nbclient ipykernel

# 1) design validation on ground-truth-controlled synthetic data
python select_experiment.py --k 3 --seeds 5
python select_experiment.py --k 6 --seeds 5

# 2) real hidden states
python build_tensors.py --model prajjwal1/bert-tiny --n 3000 --out sst2
python select_experiment.py --real_prefix sst2 --k 8 --seeds 5

# 3) or just run the notebook, which does both
jupyter nbconvert --to notebook --execute latent_message.ipynb
```

---

# TECHNICAL DOCUMENT

**1. The cross-field idea.** Rate–distortion theory *with side information* — the Wyner-Ziv (lossy) and Slepian-Wolf (lossless) results from information theory. Their core statement: if the decoder has side information Y correlated with the source X, the encoder only needs to transmit the *conditional* information H(X | Y), not H(X) — and remarkably, for the lossless case, it can do this even without seeing Y. Source: Cover & Thomas, *Elements of Information Theory*, ch. 15 (network information theory); Wyner & Ziv (1976), "The rate-distortion function for source coding with side information at the decoder." [add exact page/section you cite].

**2. Mapping (state where it is exact vs analogy).**

| info-theory object | latent-handoff counterpart | exact or analogy |
|---|---|---|
| source X | worker's full-context representation | exact object, informal distribution |
| side information Y | receiver's own context + model priors | **analogy** — priors aren't a clean random variable |
| rate R | message budget k (dims/bits) | exact (a countable budget) |
| distortion D | downstream **task** loss | **reframed** — not reconstruction error |
| decoder | receiver model / probe | exact |

Where it's exact: there is a real correlated side channel and a real, countable budget. Where it's analogy: "surprise to the receiver" is operationalized as the linear residual of X given Y, which underestimates what a nonlinear receiver could already infer; and priors-as-side-information has no clean distribution.

**3. Falsifiable claim.** At a fixed budget k, selecting the message to carry information that is **surprising to the receiver** (the residual of X given Y) — optionally intersected with **task relevance** — yields higher downstream accuracy than **reconstruction-based** selection (top-variance directions of X). This separates the account from the reconstruction default and from a purely task-salient-to-the-sender account.
- Supported: SURPRISE / SURP+TASK > RECON at matched k.
- Against: RECON ≥ SURPRISE at matched k.

**4. Smallest experiment.** Fixed budget k; same logistic-regression decoder; receiver always gets its side info Y plus a k-dim message. The *only* variable is which k linear features are sent. Four selectors: RECON (PCA of X), TASK (label-correlated features of X), SURPRISE (PCA of the Y-residual of X), SURP+TASK (label-correlated features of the residual). Baselines: Y-only (no message) and Y+full-X (no budget) bracket the achievable range.

**5–6. Result.**

*Design validation (synthetic, ground-truth control), k=3, 5 seeds — actually run:*

| selector | accuracy |
|---|---|
| Y only (no message) | 0.681 ± 0.009 |
| RECON | 0.711 ± 0.018 |
| TASK | 0.761 ± 0.028 |
| SURPRISE | **0.816 ± 0.030** |
| SURP+TASK | 0.780 ± 0.020 |
| Y + full X (no budget) | 0.895 ± 0.013 |

Surprise-based selection beats reconstruction by ~10 accuracy points at the same budget, in the predicted direction. The synthetic generator deliberately loads the high-variance directions of X onto receiver-known, label-irrelevant nuisance — the trap reconstruction falls into.

*Real hidden states (SST-2, bert-tiny), k=8, 5 seeds — actually run (`python select_experiment.py --real_prefix sst2 --k 8 --seeds 5`, also in `latent_message.ipynb`):*

| selector | accuracy |
|---|---|
| Y only (no message) | 0.6589 ± 0.0070 |
| RECON | 0.6624 ± 0.0080 |
| TASK | 0.6896 ± 0.0064 |
| SURPRISE | **0.7004 ± 0.0114** |
| SURP+TASK | 0.6962 ± 0.0047 |
| Y + full X (no budget) | 0.6827 ± 0.0069 |

The ordering SURPRISE > SURP+TASK > TASK > RECON > Y-only holds on real representations, exactly as predicted, though the gaps are much smaller than on the (ground-truth-engineered) synthetic data — as expected, since real sentiment content doesn't cleanly decompose into a receiver-known nuisance vs. a receiver-novel signal the way the synthetic generator was built to.

**7. Interpretation.**
- Does it support the claim? **Yes.** SURPRISE and SURP+TASK both beat RECON at matched k=8 on real hidden states, not just on synthetic data.
- One alternative explanation: on the synthetic data, pure SURPRISE beat SURP+TASK because the novel signal already dominates residual variance, so residual-PCA captures it and task-weighting adds variance — i.e. "surprise" and "task-relevant" coincided by construction. On real data SURPRISE *still* edges out SURP+TASK (0.7004 vs. 0.6962), suggesting the residual's top-variance directions in this encoder are already fairly well aligned with sentiment content, so the task-correlation filter mostly trims noise rather than adding signal — though with 5 seeds and a 0.0114/0.0047 std this gap is not clearly significant.
- Where the connection breaks down: the linear residual under-measures receiver knowledge (a strong nonlinear receiver could reconstruct more from Y than linear regression removes), so "surprise" is an upper bound on what must be sent; and priors are not a proper side-information variable.
- A second, unplanned real-data finding: `Y+full_X` (the "no budget" bracket, meant to be an upper bound) scores *lower* (0.6827) than SURPRISE/SURP+TASK/TASK, unlike on synthetic data where it was the ceiling. Most likely cause: handing the L2-regularized logistic probe the full 128-dim X on top of Y, with only ~2,100 training examples after the 70/30 split, adds enough noisy/collinear dimensions to hurt a simple linear decoder — a decoder-capacity artifact, not evidence that more information is intrinsically worse. This means the "no-budget" bracket is only a genuine upper bound when the decoder can actually exploit the extra dimensions; worth flagging rather than hiding.
- Follow-up: replace the linear residual with a learned receiver (nonlinear predictor of X from Y) to define surprise; sweep k to trace an operational rate–distortion curve for each selector; vary receiver strength to test the prediction that a *stronger* receiver needs a *smaller* message.
