"""
Task 5 -- What should a latent message preserve?

Cross-field idea: rate-distortion theory WITH SIDE INFORMATION (Wyner-Ziv /
Slepian-Wolf, information theory). A decoder that already has correlated side
information Y needs the encoder to transmit only what it cannot reconstruct
from Y -- the *conditional* information H(X | Y), not H(X). Reconstructing X
faithfully wastes rate on what Y already implies.

Mapping to LLM-agent latent handoff:
    worker            -> encoder (has full context X)
    receiver          -> decoder
    receiver context  -> side information Y (its own tokens + model priors)
    message           -> the k transmitted numbers (the "rate"/budget)
    error             -> downstream TASK loss (distortion), NOT reconstruction error

Falsifiable claim:
    At a fixed budget k, selecting the message to maximize *conditional,
    task-relevant* information (what is surprising to the receiver AND predictive
    of the label) beats reconstruction-based selection (top-variance / best
    reconstruction of the worker representation).
    -> Supported if task-conditioned+receiver-aware selection has higher
       downstream accuracy at matched k.
    -> Against if reconstruction-based selection matches or beats it.

This file runs the SELECTION experiment on feature matrices. It works on:
  (a) real saved hidden states  (build_tensors.py produces X.npy, Y.npy, labels.npy)
  (b) a synthetic generator (default) that gives ground-truth control, so the
      experimental design itself can be validated before touching a real model.

Selection strategies compared at fixed budget k (all pick k linear features):
  RECON     : top-k principal directions of X            (best reconstruction of X)
  SURPRISE  : top-k principal directions of the RESIDUAL of X after linearly
              predicting X from side info Y              (max conditional info)
  TASK      : top-k features of X by |correlation with the label|  (task-relevant)
  SURP+TASK : task-relevant features of the residual     (conditional AND relevant)

Decoder = logistic-regression probe on [Y_features , message].  Same probe,
same k, same data for every strategy -> the only variable is WHAT is sent.
"""

import argparse
import numpy as np


# ----------------------------- data sources --------------------------------
def synthetic(n=4000, d=64, seed=0):
    """Ground-truth generator we can reason about.

    Label depends on TWO latent factors:
      z_known    -- already recoverable from the receiver's side info Y
      z_unknown  -- present in the worker's X but NOT in Y (the surprising part)
    A reconstruction-based selector chases the high-variance directions of X,
    which we deliberately load onto RECEIVER-KNOWN, label-irrelevant noise.
    So reconstruction should waste budget; surprise+task should not.
    """
    rng = np.random.default_rng(seed)
    z_known = rng.standard_normal(n)
    z_unknown = rng.standard_normal(n)
    # label needs BOTH factors, but z_unknown carries the receiver-novel signal
    logits = 1.2 * z_unknown + 0.8 * z_known
    labels = (logits + 0.3 * rng.standard_normal(n) > 0).astype(int)

    # high-variance, receiver-known, label-IRRELEVANT nuisance (the recon trap)
    nuisance = 4.0 * rng.standard_normal((n, 8))

    # worker representation X mixes everything (+ big nuisance variance)
    parts = [z_unknown[:, None], z_known[:, None], nuisance,
             0.5 * rng.standard_normal((n, d - 10))]
    X = np.concatenate(parts, axis=1)
    X = X @ rng.standard_normal((X.shape[1], d))      # random rotation -> entangled

    # receiver side info Y knows z_known and the nuisance, but NOT z_unknown
    Yparts = [z_known[:, None], nuisance, 0.3 * rng.standard_normal((n, 6))]
    Y = np.concatenate(Yparts, axis=1)
    Y = Y @ rng.standard_normal((Y.shape[1], 24))
    return X.astype(np.float64), Y.astype(np.float64), labels


def load_real(prefix):
    X = np.load(f"{prefix}_X.npy"); Y = np.load(f"{prefix}_Y.npy")
    labels = np.load(f"{prefix}_labels.npy")
    return X, Y, labels


# ----------------------------- helpers -------------------------------------
def zscore(A, mu=None, sd=None):
    if mu is None:
        mu, sd = A.mean(0), A.std(0) + 1e-8
    return (A - mu) / sd, mu, sd


def residual_after_Y(X, Y):
    """Part of X not linearly predictable from side info Y (the 'surprise')."""
    Yb = np.concatenate([Y, np.ones((len(Y), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Yb, X, rcond=None)
    return X - Yb @ W


def topk_pca_dirs(M, k):
    Mc = M - M.mean(0)
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return Vt[:k].T                                   # (d, k) projection


def topk_task_dirs(M, y, k):
    """k standardized features of M most correlated with the label (as axis picks)."""
    Mc, _, _ = zscore(M)
    corr = np.abs((Mc * (y - y.mean())[:, None]).mean(0))
    idx = np.argsort(-corr)[:k]
    P = np.zeros((M.shape[1], k))
    for j, i in enumerate(idx):
        P[i, j] = 1.0
    return P


def logreg_fit(F, y, l2=1.0, iters=300, lr=0.5):
    """Tiny logistic regression (no sklearn dependency)."""
    Fb = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    w = np.zeros(Fb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Fb @ w)))
        grad = Fb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0] / len(y)
        w -= lr * grad
    return w


def logreg_acc(w, F, y):
    Fb = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    return float((((Fb @ w) > 0).astype(int) == y).mean())


# ----------------------------- experiment ----------------------------------
def run(X, Y, labels, k, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels); perm = rng.permutation(n); cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]

    Xs, mu, sd = zscore(X[tr]);   Xte = (X[te] - mu) / sd
    Ys, muy, sdy = zscore(Y[tr]); Yte = (Y[te] - muy) / sdy
    ytr, yte = labels[tr], labels[te]

    Rtr = residual_after_Y(Xs, Ys)
    Rte = Xte - np.concatenate([Yte, np.ones((len(Yte), 1))], 1) @ \
          np.linalg.lstsq(np.concatenate([Ys, np.ones((len(Ys), 1))], 1), Xs, rcond=None)[0]

    strategies = {
        "RECON":     topk_pca_dirs(Xs, k),
        "SURPRISE":  topk_pca_dirs(Rtr, k),
        "TASK":      topk_task_dirs(Xs, ytr, k),
        "SURP+TASK": topk_task_dirs(Rtr, ytr, k),
    }
    src = {"RECON": (Xs, Xte), "SURPRISE": (Rtr, Rte),
           "TASK": (Xs, Xte), "SURP+TASK": (Rtr, Rte)}

    out = {}
    for name, P in strategies.items():
        Mtr, Mte = src[name]
        msg_tr, msg_te = Mtr @ P, Mte @ P                # k-dim message
        Ftr = np.concatenate([Ys, msg_tr], 1)            # receiver side info + message
        Fte = np.concatenate([Yte, msg_te], 1)
        w = logreg_fit(Ftr, ytr)
        out[name] = logreg_acc(w, Fte, yte)

    # baselines: receiver alone (no message), full X (unlimited budget)
    w_y = logreg_fit(Ys, ytr); out["Y_only(no msg)"] = logreg_acc(w_y, Yte, yte)
    w_full = logreg_fit(np.concatenate([Ys, Xs], 1), ytr)
    out["Y+full_X(no budget)"] = logreg_acc(w_full, np.concatenate([Yte, Xte], 1), yte)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_prefix", default=None, help="use <prefix>_X/_Y/_labels.npy")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print(f"budget k = {args.k}, averaged over {args.seeds} seeds\n")
    agg = {}
    for s in range(args.seeds):
        X, Y, labels = (load_real(args.real_prefix) if args.real_prefix
                        else synthetic(seed=s))
        res = run(X, Y, labels, args.k, seed=s)
        for name, acc in res.items():
            agg.setdefault(name, []).append(acc)

    order = ["Y_only(no msg)", "RECON", "TASK", "SURPRISE", "SURP+TASK", "Y+full_X(no budget)"]
    for name in order:
        v = np.array(agg[name])
        print(f"  {name:22s} acc = {v.mean():.4f} +/- {v.std():.4f}")
    print("\nClaim SUPPORTED if SURP+TASK (and SURPRISE) > RECON at the same k.")


if __name__ == "__main__":
    main()
