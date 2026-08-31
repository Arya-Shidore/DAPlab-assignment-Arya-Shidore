"""
Task 3 -- Diagnosing optimization during fine-tuning.

Fine-tune a small pretrained encoder on SST-2 with (a) AdamW and (b) Muon
(hybrid: 2D hidden weights -> Muon, everything else -> AdamW), under a matched
setup, then measure metrics that expose *how* the optimizers differ, and
estimate the flatness of each solution three ways.

The training/eval loop is written by hand on purpose (the task asks for that);
only the model/tokenizer/dataset come from libraries.

Run:
    pip install torch transformers datasets scikit-learn matplotlib
    python run_experiment.py --model prajjwal1/bert-tiny --train_n 2000 --val_n 800 --epochs 3

Everything is deterministic given --seed. Results -> results/ (CSV + PNG + JSON).
"""

import argparse, json, os, random, time
import torch
import torch.nn.functional as F

from muon import Muon
from sharpness import top_hessian_eigenvalue, perturbation_sharpness, adversarial_sharpness


def set_seed(s):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_data(model_name, train_n, val_n, max_len, seed):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except ValueError:
        # ponytail: bert-tiny ships no fast-tokenizer file; bert-base-uncased
        # uses the identical wordpiece vocab, so this is exact, not lossy.
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = load_dataset("nyu-mll/glue", "sst2")
    train = ds["train"].shuffle(seed=seed).select(range(train_n))
    val = ds["validation"].select(range(min(val_n, len(ds["validation"]))))

    def encode(split):
        enc = tok([x for x in split["sentence"]], truncation=True,
                  padding="max_length", max_length=max_len, return_tensors="pt")
        y = torch.tensor(split["label"])
        return enc["input_ids"], enc["attention_mask"], y

    return encode(train), encode(val), tok


def iterate_batches(data, bs, shuffle, device):
    ids, mask, y = data
    idx = torch.randperm(len(y)) if shuffle else torch.arange(len(y))
    for i in range(0, len(y), bs):
        j = idx[i:i + bs]
        yield ids[j].to(device), mask[j].to(device), y[j].to(device)


def make_model(model_name, device):
    from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
    try:
        m = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except ValueError:
        # ponytail: bert-tiny's config.json predates the model_type field AutoConfig now requires.
        m = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return m.to(device)


def loss_on_batch(model, batch):
    ids, mask, y = batch
    logits = model(input_ids=ids, attention_mask=mask).logits
    return F.cross_entropy(logits, y)


@torch.no_grad()
def evaluate(model, data, bs, device):
    model.eval()
    tot, correct, loss_sum, n = 0, 0, 0.0, 0
    for batch in iterate_batches(data, bs, False, device):
        ids, mask, y = batch
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        tot += len(y); n += len(y)
    model.train()
    return {"val_loss": loss_sum / n, "val_acc": correct / tot}


def build_optimizer(kind, model, adamw_lr, muon_lr, wd):
    """AdamW on all params, or Muon-hybrid (2D encoder weights -> Muon)."""
    if kind == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=adamw_lr, weight_decay=wd)], "AdamW"

    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2D hidden weights go to Muon; embeddings, classifier head, biases,
        # LayerNorm -> AdamW (standard Muon recipe).
        is_hidden_matrix = (p.ndim == 2 and "embeddings" not in name and "classifier" not in name)
        (muon_params if is_hidden_matrix else adamw_params).append(p)
    opts = [
        Muon(muon_params, lr=muon_lr, momentum=0.95, nesterov=True, weight_decay=wd),
        torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=wd),
    ]
    return opts, "Muon-hybrid"


def train_one(kind, args, device, val_batch_for_sharpness):
    set_seed(args.seed)                                  # identical init + data order
    train_data, val_data, _ = load_data(args.model, args.train_n, args.val_n, args.max_len, args.seed)
    model = make_model(args.model, device)
    opts, label = build_optimizer(kind, model, args.adamw_lr, args.muon_lr, args.wd)

    history, step = [], 0
    prev_update_dir = None
    for epoch in range(args.epochs):
        for batch in iterate_batches(train_data, args.bs, True, device):
            for o in opts:
                o.zero_grad()
            loss = loss_on_batch(model, batch)
            loss.backward()

            # --- diagnostics captured BEFORE the step ---
            gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                   if p.grad is not None)).item()
            params_before = [p.detach().clone() for p in model.parameters()]

            for o in opts:
                o.step()

            with torch.no_grad():
                update = torch.cat([(p - b).reshape(-1) for p, b in
                                    zip(model.parameters(), params_before)])
                unorm = update.norm().item()
                udir = update / (unorm + 1e-12)
                cos = float((udir * prev_update_dir).sum()) if prev_update_dir is not None else float("nan")
                prev_update_dir = udir

            if step % args.log_every == 0:
                ev = evaluate(model, val_data, args.eval_bs, device)
                row = {"opt": label, "step": step, "epoch": epoch,
                       "train_loss": loss.item(), "grad_norm": gnorm,
                       "update_norm": unorm, "update_cos_prev": cos, **ev}
                history.append(row)
                print(f"[{label}] step {step:4d} loss {loss.item():.4f} "
                      f"val_acc {ev['val_acc']:.4f} |g| {gnorm:.3f} |u| {unorm:.4f}")
            step += 1

    final = evaluate(model, val_data, args.eval_bs, device)
    sharp = {
        "top_hessian_eig": top_hessian_eigenvalue(model, loss_on_batch, val_batch_for_sharpness),
        "perturbation": perturbation_sharpness(model, loss_on_batch, val_batch_for_sharpness,
                                               radius=args.sharp_radius, n_dirs=args.sharp_dirs),
        "adversarial": adversarial_sharpness(model, loss_on_batch, val_batch_for_sharpness,
                                             radius=args.sharp_radius),
    }
    return {"label": label, "history": history, "final": final, "sharpness": sharp}


def plot(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("train_loss", "train loss"), ("val_acc", "val accuracy"),
               ("grad_norm", "grad norm"), ("update_cos_prev", "cos(update_t, update_t-1)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for r in results:
            xs = [h["step"] for h in r["history"]]
            ys = [h[key] for h in r["history"]]
            ax.plot(xs, ys, marker=".", label=r["label"])
        ax.set_title(title); ax.set_xlabel("step"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "curves.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="prajjwal1/bert-tiny")
    ap.add_argument("--train_n", type=int, default=2000)
    ap.add_argument("--val_n", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--adamw_lr", type=float, default=5e-4)
    ap.add_argument("--muon_lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--sharp_radius", type=float, default=0.05)
    ap.add_argument("--sharp_dirs", type=int, default=10)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)
    print(f"device={device}  model={args.model}")

    # one fixed batch used for ALL sharpness measurements (same batch = fair)
    _, val_data, _ = load_data(args.model, args.train_n, args.val_n, args.max_len, args.seed)
    fixed = next(iterate_batches(val_data, 64, False, device))

    t0 = time.time()
    results = [train_one("adamw", args, device, fixed),
               train_one("muon", args, device, fixed)]
    print(f"\ntotal time {time.time()-t0:.1f}s")

    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # flat CSV of the per-step history
    import csv
    rows = [h for r in results for h in r["history"]]
    with open(os.path.join(args.outdir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot(results, args.outdir)

    print("\n=== FINAL ===")
    for r in results:
        s = r["sharpness"]
        print(f"{r['label']:12s} val_acc={r['final']['val_acc']:.4f} "
              f"val_loss={r['final']['val_loss']:.4f} "
              f"Hessian_lambda_max={s['top_hessian_eig']:.3f} "
              f"pert_mean={s['perturbation']['mean_increase']:.4f} "
              f"adv_worst={s['adversarial']['worst_increase']:.4f}")


if __name__ == "__main__":
    main()
