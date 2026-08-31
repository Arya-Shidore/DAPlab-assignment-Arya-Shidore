"""
Build real saved tensors for Task 5, then run select_experiment.py on them.

Instantiation of the Wyner-Ziv mapping with a real encoder:
  worker  sees the FULL sentence            -> X = mean-pooled last hidden state
  receiver sees only the FIRST HALF          -> Y = mean-pooled last hidden state
                                                (its "side information")
  label   = SST-2 sentiment

The receiver already knows what the first half implies; the interesting
question is what the worker should transmit about the SECOND half. This is
exactly the side-information setting.

Run:
    pip install torch transformers datasets numpy
    python build_tensors.py --model prajjwal1/bert-tiny --n 3000 --out sst2
    python select_experiment.py --real_prefix sst2 --k 8 --seeds 5
"""

import argparse, numpy as np, torch


def pooled_hidden(model, tok, texts, device, max_len, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt").to(device)
            hs = model(**enc).last_hidden_state           # (b, t, h)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy())
    return np.concatenate(out, 0)


def first_half(text):
    w = text.split()
    return " ".join(w[: max(1, len(w) // 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="prajjwal1/bert-tiny")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--out", default="sst2")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
    except ValueError:
        # ponytail: bert-tiny ships no fast-tokenizer file; bert-base-uncased
        # uses the identical wordpiece vocab, so this is exact, not lossy.
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    try:
        model = AutoModel.from_pretrained(args.model).to(device)
    except ValueError:
        # ponytail: bert-tiny's config.json predates the model_type field AutoConfig now requires.
        from transformers import BertModel
        model = BertModel.from_pretrained(args.model).to(device)

    ds = load_dataset("nyu-mll/glue", "sst2")["train"].shuffle(seed=0).select(range(args.n))
    full = [s for s in ds["sentence"]]
    half = [first_half(s) for s in full]
    labels = np.array(ds["label"])

    X = pooled_hidden(model, tok, full, device, args.max_len)   # worker (full)
    Y = pooled_hidden(model, tok, half, device, args.max_len)   # receiver side info
    np.save(f"{args.out}_X.npy", X)
    np.save(f"{args.out}_Y.npy", Y)
    np.save(f"{args.out}_labels.npy", labels)
    print(f"saved {args.out}_X.npy {X.shape}, _Y.npy {Y.shape}, _labels.npy {labels.shape}")


if __name__ == "__main__":
    main()
