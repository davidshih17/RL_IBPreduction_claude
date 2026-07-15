#!/usr/bin/env python
"""What does the OLD (full/subs) model actually read from the substitution input?

Prior observations to reconcile: random subs CONTENT left outputs unchanged, but
EMPTY subs changed them. Architecture facts (sailir/classifier.py):
  - empty subs bypass the encoder entirely -> subs_emb = exact ZERO vector;
  - non-empty subs -> attention-pooled learned vector (+ positional encodings,
    so COUNT can leak);
  - training built the model with prime=1009, production workers with the
    DEFAULT prime 2^31-1 -> coefficients were sign-folded in training but never
    at inference (a real train/inference mismatch, quantified here too).

Ablations on real states (original 10x packed shard, val split, n_subs >= 1):
  A  baseline (true subs)
  B  replacement content shuffled across batch (keys + counts kept)
  C  keys AND replacements shuffled (counts kept)
  D  content collapsed: each sample's first sub tiled over its slots (count kept)
  E  subs emptied (mask off) -> exercises the zero-vector bypass
  F  truncation sweep k in {1,2,4,8,16} (count reduced, content true)
Metrics vs A: mean KL over the valid-action softmax, top-1 flip rate, and the
subs-embedding cosine/L2 shift. Plus: per-sample KL(E) bucketed by n_subs, and
the matched-vs-production PRIME comparison on identical inputs.
"""
import os, sys, random
import torch
import torch.nn.functional as F

BASE = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "training"))
from train_classifier import PackedDatasetV5, make_collate_fn, model_forward
from sailir.classifier import IBPActionClassifier

torch.manual_seed(0); random.seed(0)
N_IND, N_DEN, N_OPS = 11, 8, 18
CKPT = os.path.join(BASE, "checkpoints/pentagonbox_10x_loop_100/best_model.pt")
SHARD = os.path.join(BASE, "data/pentagonbox_10x_packed/shard_0/val.pt")
N_SAMPLES, BATCH = 256, 64

ck = torch.load(CKPT, map_location="cpu", weights_only=False)


def build(prime):
    m = IBPActionClassifier(prime=prime, n_indices=N_IND, n_denominators=N_DEN,
                            n_ibp_ops=N_OPS)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


model_matched = build(1009)          # as trained
model_prod = build(2147483647)       # as the production worker constructs it

data = torch.load(SHARD, weights_only=False)
ds = PackedDatasetV5(data)
idx = [i for i in range(len(ds))
       if ds.data["sub_offsets"][i + 1] > ds.data["sub_offsets"][i]]
random.shuffle(idx)
idx = idx[:N_SAMPLES]
collate = make_collate_fn(N_IND, N_DEN)
print(f"samples with >=1 sub: {len(idx)} (of {len(ds)} in shard_0/val)")


def ablate(batch, mode, k=None):
    b = {key: (v.clone() if torch.is_tensor(v) else v) for key, v in batch.items()}
    B = b["sub_mask"].shape[0]
    perm = torch.randperm(B)
    if mode == "A":
        pass
    elif mode == "B":                      # shuffle replacement content across batch
        for key in ("sub_repl_ints", "sub_repl_coeffs", "sub_repl_mask"):
            b[key] = b[key][perm]
    elif mode == "C":                      # keys too
        for key in ("sub_repl_ints", "sub_repl_coeffs", "sub_repl_mask", "sub_keys"):
            b[key] = b[key][perm]
    elif mode == "D":                      # collapse content, keep count
        for i in range(B):
            n = int(b["sub_mask"][i].sum())
            if n:
                b["sub_keys"][i, :n] = b["sub_keys"][i, 0]
                b["sub_repl_ints"][i, :n] = b["sub_repl_ints"][i, 0]
                b["sub_repl_coeffs"][i, :n] = b["sub_repl_coeffs"][i, 0]
                b["sub_repl_mask"][i, :n] = b["sub_repl_mask"][i, 0]
    elif mode == "E":                      # empty
        b["sub_mask"] = torch.zeros_like(b["sub_mask"])
    elif mode == "F":                      # truncate to k
        b["sub_mask"][:, k:] = False
    return b


def metrics(model, batch_a, batch_x):
    with torch.no_grad():
        la, _ = model_forward(model, batch_a)
        lx, _ = model_forward(model, batch_x)
        mask = batch_a["action_mask"]
        la = la.masked_fill(~mask, -1e9); lx = lx.masked_fill(~mask, -1e9)
        pa = F.log_softmax(la, dim=-1); px = F.log_softmax(lx, dim=-1)
        kl = (pa.exp() * (pa - px)).sum(-1)
        flip = (la.argmax(-1) != lx.argmax(-1)).float()
        ea = model.subs_enc(batch_a["sub_keys"], batch_a["sub_repl_ints"],
                            batch_a["sub_repl_coeffs"], batch_a["sub_repl_mask"],
                            batch_a["sub_mask"])
        ex = model.subs_enc(batch_x["sub_keys"], batch_x["sub_repl_ints"],
                            batch_x["sub_repl_coeffs"], batch_x["sub_repl_mask"],
                            batch_x["sub_mask"])
        cos = F.cosine_similarity(ea, ex, dim=-1)
    return kl, flip, cos


modes = [("B", "repl content shuffled"), ("C", "keys+repl shuffled"),
         ("D", "content collapsed (count kept)"), ("E", "subs EMPTIED"),
         ("F1", "truncate k=1"), ("F2", "truncate k=2"), ("F4", "truncate k=4"),
         ("F8", "truncate k=8"), ("F16", "truncate k=16")]
agg = {m: [[], [], []] for m, _ in modes}
kl_e_by_n = {}
prime_kl, prime_flip = [], []

for s in range(0, len(idx), BATCH):
    chunk = [ds[i] for i in idx[s:s + BATCH]]
    ba = collate(chunk)
    nsubs = ba["sub_mask"].sum(1)
    for m, _ in modes:
        k = int(m[1:]) if m.startswith("F") else None
        bx = ablate(ba, m[0] if k else m, k=k)
        kl, flip, cos = metrics(model_matched, ba, bx)
        agg[m][0] += kl.tolist(); agg[m][1] += flip.tolist(); agg[m][2] += cos.tolist()
        if m == "E":
            for n, v in zip(nsubs.tolist(), kl.tolist()):
                kl_e_by_n.setdefault(min(int(n), 20), []).append(v)
    # prime mismatch: same inputs, two constructions
    with torch.no_grad():
        lm, _ = model_forward(model_matched, ba)
        lp, _ = model_forward(model_prod, ba)
        mask = ba["action_mask"]
        lm = lm.masked_fill(~mask, -1e9); lp = lp.masked_fill(~mask, -1e9)
        pm = F.log_softmax(lm, -1); pp = F.log_softmax(lp, -1)
        prime_kl += (pm.exp() * (pm - pp)).sum(-1).tolist()
        prime_flip += (lm.argmax(-1) != lp.argmax(-1)).float().tolist()

print(f"\n{'ablation':<32} {'mean KL':>9} {'top1 flip%':>11} {'emb cos':>8}")
for m, desc in modes:
    kl, fl, cos = agg[m]
    print(f"{m:>3} {desc:<28} {sum(kl)/len(kl):9.4f} {100*sum(fl)/len(fl):10.1f}% "
          f"{sum(cos)/len(cos):8.3f}")

print("\nKL(EMPTY) by n_subs bucket:")
for n in sorted(kl_e_by_n):
    v = kl_e_by_n[n]
    print(f"  n_subs={n:>2}: mean KL {sum(v)/len(v):.4f}  (n={len(v)})")

print(f"\nPRIME mismatch (matched-1009 vs production-2^31): "
      f"mean KL {sum(prime_kl)/len(prime_kl):.4f}, top1 flips "
      f"{100*sum(prime_flip)/len(prime_flip):.1f}%")
