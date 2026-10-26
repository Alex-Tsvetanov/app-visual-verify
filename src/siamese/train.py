# Siamese tile-comparison head on frozen ResNet-18 features.
# E1: pretrain on DiffSpot (public), zero-shot score on the 108 real labels.
# E2: fine-tune from E1 on the original 76, test on the 32 new labels; LOO over 108.
# E3: baseline, train from scratch on 76 (held-out) and LOO over 108.
import json, math, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
torch.manual_seed(0); np.random.seed(0)
S = os.environ["SCRATCH"]; D = f"{S}/proto"
torch.set_num_threads(os.cpu_count())

def load(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in ["A", "B", "P", "y", "ids"]}

class Head(nn.Module):
    def __init__(self, d=256, h=64):
        super().__init__()
        self.tile = nn.Sequential(nn.Linear(2 * d + 2, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def forward(self, A, B, P):
        A, B = F.normalize(A, dim=-1), F.normalize(B, dim=-1)
        x = torch.cat([(A - B).abs(), A * B, P], -1)
        t = self.tile(x).squeeze(-1)                       # N,T tile logits
        return torch.logsumexp(5 * t, 1) / 5                 # soft-max pooling over tiles

def tens(d):
    return [torch.tensor(d[k], dtype=torch.float32) if k != "y" else torch.tensor(d[k], dtype=torch.float32) for k in ["A", "B", "P", "y"]]

def fit(model, data, epochs, lr, wd=1e-4, bs=64):
    A, B, P, y = tens(data)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pw = torch.tensor((len(y) - y.sum()) / max(1.0, y.sum()))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), bs):
            j = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(model(A[j], B[j], P[j]), y[j], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model

@torch.no_grad()
def score(model, data):
    A, B, P, _ = tens(data)
    return torch.sigmoid(model(A, B, P)).numpy()

def boot_ci(y, s, n=2000, seed=0):
    r = np.random.default_rng(seed); out = []; idx = np.arange(len(y))
    for _ in range(n):
        b = r.choice(idx, len(idx))
        if len(set(y[b])) == 2: out.append(roc_auc_score(y[b], s[b]))
    return np.percentile(out, [2.5, 97.5])

def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}

def report(name, y, s):
    lo, hi = boot_ci(y, s)
    print(f"{name}: AUC {roc_auc_score(y, s):.3f} ({lo:.2f}-{hi:.2f}) acc {((s >= .5) == y).mean():.2f} n={len(y)} pos={int(y.sum())}", flush=True)

real = load(f"{D}/real.npz")
orig = set(json.load(open(f"{S}/ml/port/cpp/tools/parity/ml/data/orig76.json")))
is_orig = np.array([i in orig for i in real["ids"]])
print(f"real: {len(real['y'])} pairs, orig76 {is_orig.sum()}, new {(~is_orig).sum()}")

# ---- E1: pretrain on DiffSpot
if os.path.exists(f"{D}/diffspot.npz"):
    ds = load(f"{D}/diffspot.npz")
    rs = np.random.default_rng(1); perm = rs.permutation(len(ds["y"])); cut = int(.85 * len(perm))
    tr, va = subset(ds, perm[:cut]), subset(ds, perm[cut:])
    pre = fit(Head(), tr, epochs=25, lr=1e-3)
    report("E1 DiffSpot val (synthetic)", va["y"], score(pre, va))
    report("E1 zero-shot on 108 real labels", real["y"], score(pre, real))
    pre_state = pre.state_dict()
else:
    pre_state = None
    print("no DiffSpot features; skipping pretrain")

# ---- E2/E3: held-out 32 and LOO over 108
def finetune_from(state, data, epochs=30, lr=3e-4):
    m = Head()
    if state is not None: m.load_state_dict(state)
    return fit(m, data, epochs=epochs, lr=lr if state is not None else 1e-3)

for name, state in [("E2 pretrained+finetune", pre_state), ("E3 from scratch", None)]:
    m = finetune_from(state, subset(real, is_orig))
    report(f"{name} HELD-OUT 32", real["y"][~is_orig], score(m, subset(real, ~is_orig)))
    s = np.zeros(len(real["y"]))
    for i in range(len(s)):
        mask = np.ones(len(s), bool); mask[i] = False
        mi = finetune_from(state, subset(real, mask))
        s[i] = score(mi, subset(real, np.arange(len(s)) == i))[0]
    report(f"{name} LOO 108", real["y"], s)
    np.save(f"{D}/loo_{name.split()[0]}.npy", s)
