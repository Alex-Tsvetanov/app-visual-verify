# Extract frozen ResNet-18 (layer3) tile embeddings + pixel-diff tile stats for image pairs.
# Output: npz with A,B (N,100,256) fp16 embeddings, P (N,100,2) pixel stats, y, src, ids.
import io, json, os, sys, numpy as np, torch, torchvision, pyarrow.parquet as pq
from PIL import Image
import torch.nn.functional as F

S = os.environ["SCRATCH"]
CAP = os.environ["VERIFY_CAPTURES"]  # captures root, see src/verify/config.py
LABELS = json.load(open(f"{S}/ml/port/cpp/tools/parity/ml/data/labels.json"))
SZ, GRID = 640, 16
ENC = os.environ.get("ENC", "rn")
rng = np.random.default_rng(0)

if ENC == "rn":
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
    trunk2 = torch.nn.Sequential(m.layer2)
    trunk3 = torch.nn.Sequential(m.layer3)
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
else:
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def prep(im):
    im = im.convert("RGB").resize((SZ, SZ), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)
    g = np.asarray(im.convert("L"), dtype=np.float32) / 255
    return x, g

@torch.no_grad()
def encode(ims):
    xs, gs = zip(*[prep(im) for im in ims])
    x = (torch.stack(xs) - MEAN) / STD
    if ENC == "rn":
        f2 = trunk2(stem(x)); f3 = trunk3(f2)
        t = torch.cat([F.adaptive_avg_pool2d(f2, GRID), F.adaptive_avg_pool2d(f3, GRID)], 1)  # N,384,16,16
    else:
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        tok = dino.forward_features(x)["x_norm_patchtokens"]  # N,1024,384
        t = tok.transpose(1, 2).reshape(len(xs), -1, 32, 32)
        t = F.adaptive_avg_pool2d(t, GRID)
    t = t.flatten(2).transpose(1, 2)  # N,T,D
    return t.numpy().astype(np.float16), list(gs)

def pix_stats(ga, gb):
    d = np.abs(ga - gb)
    blk = SZ // GRID
    mean = d.reshape(GRID, blk, GRID, blk).mean((1, 3))
    frac = (d > 0.1).reshape(GRID, blk, GRID, blk).mean((1, 3))
    return np.stack([mean.ravel() * 10, frac.ravel() * 5], -1).astype(np.float32)

def run(pairs, out):
    """pairs: list of (id, src, label, imA, imB) with PIL images already decoded lazily."""
    A, B, P, Y, SRC, IDS = [], [], [], [], [], []
    for i in range(0, len(pairs), 8):
        chunk = pairs[i:i + 8]
        ims = [p[3]() for p in chunk] + [p[4]() for p in chunk]
        emb, gs = encode(ims)
        n = len(chunk)
        for j, p in enumerate(chunk):
            A.append(emb[j]); B.append(emb[n + j])
            P.append(pix_stats(gs[j], gs[n + j]))
            Y.append(p[2]); SRC.append(p[1]); IDS.append(p[0])
        print(f"{out}: {min(i + 8, len(pairs))}/{len(pairs)}", flush=True)
    np.savez(out, A=np.stack(A), B=np.stack(B), P=np.stack(P), y=np.array(Y), src=np.array(SRC), ids=np.array(IDS))

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "real":
        pairs = []
        for k, v in LABELS.items():
            page, plat, theme = k.split("/")
            y = int(v["label"] == "DIFFERENT")  # MINOR counts as SAME, as in analyze3.py
            pa = f"{CAP}/{plat}/maui/{page}_{theme}.png"
            pb = f"{CAP}/{plat}/cpp/{page}_{theme}.png"
            pairs.append((k, "real", y, (lambda p: lambda: Image.open(p))(pa), (lambda p: lambda: Image.open(p))(pb)))
        run(pairs, f"{S}/proto/real_{ENC}.npz")
    else:
        import pyarrow.parquet as pq
        t = pq.read_table(f"{S}/public/diffspot_test.parquet", columns=["id", "image_before", "image_after", "task_type", "difficulty"]) if False else pq.read_table(f"{S}/public/diffspot_test.parquet", columns=["image_before", "image_after", "task_type", "difficulty"])
        tt = t.column("task_type").to_pylist(); dif = t.column("difficulty").to_pylist()
        nod = [i for i, x in enumerate(tt) if x == "no_diff"]
        has = {d: [i for i, x in enumerate(tt) if x == "visual_diff" and dif[i] == d] for d in ["easy", "medium", "hard"]}
        pick = nod + [i for d in has for i in rng.choice(has[d], 333, replace=False)]
        bef = t.column("image_before").to_pylist(); aft = t.column("image_after").to_pylist()
        dec = lambda s: Image.open(io.BytesIO(s["bytes"]))
        pairs = [(f"ds{i}", "diffspot", int(tt[i] == "visual_diff"),
                  (lambda s: lambda: dec(s))(bef[i]), (lambda s: lambda: dec(s))(aft[i])) for i in pick]
        run(pairs, f"{S}/proto/diffspot_{ENC}.npz")
