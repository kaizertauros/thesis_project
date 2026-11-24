import copy
import pickle
import random
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

DATA_PATH = "ai2thor_dataset_masked.pkl"   # <- the file your regenerated data script wrote
TARGET_SIZE = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ======================= data =======================
with open(DATA_PATH, "rb") as f:
    dataset = pickle.load(f)
assert "mask" in dataset[0], "No masks in this pickle: regenerate with the masked data script."


def resize_sample(s):
    out = {**s,
           "rgb": cv2.resize(s["rgb"], TARGET_SIZE, interpolation=cv2.INTER_LINEAR),
           "depth": cv2.resize(s["depth"], TARGET_SIZE, interpolation=cv2.INTER_LINEAR),
           "mask": cv2.resize(s["mask"], TARGET_SIZE, interpolation=cv2.INTER_NEAREST)}
    return out


resized_dataset = [resize_sample(s) for s in dataset]
del dataset


def strat_key(s):
    return f"{int(s['movable'])}_{int(s['unsafe'])}"


def group_split(samples, held_out_frac, group_by="object_type", seed=42):
    y = [strat_key(s) for s in samples]
    groups = [s[group_by] for s in samples]          # object_type, object_id or scene
    sgkf = StratifiedGroupKFold(n_splits=round(1 / held_out_frac), shuffle=True, random_state=seed)
    a_idx, b_idx = next(sgkf.split(samples, y, groups))
    return [samples[i] for i in a_idx], [samples[i] for i in b_idx]


def group_cv_splits(samples, n_folds=5, group_by="object_type", seed=42):
    y = [strat_key(s) for s in samples]
    groups = [s[group_by] for s in samples]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in sgkf.split(samples, y, groups):
        yield [samples[i] for i in tr_idx], [samples[i] for i in te_idx]


# ======================= dataset / model =======================
class MovabilityDataset(Dataset):
    def __init__(self, samples, condition="fused", train=False, masked=True):
        self.samples, self.condition, self.train, self.masked = samples, condition, train, masked

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rgb = s["rgb"].astype(np.float32) / 255.0
        depth = (np.clip(s["depth"].astype(np.float32), 0, 5.0) / 5.0)[..., None]

        if self.masked:                       # blank out everything except the target object
            m = s["mask"][..., None].astype(np.float32)
            rgb, depth = rgb * m, depth * m

        if self.condition == "rgb":
            img = rgb
        elif self.condition == "depth":
            img = np.repeat(depth, 3, axis=-1)
        else:
            img = np.concatenate([rgb, depth], axis=-1)

        img = torch.from_numpy(img).permute(2, 0, 1).float()

        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
            img = TF.rotate(img, random.uniform(-10, 10))
            if self.condition in ("rgb", "fused"):
                rgb_part = TF.adjust_brightness(img[:3], random.uniform(0.8, 1.2))
                rgb_part = TF.adjust_contrast(rgb_part, random.uniform(0.8, 1.2))
                img = torch.cat([rgb_part, img[3:]], dim=0)

        return img, torch.tensor(float(s["movable"])), torch.tensor(float(s["unsafe"]))


class MovabilityNet(nn.Module):
    def __init__(self, in_channels=4, pretrained=True, freeze_until="layer4"):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        old = base.conv1
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=(old.bias is not None))
        with torch.no_grad():
            if in_channels == 3:
                new.weight.copy_(old.weight)
            else:
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        base.conv1 = new
        num_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base

        if freeze_until is not None:          # conv1 always trains; freeze until `freeze_until`
            unfreeze = False
            for name, module in self.backbone.named_children():
                if name == "conv1":
                    continue
                if name == freeze_until:
                    unfreeze = True
                for p in module.parameters():
                    p.requires_grad = unfreeze

        self.dropout = nn.Dropout(p=0.4)
        self.movable_head = nn.Linear(num_features, 1)
        self.unsafe_head = nn.Linear(num_features, 1)

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.movable_head(f).squeeze(-1), self.unsafe_head(f).squeeze(-1)


# ======================= metrics / helpers =======================
def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def compute_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    both = len(np.unique(y)) > 1
    return {"auc": safe_auc(y, p),
            "ap": average_precision_score(y, p) if both else float("nan"),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0)}


def best_f1_threshold(y, p, default=0.5):
    if len(np.unique(y)) < 2:                 # no positives (or no negatives) in this split
        return default
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-8, None)
    return float(thr[np.argmax(f1[:-1])])


def make_criteria(train_samples):
    n = len(train_samples)
    nm = sum(s["movable"] for s in train_samples)
    nu = sum(s["unsafe"] for s in train_samples)
    pw_m = torch.tensor((n - nm) / max(nm, 1), dtype=torch.float32, device=device)
    pw_u = torch.tensor((n - nu) / max(nu, 1), dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pw_m), nn.BCEWithLogitsLoss(pos_weight=pw_u)


def run_epoch(model, loader, crits, optimizer=None):
    m_crit, u_crit = crits
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total, n = 0.0, 0
    ym, yu, pm, pu = [], [], [], []
    with torch.set_grad_enabled(train_mode):
        for imgs, movable, unsafe in loader:
            imgs, movable, unsafe = imgs.to(device), movable.to(device), unsafe.to(device)
            m_logit, u_logit = model(imgs)
            loss = m_crit(m_logit, movable) + u_crit(u_logit, unsafe)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
            ym.append(movable.cpu()); yu.append(unsafe.cpu())
            pm.append(torch.sigmoid(m_logit).detach().cpu())
            pu.append(torch.sigmoid(u_logit).detach().cpu())
    cat = lambda xs: torch.cat(xs).numpy()
    return total / n, cat(ym).astype(int), cat(pm), cat(yu).astype(int), cat(pu)


def run_experiment(condition, tr_s, te_s, masked=True, epochs=5, lr=3e-4, batch_size=32,
                   seed=0, freeze_until="layer4"):
    set_seed(seed)
    in_ch = 4 if condition == "fused" else 3
    tr = DataLoader(MovabilityDataset(tr_s, condition, True, masked), batch_size=batch_size, shuffle=True)
    tr_eval = DataLoader(MovabilityDataset(tr_s, condition, False, masked), batch_size=batch_size)
    te = DataLoader(MovabilityDataset(te_s, condition, False, masked), batch_size=batch_size)

    crits = make_criteria(tr_s)
    model = MovabilityNet(in_channels=in_ch, freeze_until=freeze_until).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ in range(epochs):                      # fixed schedule, the last epoch is used
        run_epoch(model, tr, crits, opt)
        sched.step()

    _, tm_y, tm_p, tu_y, tu_p = run_epoch(model, tr_eval, crits)
    _, m_y, m_p, u_y, u_p = run_epoch(model, te, crits)
    THR = 0.5
    return {
        "non_movable": compute_metrics(1 - m_y, 1 - m_p, THR),
        "unsafe": compute_metrics(u_y, u_p, THR),
        "train_auc": {"nm": safe_auc(1 - tm_y, 1 - tm_p), "u": safe_auc(tu_y, tu_p)},
        "raw": {"samples": te_s, "y_nm": 1 - m_y, "s_nm": 1 - m_p, "thr_nm": THR,
                "y_u": u_y, "s_u": u_p, "thr_u": THR},
    }


def run_cv(condition, masked=True, n_folds=5, seed=0, **kw):
    folds = []
    for k, (tr, te) in enumerate(group_cv_splits(resized_dataset, n_folds, seed=42 + seed)):
        assert not ({s["object_type"] for s in tr} & {s["object_type"] for s in te})
        folds.append(run_experiment(condition, tr, te, masked, seed=seed, **kw))
        print(f"[{condition} seed={seed}] fold {k+1}/{n_folds} done "
              f"(test n={len(te)}, non-movable={sum(not s['movable'] for s in te)})")
    return folds    


def summarize(folds, task):
    return {k: (np.nanmean([f[task][k] for f in folds]), np.nanstd([f[task][k] for f in folds]))
            for k in folds[0][task]}


def pooled_metrics(folds, tag):            # tag: "nm" or "u"
    y = np.concatenate([f["raw"][f"y_{tag}"] for f in folds])
    s = np.concatenate([f["raw"][f"s_{tag}"] for f in folds])
    pred = np.concatenate([(f["raw"][f"s_{tag}"] >= f["raw"][f"thr_{tag}"]).astype(int) for f in folds])
    return {"base_rate": y.mean(), "auc": safe_auc(y, s),
            "ap": average_precision_score(y, s),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0)}


def pooled_raw(folds):
    r = [f["raw"] for f in folds]
    samples = [s for x in r for s in x["samples"]]
    P = {k: np.concatenate([x[k] for x in r]) for k in ["y_nm", "s_nm", "y_u", "s_u"]}
    T = {k: float(np.mean([x[k] for x in r])) for k in ["thr_nm", "thr_u"]}
    return samples, P, T


def type_baseline(tr_s, te_s, key, flip=False):
    f = (lambda s: 1.0 - float(s[key])) if flip else (lambda s: float(s[key]))
    d = defaultdict(list)
    for s in tr_s:
        d[s["object_type"]].append(f(s))
    prior = np.mean([f(s) for s in tr_s])
    p = np.array([np.mean(d[s["object_type"]]) if s["object_type"] in d else prior for s in te_s])
    y = np.array([int(f(s)) for s in te_s])
    return compute_metrics(y, p, 0.5)


def per_type_report(samples, y, p, thr):
    df = pd.DataFrame({"type": [s["object_type"] for s in samples], "y": y,
                       "correct": (p >= thr).astype(int) == y})
    return (df.groupby("type")
              .agg(n=("y", "size"), pos_rate=("y", "mean"), acc=("correct", "mean"))
              .sort_values("n", ascending=False))


def show_type_scores(samples, probs, obj_type, task="movable", max_n=12, cols=6):
    idx = sorted([i for i, s in enumerate(samples) if s["object_type"] == obj_type],
                 key=lambda i: probs[i])
    if len(idx) > max_n:
        idx = [idx[j] for j in np.linspace(0, len(idx) - 1, max_n).astype(int)]
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, idx):
        s = samples[i]
        ax.imshow(s["rgb"] * s["mask"][..., None])        # show exactly what the model sees
        ax.set_title(f"score={probs[i]:.2f}\ntrue={int(s[task])}  mass={s['mass']:.1f}kg", fontsize=8)
    fig.suptitle(f"{obj_type}: predicted {task} score (low to high)")
    plt.tight_layout()
    plt.savefig(f"scores_{obj_type}_{task}.png", dpi=150)
    plt.show()


def plot_within_type(samples, probs, obj_type, thr, task="movable"):
    df = pd.DataFrame({"object_id": [s["object_id"] for s in samples],
                       "type": [s["object_type"] for s in samples], "p": probs})
    df = df[df["type"] == obj_type]
    order = df.groupby("object_id")["p"].mean().sort_values().index.tolist()
    pos = {oid: k for k, oid in enumerate(order)}
    plt.figure(figsize=(9, 4))
    plt.scatter(df["object_id"].map(pos), df["p"], alpha=0.7)
    plt.axhline(thr, color="red", linestyle="--", label=f"threshold {thr:.2f}")
    plt.xlabel("individual objects (sorted by mean score)")
    plt.ylabel(f"predicted {task} score")
    plt.title(f"{obj_type}: score per object (dots = individual views)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"spread_{obj_type}_{task}.png", dpi=150)
    plt.show()




# ======================= leakage demo (optional) =======================
for gb in ["object_id", "object_type"]:
    tv, te_ = group_split(resized_dataset, 0.15, group_by=gb)
    print(f"type-only baseline, split by {gb:11s} unsafe AUC = "
          f"{type_baseline(tv, te_, 'unsafe')['auc']:.3f}   "
          f"non-movable AUC = {type_baseline(tv, te_, 'movable', flip=True)['auc']:.3f}")

# ======================= main experiment =======================
MASKED_OPTIONS = [True, False]
SEEDS = [0, 1, 2]
CONDS = ["rgb", "depth", "fused"]
results = {(c, s): run_cv(c, True, seed=s) for c in CONDS for s in SEEDS}

for c in CONDS:
    for task, tag in [("non_movable", "nm"), ("unsafe", "u")]:
        ms = [pooled_metrics(results[(c, s)], tag) for s in SEEDS]
        print(f"{c:6s} {task:11s} pooled, mean±std over {len(SEEDS)} seeds: " +
              "  ".join(f"{k}={np.mean([m[k] for m in ms]):.3f}±{np.std([m[k] for m in ms]):.3f}"
                        for k in ms[0]))
        tr_auc = np.nanmean([f["train_auc"][tag] for s in SEEDS for f in results[(c, s)]])
        te_auc = np.nanmean([f[task]["auc"] for s in SEEDS for f in results[(c, s)]])
        print(f"{'':6s} {'':11s} train AUC {tr_auc:.3f} vs per-fold test AUC {te_auc:.3f}")

# ======================= score plots =======================
folds = results[("fused", True)]
samples, P, T = pooled_raw(folds)
movable_score = 1 - P["s_nm"]

show_type_scores(samples, movable_score, "Chair", "movable")
plot_within_type(samples, movable_score, "Chair", 1 - T["thr_nm"], "movable")
show_type_scores(samples, P["s_u"], "Chair", "unsafe")
plot_within_type(samples, P["s_u"], "Chair", T["thr_u"], "unsafe")

df = pd.DataFrame({"oid": [s["object_id"] for s in samples],
                   "type": [s["object_type"] for s in samples], "p": movable_score})
c = df[df.type == "Chair"]
print("Chair movable score: within-object var =", round(c.groupby("oid")["p"].var().mean(), 4),
      " between-object var =", round(c.groupby("oid")["p"].mean().var(), 4))

print(per_type_report(samples, P["y_u"], P["s_u"], T["thr_u"]).head(15))

MIN_VIS = 200     # try 100, 200, 300 (same value for train and test)
resized_dataset = [s for s in resized_dataset if s["visible_px"] >= MIN_VIS]
print(len(resized_dataset), "samples at MIN_VIS =", MIN_VIS)