import pickle
import random
import os
import re
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

# configuration
DATA_PATH = "ai2thor_dataset_masked.pkl"
TARGET_SIZE = (224, 224)
MIN_VIS = 200            # drop samples with fewer visible px
MASKED = True            # blank out everything except the target object
CONDS = ["rgb", "depth", "fused"]

QUICK = False            # True: 1 seed, for a fast sanity run
SEEDS = [0] if QUICK else [0, 1, 2]
N_FOLDS = 5
EPOCHS = 5
LR = 3e-4
BATCH_SIZE = 32
THR = 0.5                # decision threshold for accuracy/precision/recall/f1
VERBOSE_EPOCHS = True    # False: one summary line per fold
MIN_TYPE_N = 10          # by-type tables only show types with at least this many samples
TYPE_ROWS = 60           # rows shown in the by-type tables
PLOT_MIN_VIEWS = 1       # per-object plots are made for types with at least this many views
SHOW_PLOTS = False       # plots are always saved to png; True also opens windows (don't with many types)
RUN_LEAKAGE_DEMO = False

# split / model / regularisation settings
GROUP_BY = "scene"   # "object_type" = unseen types | "scene" = unseen scenes, types seen | "object_id"
FREEZE_UNTIL = "layer4"    # "layer3" | "layer4" | "all" (only conv1 + heads train) | None (train everything)
AUG = "base"               # "base" | "strong"
WEIGHT_DECAY = 1e-2

# what to run
RUN_MAIN = True            # the full rgb/depth/fused experiment, summary, by-type tables and plots
RUN_ABLATIONS = False      # overfitting experiments (about as expensive as a full run)
ABL_COND = "rgb"           # one condition is enough to see the effect
ABL_SEEDS = [0, 1]
ABLATIONS = {
    "baseline (5 ep, layer4, base aug)": dict(epochs=5, freeze_until="layer4", aug="base"),
    "3 epochs":                          dict(epochs=3, freeze_until="layer4", aug="base"),
    "strong aug":                        dict(epochs=5, freeze_until="layer4", aug="strong"),
    "freeze layer4":                     dict(epochs=5, freeze_until="all", aug="base"),
    "freeze layer4 + strong aug":        dict(epochs=5, freeze_until="all", aug="strong"),
    "weight decay 0.1":                  dict(epochs=5, freeze_until="layer4", aug="base", weight_decay=0.1),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PLOT_DIR = os.path.join("output", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)      # no error if it already exists; also creates "output"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# data
with open(DATA_PATH, "rb") as f:
    dataset = pickle.load(f)
assert "mask" in dataset[0], "No masks in this pickle: regenerate with the masked data script."

n_before = len(dataset)
dataset = [s for s in dataset if s["visible_px"] >= MIN_VIS]
print(f"{len(dataset)}/{n_before} samples kept at MIN_VIS = {MIN_VIS}")


def resize_sample(s):
    return {**s,
            "rgb": cv2.resize(s["rgb"], TARGET_SIZE, interpolation=cv2.INTER_LINEAR),
            "depth": cv2.resize(s["depth"], TARGET_SIZE, interpolation=cv2.INTER_LINEAR),
            "mask": cv2.resize(s["mask"], TARGET_SIZE, interpolation=cv2.INTER_NEAREST)}


resized_dataset = [resize_sample(s) for s in dataset]
del dataset


def strat_key(s):
    return f"{int(s['movable'])}_{int(s['unsafe'])}"


def seeded_group_cv_splits(samples, n_folds, seed, group_by="object_type"):
    """Grouped, label-balanced K-fold where the seed really changes the split."""
    rng = np.random.RandomState(seed)
    classes = sorted({strat_key(s) for s in samples})
    cidx = {c: i for i, c in enumerate(classes)}

    group_members = defaultdict(list)
    for i, s in enumerate(samples):
        group_members[s[group_by]].append(i)

    group_counts = {}
    for g, idxs in group_members.items():
        v = np.zeros(len(classes))
        for i in idxs:
            v[cidx[strat_key(samples[i])]] += 1
        group_counts[g] = v
    total = np.maximum(sum(group_counts.values()), 1)

    # big groups first, with random jitter so each seed gives a different split
    order = sorted(group_members, key=lambda g: -len(group_members[g]) * rng.uniform(0.6, 1.4))

    fold_counts = np.zeros((n_folds, len(classes)))
    assign = {}
    for g in order:
        costs = []
        for f in range(n_folds):
            fold_counts[f] += group_counts[g]
            costs.append(np.mean(np.std(fold_counts / total, axis=0)) + rng.uniform(0, 1e-3))
            fold_counts[f] -= group_counts[g]
        best = int(np.argmin(costs))
        fold_counts[best] += group_counts[g]
        assign[g] = best

    for f in range(n_folds):
        te_idx = [i for i, s in enumerate(samples) if assign[s[group_by]] == f]
        tr_idx = [i for i, s in enumerate(samples) if assign[s[group_by]] != f]
        yield [samples[i] for i in tr_idx], [samples[i] for i in te_idx]


# truth table (post-filter)
def label_truth_table(samples, title):
    d = pd.DataFrame({"movable": [s["movable"] for s in samples],
                      "unsafe": [s["unsafe"] for s in samples]})
    print(f"\n{title}")
    print(pd.crosstab(d["movable"], d["unsafe"], margins=True))


label_truth_table(resized_dataset, "Label truth table of the data used here (movable x unsafe):")


# dataset / model
class MovabilityDataset(Dataset):
    def __init__(self, samples, condition="fused", train=False, masked=True, aug="base"):
        self.samples, self.condition, self.train = samples, condition, train
        self.masked, self.aug = masked, aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rgb = s["rgb"].astype(np.float32) / 255.0
        depth = (np.clip(s["depth"].astype(np.float32), 0, 5.0) / 5.0)[..., None]

        if self.masked:     # blank out everything except the target object
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
            strong = self.aug == "strong"
            if random.random() < 0.5:
                img = TF.hflip(img)
            if strong:
                size = TARGET_SIZE[0]
                tx, ty = [int(random.uniform(-0.1, 0.1) * size) for _ in range(2)]
                img = TF.affine(img, angle=random.uniform(-20, 20), translate=[tx, ty],
                                scale=random.uniform(0.8, 1.2), shear=[0.0, 0.0])
            else:
                img = TF.rotate(img, random.uniform(-10, 10))
            if self.condition in ("rgb", "fused"):
                lo, hi = (0.6, 1.4) if strong else (0.8, 1.2)
                rgb_part = TF.adjust_brightness(img[:3], random.uniform(lo, hi))
                rgb_part = TF.adjust_contrast(rgb_part, random.uniform(lo, hi))
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

        # conv1 always trains; layers before `freeze_until` are frozen.
        # "all" matches no layer name, so everything except conv1 stays frozen.
        if freeze_until is not None:
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


# metrics / helpers
def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def compute_metrics(y, p, thr=THR):
    pred = (p >= thr).astype(int)
    both = len(np.unique(y)) > 1
    return {"accuracy": float((pred == y).mean()),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "auc": safe_auc(y, p),
            "ap": average_precision_score(y, p) if both else float("nan")}


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


def run_experiment(condition, tr_s, te_s, masked=True, epochs=EPOCHS, lr=LR,
                   batch_size=BATCH_SIZE, seed=0, fold=0, freeze_until=FREEZE_UNTIL,
                   aug=AUG, weight_decay=WEIGHT_DECAY):
    set_seed(seed)
    tag = f"{condition} s{seed} f{fold + 1}"
    in_ch = 4 if condition == "fused" else 3
    tr = DataLoader(MovabilityDataset(tr_s, condition, True, masked, aug), batch_size=batch_size, shuffle=True)
    tr_eval = DataLoader(MovabilityDataset(tr_s, condition, False, masked), batch_size=batch_size)
    te = DataLoader(MovabilityDataset(te_s, condition, False, masked), batch_size=batch_size)

    crits = make_criteria(tr_s)
    model = MovabilityNet(in_channels=in_ch, freeze_until=freeze_until).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs + 1):               # the LAST epoch is what gets reported
        train_loss, *_ = run_epoch(model, tr, crits, opt)
        sched.step()
        # "val" = the held-out fold. Only monitored, not used for training or early stopping
        val_loss, m_y, m_p, u_y, u_p = run_epoch(model, te, crits)
        if VERBOSE_EPOCHS:
            print(f"[{tag}] ep {ep}  train {train_loss:.3f}  val {val_loss:.3f}  "
                  f"val_auc nm {safe_auc(1 - m_y, 1 - m_p):.3f}  u {safe_auc(u_y, u_p):.3f}")

    _, tm_y, tm_p, tu_y, tu_p = run_epoch(model, tr_eval, crits)    # clean (no augmentation) train pass
    return {
        "non_movable": compute_metrics(1 - m_y, 1 - m_p),
        "unsafe": compute_metrics(u_y, u_p),
        "train_auc": {"nm": safe_auc(1 - tm_y, 1 - tm_p), "u": safe_auc(tu_y, tu_p)},
        "raw": {"samples": te_s, "y_nm": 1 - m_y, "s_nm": 1 - m_p, "y_u": u_y, "s_u": u_p},
    }


def run_cv(condition, masked=True, n_folds=N_FOLDS, seed=0, **kw):
    folds = []
    for k, (tr, te) in enumerate(seeded_group_cv_splits(resized_dataset, n_folds, seed=42 + seed,
                                                       group_by=GROUP_BY)):
        assert not ({s[GROUP_BY] for s in tr} & {s[GROUP_BY] for s in te})
        folds.append(run_experiment(condition, tr, te, masked, seed=seed, fold=k, **kw))
        m = folds[-1]
        print(f"[{condition} seed={seed}] fold {k + 1}/{n_folds} done "
              f"(test n={len(te)}, non-movable={sum(not s['movable'] for s in te)})  "
              f"nm auc {m['non_movable']['auc']:.3f}  unsafe auc {m['unsafe']['auc']:.3f}")
    return folds


def pooled(folds):
    """Concatenate the held-out predictions of all folds."""
    r = [f["raw"] for f in folds]
    samples = [s for x in r for s in x["samples"]]
    P = {k: np.concatenate([x[k] for x in r]) for k in ["y_nm", "s_nm", "y_u", "s_u"]}
    return samples, P


def type_baseline(tr_s, te_s, key, flip=False):
    f = (lambda s: 1.0 - float(s[key])) if flip else (lambda s: float(s[key]))
    d = defaultdict(list)
    for s in tr_s:
        d[s["object_type"]].append(f(s))
    prior = np.mean([f(s) for s in tr_s])
    p = np.array([np.mean(d[s["object_type"]]) if s["object_type"] in d else prior for s in te_s])
    y = np.array([int(f(s)) for s in te_s])
    return compute_metrics(y, p)


def type_baseline_pooled(group_by, seed=42):
    """AUC of a predictor that only knows the object type (no image), pooled over the CV folds."""
    out = {}
    for name, key, flip in [("non_movable", "movable", True), ("unsafe", "unsafe", False)]:
        ys, ps = [], []
        for tr, te in seeded_group_cv_splits(resized_dataset, N_FOLDS, seed, group_by):
            f = (lambda s: 1.0 - float(s[key])) if flip else (lambda s: float(s[key]))
            d = defaultdict(list)
            for s in tr:
                d[s["object_type"]].append(f(s))
            prior = np.mean([f(s) for s in tr])
            ps += [np.mean(d[s["object_type"]]) if s["object_type"] in d else prior for s in te]
            ys += [int(f(s)) for s in te]
        out[name] = safe_auc(np.array(ys), np.array(ps))
    return out


def per_type_metrics(samples, y, s, thr=THR, min_n=MIN_TYPE_N, n_rep=1):
    """n_rep = how many times each sample appears (number of seeds pooled)."""
    types = np.array([x["object_type"] for x in samples])
    rows = []
    for t in np.unique(types):
        m = types == t
        n_unique = m.sum() // n_rep
        if n_unique < min_n:
            continue
        yt, pred = y[m], (s[m] >= thr).astype(int)
        pos, neg = yt == 1, yt == 0
        mixed = pos.any() and neg.any()
        rows.append({
            "type": t, "n": int(n_unique), "pos_rate": yt.mean(),
            "acc": (pred == yt).mean(),
            "recall": pred[pos].mean() if pos.any() else np.nan,      # hit rate on true positives
            "fpr": pred[neg].mean() if neg.any() else np.nan,         # FP rate
            "precision": precision_score(yt, pred, zero_division=0) if mixed else np.nan,
            "f1": f1_score(yt, pred, zero_division=0) if mixed else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).set_index("type")


# plotting
def safe_name(t):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(t))


def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _group_by_object(samples, probs, obj_type):
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        if s["object_type"] == obj_type:
            groups[s["object_id"]].append(i)
    # objects ordered by mean predicted score
    return sorted(groups.items(), key=lambda kv: np.mean([probs[i] for i in kv[1]]))


def show_object_scores(samples, probs, obj_type, thr, task="movable",
                       max_objects=12, max_views=8):
    """One row per object; its views sorted by score (what the model sees)."""
    groups = _group_by_object(samples, probs, obj_type)
    if not groups:
        return
    if len(groups) > max_objects:
        keep = np.linspace(0, len(groups) - 1, max_objects).astype(int)
        groups = [groups[j] for j in keep]

    n_cols = min(max_views, max(len(idx) for _, idx in groups))
    fig, axes = plt.subplots(len(groups), n_cols,
                             figsize=(2.0 * n_cols, 2.3 * len(groups)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for r, (oid, idx) in enumerate(groups):
        idx = sorted(idx, key=lambda i: probs[i])
        if len(idx) > n_cols:
            idx = [idx[j] for j in np.linspace(0, len(idx) - 1, n_cols).astype(int)]
        mean_p = np.mean([probs[i] for i in idx])
        for c, i in enumerate(idx):
            ax, s = axes[r, c], samples[i]
            ax.imshow(s["rgb"] * s["mask"][..., None])
            correct = int(probs[i] >= thr) == int(s[task])
            ax.set_title(f"{probs[i]:.2f}", fontsize=8, color="green" if correct else "red")
        axes[r, 0].text(-0.05, 0.5, f"{str(oid)[:22]}\ntrue={int(samples[idx[0]][task])}  "
                        f"mean={mean_p:.2f}\nmass={samples[idx[0]]['mass']:.1f}kg",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)

    fig.suptitle(f"{obj_type}: predicted {task} score per object "
                 f"(rows = objects, columns = views, low to high; green = correct at thr {thr:.2f})")
    fig.tight_layout()
    fig.subplots_adjust(left=0.18)
    save_fig(fig, f"{safe_name(obj_type)}/objects_{task}.png")


def plot_per_object(samples, probs, obj_type, thr, task="movable"):
    """One column per object: dots = individual views, bar = object mean."""
    groups = _group_by_object(samples, probs, obj_type)
    if not groups:
        return
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(groups)), 4.5))
    for x, (oid, idx) in enumerate(groups):
        p = np.array([probs[i] for i in idx])
        y = np.array([int(samples[i][task]) for i in idx])
        ax.scatter(x + rng.uniform(-0.15, 0.15, len(p)), p, alpha=0.7, s=18,
                   c=np.where(y == 1, "tab:blue", "tab:orange"))
        ax.hlines(p.mean(), x - 0.3, x + 0.3, color="black", lw=2)
    ax.axhline(thr, color="red", linestyle="--", label=f"threshold {thr:.2f}")
    ax.set_xticks(range(len(groups)))
    if len(groups) <= 40:
        ax.set_xticklabels([str(oid)[:18] for oid, _ in groups], rotation=75, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([])
    ax.scatter([], [], c="tab:blue", label=f"true {task}=1")
    ax.scatter([], [], c="tab:orange", label=f"true {task}=0")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("individual objects (sorted by mean score)")
    ax.set_ylabel(f"predicted {task} score")
    ax.set_title(f"{obj_type}: score per object (dots = views, bar = mean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, f"{safe_name(obj_type)}/spread_{task}.png")


def plot_all_types(samples, movable_score, unsafe_score, min_views=1):
    """Per-object plots for every object type (movable + unsafe)."""
    counts = defaultdict(int)
    for s in samples:
        counts[s["object_type"]] += 1
    types = sorted(t for t, n in counts.items() if n >= min_views)
    print(f"\nPlotting {len(types)} object types -> {PLOT_DIR}/<type>/")
    for t in types:
        show_object_scores(samples, movable_score, t, 1 - THR, "movable")
        plot_per_object(samples, movable_score, t, 1 - THR, "movable")
        show_object_scores(samples, unsafe_score, t, THR, "unsafe")
        plot_per_object(samples, unsafe_score, t, THR, "unsafe")


def variance_table(samples, movable_score, min_views=1):
    """Within-object vs between-object variance of the movable score, per type."""
    df = pd.DataFrame({"oid": [s["object_id"] for s in samples],
                       "type": [s["object_type"] for s in samples],
                       "p": movable_score})
    rows = {}
    for t, g in df.groupby("type"):
        if len(g) < min_views:
            continue
        rows[t] = {"views": len(g),
                   "objects": g["oid"].nunique(),
                   "within_var": g.groupby("oid")["p"].var().mean(),
                   "between_var": g.groupby("oid")["p"].mean().var()}
    return pd.DataFrame(rows).T.sort_values("views", ascending=False).round(4)


def run_ablations():
    global VERBOSE_EPOCHS
    old, VERBOSE_EPOCHS = VERBOSE_EPOCHS, False
    rows = {}
    for name, cfg in ABLATIONS.items():
        print(f"\n##### ablation: {name}")
        per_seed = []
        for sd in ABL_SEEDS:
            folds = run_cv(ABL_COND, MASKED, seed=sd, **cfg)
            _, P = pooled(folds)
            nm = compute_metrics(P["y_nm"], P["s_nm"])
            u = compute_metrics(P["y_u"], P["s_u"])
            per_seed.append({
                "nm_auc": nm["auc"], "nm_f1": nm["f1"], "u_auc": u["auc"], "u_f1": u["f1"],
                "train_auc_nm": np.nanmean([f["train_auc"]["nm"] for f in folds]),
                "train_auc_u": np.nanmean([f["train_auc"]["u"] for f in folds])})
        rows[name] = {k: f"{np.mean([m[k] for m in per_seed]):.3f}±{np.std([m[k] for m in per_seed]):.3f}"
                      for k in per_seed[0]}
    VERBOSE_EPOCHS = old
    print("\n" + "=" * 70)
    print(f"ABLATIONS ({ABL_COND}, split by {GROUP_BY}, seeds {ABL_SEEDS})")
    print("=" * 70)
    print(pd.DataFrame(rows).T.to_string())


# leakage demo
if RUN_LEAKAGE_DEMO:
    for gb in ["object_id", "object_type"]:
        folds_ = list(seeded_group_cv_splits(resized_dataset, N_FOLDS, seed=42, group_by=gb))
        tv, te_ = folds_[0]
        print(f"type-only baseline, split by {gb:11s} unsafe AUC = "
              f"{type_baseline(tv, te_, 'unsafe')['auc']:.3f}   "
              f"non-movable AUC = {type_baseline(tv, te_, 'movable', flip=True)['auc']:.3f}")

print(f"\nSplit by '{GROUP_BY}'. Type-only baseline AUC (no image used): "
      + "  ".join(f"{k} {v:.3f}" for k, v in type_baseline_pooled(GROUP_BY).items()))

# main experiment
TASKS = [("non_movable", "nm"), ("unsafe", "u")]

if RUN_MAIN:
    results = {}
    for c in CONDS:
        for sd in SEEDS:
            print(f"\n Condition={c}  Seed={sd}")
            results[(c, sd)] = run_cv(c, MASKED, seed=sd)

    # summary per condition
    print("\n" + "=" * 70)
    print(f"SUMMARY  (split by {GROUP_BY}, pooled over {N_FOLDS} folds, "
          f"mean ± std over {len(SEEDS)} seed(s), threshold {THR})")
    print("=" * 70)
    for task, tag in TASKS:
        rows = {}
        for c in CONDS:
            per_seed = []
            for sd in SEEDS:
                _, P = pooled(results[(c, sd)])
                y, s = (P["y_nm"], P["s_nm"]) if tag == "nm" else (P["y_u"], P["s_u"])
                per_seed.append({"base_rate": y.mean(), **compute_metrics(y, s)})
            rows[c] = {k: f"{np.mean([m[k] for m in per_seed]):.3f}±{np.std([m[k] for m in per_seed]):.3f}"
                       for k in per_seed[0]}
            rows[c]["train_auc"] = f"{np.nanmean([f['train_auc'][tag] for sd in SEEDS for f in results[(c, sd)]]):.3f}"
        print(f"\n--- task: {task} ---")
        print(pd.DataFrame(rows).T[["base_rate", "accuracy", "precision", "recall", "f1",
                                    "auc", "ap", "train_auc"]])

    # by type, per condition
    print("\n" + "=" * 70)
    print(f"BY TYPE  (all folds and seeds pooled, types with n >= {MIN_TYPE_N})")
    print("recall = hit rate on true positives | fpr = false-alarm rate on true negatives")
    print("=" * 70)
    for c in CONDS:
        samples, P = pooled([f for sd in SEEDS for f in results[(c, sd)]])
        for task, tag in TASKS:
            y, s = (P["y_nm"], P["s_nm"]) if tag == "nm" else (P["y_u"], P["s_u"])
            print(f"\n[{c}] {task}")
            print(per_type_metrics(samples, y, s, n_rep=len(SEEDS)).round(3).head(TYPE_ROWS).to_string())

    # score plots (fused, first seed), for ALL object types
    samples, P = pooled(results[("fused", SEEDS[0])])
    movable_score = 1 - P["s_nm"]
    plot_all_types(samples, movable_score, P["s_u"], min_views=PLOT_MIN_VIEWS)
    print("\nmovable score variance by type (within-object vs between-object):")
    print(variance_table(samples, movable_score).to_string())

# ablations
if RUN_ABLATIONS:
    run_ablations()

print("\nDone.")