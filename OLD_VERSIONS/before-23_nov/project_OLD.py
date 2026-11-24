import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ai2thor.controller import Controller

# config 
MASS_THRESHOLD_KG = 25.0
PADDING = 10                    # pixels of context around each bounding box
MIN_BBOX_AREA = 400             # px, skip tiny crops
MIN_VISIBLE_PX = 300            # px of the object actually visible in the full frame
MAX_VIEWS_PER_OBJECT = 3
MAX_POSITIONS_PER_SCENE = 4
ROTATIONS = [0, 90, 180, 270]
EXCLUDE_TYPES = {"Floor", "Wall", "Ceiling", "Room"}
WIDTH, HEIGHT = 800, 600        
OUT_PATH = "ai2thor_dataset_masked.pkl"

SCENE_RANGES = {    # range(_1, _31) for the full dataset
    "Kitchen": range(1, 11),
    "LivingRoom": range(201, 211),
    "Bedroom": range(301, 311),
    "Bathroom": range(401, 411),
    }

# confidence score settings
REF_VISIBLE_PX = 2000      # visible pixels at which the size score saturates to 1.0
NEAR_M, FAR_M = 1.0, 3.5   # distance score: 1.0 at/below NEAR_M, ~0.1 at/above FAR_M
THIN_REF_PX = 24           # crop side (px) at which the thinness score saturates
TRUNC_SCORE = 0.3          # score for objects touching the frame border
CONF_WEIGHTS = {"size": 0.40, "trunc": 0.20, "dist": 0.15, "thin": 0.10, "rel_vis": 0.15}

# outputs
SAVE_ENV_PLOTS = True           # one full-frame segmentation plot per scene
SAVE_PER_OBJECT_PLOTS = False   # one figure per sample (slow for big runs)
SAVE_PICKLE = True
OUT_DIR = Path("output")
ENV_DIR = OUT_DIR / "environments"
OBJ_DIR = OUT_DIR / "per_object"

random.seed(42)
drops = Counter()


def make_controller():
    return Controller(
        renderInstanceSegmentation=True,
        renderDepthImage=True,
        width=WIDTH,
        height=HEIGHT,
    )


# confidence score calculation: 
# weighted geometric mean of size, truncation, distance, thinness, and relative visibility
def compute_confidence(s):
    comps = {
        "size":    min(1.0, (s["visible_px"] / REF_VISIBLE_PX) ** 0.5),
        "trunc":   TRUNC_SCORE if s["truncated"] else 1.0,
        "dist":    1.0 if s["distance"] is None else
                   float(np.clip(1 - 0.9 * (s["distance"] - NEAR_M) / (FAR_M - NEAR_M), 0.1, 1.0)),
        "thin":    min(1.0, min(s["mask"].shape) / THIN_REF_PX),
        "rel_vis": s["rel_vis"],
    }
    # weighted geometric mean
    logc = sum(CONF_WEIGHTS[k] * np.log(max(v, 0.05)) for k, v in comps.items())
    return float(np.exp(logc)), comps


# sample extraction 
def extract_object_samples(event, scene_name, room_type):
    samples = []
    rgb, depth = event.frame, event.depth_frame
    h, w = rgb.shape[:2]
    meta_by_id = {o["objectId"]: o for o in event.metadata["objects"]}
    masks = event.instance_masks            # {objectId: bool array (H, W)}

    for object_id, bbox in event.instance_detections2D.items():
        obj = meta_by_id.get(object_id)
        if obj is None or obj["objectType"] in EXCLUDE_TYPES:
            continue

        mass = obj.get("mass")
        if mass is None or mass == 0.0:
            drops["no_mass"] += 1
            continue

        mask_full = masks.get(object_id)
        if mask_full is None:
            drops["no_mask"] += 1
            continue
        visible_px = int(mask_full.sum())
        if visible_px < MIN_VISIBLE_PX:
            drops["too_few_visible_px"] += 1
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        truncated = x1 <= 0 or y1 <= 0 or x2 >= w - 1 or y2 >= h - 1
        x1, y1 = max(0, x1 - PADDING), max(0, y1 - PADDING)
        x2, y2 = min(w, x2 + PADDING), min(h, y2 + PADDING)
        if x2 <= x1 or y2 <= y1:
            drops["invalid_bbox"] += 1
            continue
        if (y2 - y1) * (x2 - x1) < MIN_BBOX_AREA:
            drops["bbox_too_small"] += 1
            continue

        samples.append({
            "object_id": object_id,
            "object_type": obj["objectType"],
            "scene": scene_name,
            "room_type": room_type,
            "rgb": rgb[y1:y2, x1:x2].copy(),
            "depth": depth[y1:y2, x1:x2].copy(),
            "mask": mask_full[y1:y2, x1:x2].astype(np.uint8),   # crop only, not full frame
            "visible_px": visible_px,
            "movable": mass <= MASS_THRESHOLD_KG,
            "unsafe": bool(obj.get("breakable", False)) and bool(obj.get("pickupable", False)),
            "mass": mass,
            "truncated": truncated,
            "distance": obj.get("distance"),
            "rel_vis": 1.0,          # filled in once all views of the object are collected
            "confidence": 0.0,       # filled in once all views of the object are collected
        })
    return samples


#  plotting
def save_full_environment_plot(event, scene_name):
    """Full-frame view with masks + bounding boxes for every valid object."""
    rgb = event.frame.copy()
    h, w = rgb.shape[:2]
    masks = event.instance_masks
    detections = event.instance_detections2D
    meta_by_id = {o["objectId"]: o for o in event.metadata["objects"]}

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(rgb)

    cmap = plt.get_cmap("tab20")
    color_idx = 0
    drawn = 0

    for object_id, bbox in detections.items():
        obj = meta_by_id.get(object_id)
        if obj is None or obj["objectType"] in EXCLUDE_TYPES:
            continue

        mask = masks.get(object_id)
        if mask is None or mask.sum() < MIN_VISIBLE_PX:
            continue

        mass = obj.get("mass") or 0.0
        if mass == 0.0:
            continue

        color = np.array(cmap(color_idx % 20)[:3])
        color_idx += 1

        colored_mask = np.zeros((h, w, 4), dtype=np.float32)
        colored_mask[mask] = [color[0], color[1], color[2], 0.45]
        ax.imshow(colored_mask)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=2, edgecolor=color, facecolor="none"))
        ax.text(x1, max(y1 - 8, 12), f"{obj['objectType']} ({mass:.1f}kg)",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=2))
        drawn += 1

    ax.set_title(f"Scene: {scene_name} - Full Environment Segmentation ({drawn} objects)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    out_path = ENV_DIR / f"{scene_name}_environment_segmentation.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved environment plot -> {out_path}")


def save_sample_plot(s, idx):
    """One figure per sample: RGB, depth, object-only depth, mask overlay, isolated object."""
    m = s["mask"].astype(bool)

    overlay = s["rgb"].astype(np.float32)
    overlay[m] = overlay[m] * 0.6 + np.array([0, 255, 120]) * 0.4
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    isolated = s["rgb"] * m[..., None]
    depth_obj = np.where(m, s["depth"], np.nan)

    fig, ax = plt.subplots(1, 5, figsize=(17, 3.5))
    ax[0].imshow(s["rgb"])
    ax[0].set_title(f"{s['object_type']} ({s['scene']})")
    ax[1].imshow(s["depth"], cmap="viridis")
    ax[1].set_title("Depth (crop)")
    ax[2].imshow(depth_obj, cmap="viridis")
    ax[2].set_title("Depth (object only)")
    ax[3].imshow(overlay)
    ax[3].set_title("Mask overlay")
    ax[4].imshow(isolated)
    ax[4].set_title(f"movable={s['movable']} unsafe={s['unsafe']}\n"
                    f"{s['mass']:.1f}kg, {s['visible_px']}px, conf={s['confidence']:.2f}")
    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(OBJ_DIR / f"{idx:05d}_{s['scene']}_{s['object_type']}.png", dpi=110)
    plt.close()


# per-scene collection
def collect_scene_samples(controller, scene, room_type):
    controller.reset(scene=scene)
    event = controller.step("Pass")
    assert hasattr(event, "instance_masks"), "This AI2-THOR version has no event.instance_masks"

    positions = controller.step("GetReachablePositions").metadata["actionReturn"]
    if not positions:
        print(f"  No reachable positions found for {scene}.")
        return []
    chosen = random.sample(positions, k=min(MAX_POSITIONS_PER_SCENE, len(positions)))

    candidates = {}                         # object_id -> all views collected
    env_plotted = False

    for pos in chosen:
        for rot in ROTATIONS:
            event = controller.step(action="Teleport", position=pos,
                                    rotation={"x": 0, "y": rot, "z": 0})
            if not event.metadata["lastActionSuccess"]:
                continue

            if SAVE_ENV_PLOTS and not env_plotted and len(event.instance_detections2D) > 0:
                save_full_environment_plot(event, scene)
                env_plotted = True

            for s in extract_object_samples(event, scene, room_type):
                candidates.setdefault(s["object_id"], []).append(s)

    out = []
    for views in candidates.values():
        max_px = max(v["visible_px"] for v in views)
        for v in views:
            v["rel_vis"] = v["visible_px"] / max_px      # occlusion proxy vs the best view
            v["confidence"], v["conf_parts"] = compute_confidence(v)
        views.sort(key=lambda s: s["confidence"], reverse=True)   # keep the BEST views, not the first
        out.extend(views[:MAX_VIEWS_PER_OBJECT])
    return out


# run
OUT_DIR.mkdir(exist_ok=True)
ENV_DIR.mkdir(exist_ok=True)
OBJ_DIR.mkdir(exist_ok=True)

controller = make_controller()
dataset = []

for room_type, ids in SCENE_RANGES.items():
    for i in ids:
        scene = f"FloorPlan{i}"
        print(f"\nProcessing scene: {scene}")
        try:
            scene_samples = collect_scene_samples(controller, scene, room_type)
            dataset.extend(scene_samples)
            print(f"  {scene}: {len(scene_samples)} samples (total: {len(dataset)})")
        except Exception as e:
            print(f"  Skipped {scene}: {e}")
            try:
                controller.stop()
            except Exception:
                pass
            controller = make_controller()

controller.stop()

# save 
if SAVE_PER_OBJECT_PLOTS:
    print(f"\nSaving {len(dataset)} per-object plots -> {OBJ_DIR}")
    for idx, s in enumerate(dataset):
        save_sample_plot(s, idx)

if SAVE_PICKLE:
    with open(OUT_PATH, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset -> {OUT_PATH}")

# summary 
if not dataset:
    print("\nDataset is empty, nothing to summarize.")
else:
    n = len(dataset)
    n_mov = sum(s["movable"] for s in dataset)
    n_uns = sum(s["unsafe"] for s in dataset)
    n_unique = len({(s["scene"], s["object_id"]) for s in dataset})

    print(f"\nFinal dataset size: {n} samples ({n_unique} unique objects)")
    print(f"Movable: {n_mov} ({100 * n_mov / n:.1f}%)   Unsafe: {n_uns} ({100 * n_uns / n:.1f}%)")
    print(f"Dropped samples: {dict(drops)}")
    print(f"Median visible px: {int(np.median([s['visible_px'] for s in dataset]))}")

    non_mov_types = Counter(s["object_type"] for s in dataset if not s["movable"])
    print(f"Non-movable samples by type: {dict(non_mov_types)}")

    df = pd.DataFrame([{k: s[k] for k in
                        ("scene", "room_type", "object_type", "movable", "unsafe",
                         "mass", "visible_px", "confidence", "truncated", "distance")}
                       for s in dataset])

    print("\nTruth table (movable x unsafe):")
    print(pd.crosstab(df["movable"], df["unsafe"], margins=True))

    print("\nPer room type:")
    print(df.groupby("room_type")[["movable", "unsafe"]].agg(["sum", "mean"]).round(2))

    print("\nPer object type (top 20):")
    per_type = (df.groupby("object_type")
                  .agg(n=("movable", "size"), movable=("movable", "mean"),
                       unsafe=("unsafe", "mean"), mass_med=("mass", "median"))
                  .sort_values("n", ascending=False).head(20).round(2))
    print(per_type)

    print("\nMass / visible px stats:")
    print(df[["mass", "visible_px"]].describe().round(1))

    df["conf_bucket"] = pd.cut(df["confidence"], [0, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    print("\nConfidence distribution:")
    print(df["confidence"].describe().round(2))
    print(df.groupby("conf_bucket", observed=True)[["movable", "unsafe"]].agg(["size", "mean"]).round(2))

    print("\nLowest-confidence object types:")
    print(df.groupby("object_type")["confidence"].mean().sort_values().head(10).round(2))

    print("\nConfidence by truncated:")
    print(df.groupby("truncated")["confidence"].agg(["size", "mean"]).round(2))

print("\nAll scenes processed successfully!")