import pickle
import random
from collections import Counter
import numpy as np
from ai2thor.controller import Controller

# ---------------- config ----------------
MASS_THRESHOLD_KG = 25.0
PADDING = 10                    # pixels of context around each bounding box
MIN_BBOX_AREA = 400             # px, skip tiny crops
MIN_VISIBLE_PX = 300            # px of the object actually visible in the full frame
MAX_VIEWS_PER_OBJECT = 3
MAX_POSITIONS_PER_SCENE = 4
ROTATIONS = [0, 90, 180, 270]
EXCLUDE_TYPES = {"Floor", "Wall", "Ceiling", "Room"}
OUT_PATH = "ai2thor_dataset_masked.pkl"   

SCENE_RANGES = {                # small set for now; range(_1, _31) for full dataset
    "Kitchen": range(1, 11),
    "LivingRoom": range(201, 211),
    "Bedroom": range(301, 311),
    "Bathroom": range(401, 411),
}

random.seed(42)
drops = Counter()


def make_controller():
    return Controller(renderInstanceSegmentation=True, renderDepthImage=True)


def extract_object_samples(event):
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
        x1, y1 = max(0, x1 - PADDING), max(0, y1 - PADDING)
        x2, y2 = min(w, x2 + PADDING), min(h, y2 + PADDING)
        if x2 <= x1 or y2 <= y1:
            continue
        if (y2 - y1) * (x2 - x1) < MIN_BBOX_AREA:
            drops["bbox_too_small"] += 1
            continue

        samples.append({
            "object_id": object_id,
            "object_type": obj["objectType"],
            "rgb": rgb[y1:y2, x1:x2].copy(),
            "depth": depth[y1:y2, x1:x2].copy(),
            "mask": mask_full[y1:y2, x1:x2].astype(np.uint8),   # only the crop, not the full frame
            "visible_px": visible_px,
            "movable": mass <= MASS_THRESHOLD_KG,
            "unsafe": bool(obj.get("breakable", False)) and bool(obj.get("pickupable", False)),
            "mass": mass,
        })
    return samples


def collect_scene_samples(controller, scene):
    controller.reset(scene=scene)
    event = controller.step("Pass")
    assert hasattr(event, "instance_masks"), "This AI2-THOR version has no event.instance_masks"

    positions = controller.step("GetReachablePositions").metadata["actionReturn"]
    if not positions:
        return []
    chosen = random.sample(positions, k=min(MAX_POSITIONS_PER_SCENE, len(positions)))

    candidates = {}                         # object_id -> all views collected
    for pos in chosen:
        for rot in ROTATIONS:
            event = controller.step(action="Teleport", position=pos,
                                    rotation={"x": 0, "y": rot, "z": 0})
            if not event.metadata["lastActionSuccess"]:
                continue
            for s in extract_object_samples(event):
                s["scene"] = scene
                candidates.setdefault(s["object_id"], []).append(s)

    out = []
    for views in candidates.values():       # keep the MOST visible views, not the first ones
        views.sort(key=lambda s: s["visible_px"], reverse=True)
        out.extend(views[:MAX_VIEWS_PER_OBJECT])
    return out


# ---------------- run ----------------
controller = make_controller()
dataset = []

for room_type, ids in SCENE_RANGES.items():
    for i in ids:
        scene = f"FloorPlan{i}"
        try:
            scene_samples = collect_scene_samples(controller, scene)
            dataset.extend(scene_samples)
            print(f"{scene}: {len(scene_samples)} samples (total: {len(dataset)})")
        except Exception as e:
            print(f"  Skipped {scene}: {e}")
            try:
                controller.stop()
            except Exception:
                pass
            controller = make_controller()

controller.stop()

# ---------------- summary + save ----------------
n = len(dataset)
n_mov = sum(s["movable"] for s in dataset)
n_uns = sum(s["unsafe"] for s in dataset)
print(f"\nFinal dataset size: {n} samples")
print(f"Movable: {n_mov} ({100 * n_mov / n:.1f}%)   Unsafe: {n_uns} ({100 * n_uns / n:.1f}%)")
print(f"Dropped samples: {dict(drops)}")
print(f"Median visible px: {int(np.median([s['visible_px'] for s in dataset]))}")

non_mov_types = Counter(s["object_type"] for s in dataset if not s["movable"])
print(f"Non-movable samples by type: {dict(non_mov_types)}")

with open(OUT_PATH, "wb") as f:
    pickle.dump(dataset, f)
print(f"Saved to {OUT_PATH}")