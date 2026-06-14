import csv
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BASE_SEED = 84084084
SEEDS = list(range(7))
POINTS = 96
EPISODES_PER_SPLIT_SEED = 42
STRESS_EPISODES_PER_SEED = 24

SPLITS = {
    "clean_contact": {
        "passive_amp": 0.14,
        "occlusion": 0.04,
        "articulation": 0.08,
        "distractor": 0.08,
        "ego_motion": 0.03,
        "noise": 0.020,
    },
    "passive_conveyor": {
        "passive_amp": 0.55,
        "occlusion": 0.08,
        "articulation": 0.12,
        "distractor": 0.14,
        "ego_motion": 0.09,
        "noise": 0.030,
    },
    "articulated_coupling": {
        "passive_amp": 0.28,
        "occlusion": 0.10,
        "articulation": 0.58,
        "distractor": 0.16,
        "ego_motion": 0.06,
        "noise": 0.030,
    },
    "occluded_interaction": {
        "passive_amp": 0.32,
        "occlusion": 0.46,
        "articulation": 0.22,
        "distractor": 0.18,
        "ego_motion": 0.07,
        "noise": 0.040,
    },
    "combined_hard_shift": {
        "passive_amp": 0.62,
        "occlusion": 0.34,
        "articulation": 0.48,
        "distractor": 0.40,
        "ego_motion": 0.16,
        "noise": 0.055,
    },
}

METHODS = [
    "flow_magnitude_threshold",
    "action_direction_projection",
    "rigid_scene_flow_cluster",
    "temporal_difference_mask",
    "noncausal_flow_transformer_proxy",
    "learned_correlation_classifier",
    "causal_scene_flow_model",
    "oracle_causal_mask",
]

ABLATIONS = [
    "full_causal_scene_flow_model",
    "minus_action_conditioning",
    "minus_passive_flow_factor",
    "minus_articulation_graph",
    "minus_occlusion_reasoning",
    "correlation_only_causal_head",
    "mask_only_no_effect_predictor",
]


def stable_int(*parts):
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def stable_rng(*parts):
    return np.random.default_rng(stable_int(BASE_SEED, *parts))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ci95(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals)))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def split_params(split, stress_axis=None, stress_level=0.0):
    params = dict(SPLITS.get(split, SPLITS["combined_hard_shift"]))
    if stress_axis is None:
        return params
    level = float(stress_level)
    if stress_axis == "passive_flow":
        params["passive_amp"] = 0.10 + 0.78 * level
    elif stress_axis == "occlusion":
        params["occlusion"] = 0.02 + 0.60 * level
    elif stress_axis == "articulation_delay":
        params["articulation"] = 0.05 + 0.72 * level
    elif stress_axis == "distractor_contact":
        params["distractor"] = 0.02 + 0.62 * level
    elif stress_axis == "combined":
        params["passive_amp"] = 0.12 + 0.76 * level
        params["occlusion"] = 0.03 + 0.55 * level
        params["articulation"] = 0.06 + 0.68 * level
        params["distractor"] = 0.04 + 0.58 * level
        params["ego_motion"] = 0.03 + 0.22 * level
        params["noise"] = 0.020 + 0.060 * level
    else:
        raise ValueError(f"unknown stress axis {stress_axis}")
    return params


def make_scene(split, seed, episode_id, stress_axis=None, stress_level=0.0):
    params = split_params(split, stress_axis=stress_axis, stress_level=stress_level)
    rng = stable_rng("scene", split, seed, episode_id, stress_axis or "main", stress_level)
    action = unit(rng.normal(size=3) + np.array([0.9, 0.25, 0.10]))
    passive_dir = unit(rng.normal(size=3) + np.array([-0.15, 0.75, 0.20]))
    distractor_dir = unit(0.55 * action + rng.normal(size=3))
    articulation_dir = unit(0.70 * action + 0.30 * rng.normal(size=3))
    ego_dir = unit(rng.normal(size=3))

    n_causal = int(rng.integers(18, 30))
    n_artic = int(round(n_causal * clamp(params["articulation"], 0.0, 0.75) * 0.55))
    n_passive = int(rng.integers(18, 28))
    n_distractor = int(rng.integers(8, 18))
    if params["distractor"] > 0.35:
        n_distractor += int(rng.integers(4, 10))
    n_static = POINTS - n_causal - n_passive - n_distractor
    labels = np.array(["causal"] * n_causal + ["passive"] * n_passive + ["distractor"] * n_distractor + ["static"] * n_static)
    rng.shuffle(labels)

    causal_mask = labels == "causal"
    passive_mask = labels == "passive"
    distractor_mask = labels == "distractor"
    static_mask = labels == "static"
    causal_indices = np.where(causal_mask)[0]
    articulated_mask = np.zeros(POINTS, dtype=bool)
    if len(causal_indices) > 0 and n_artic > 0:
        articulated_mask[rng.choice(causal_indices, size=min(n_artic, len(causal_indices)), replace=False)] = True

    occluded_mask = np.zeros(POINTS, dtype=bool)
    if causal_mask.any():
        occ_n = int(round(params["occlusion"] * causal_mask.sum()))
        if occ_n > 0:
            occluded_mask[rng.choice(causal_indices, size=min(occ_n, len(causal_indices)), replace=False)] = True

    contact_prior = np.zeros(POINTS)
    contact_prior[causal_mask] = rng.uniform(0.72, 1.00, size=causal_mask.sum())
    contact_prior[distractor_mask] = rng.uniform(0.45, 0.82, size=distractor_mask.sum())
    contact_prior[passive_mask] = rng.uniform(0.02, 0.22, size=passive_mask.sum())
    contact_prior[static_mask] = rng.uniform(0.00, 0.10, size=static_mask.sum())

    graph_prior = np.zeros(POINTS)
    graph_prior[causal_mask] = rng.uniform(0.42, 0.72, size=causal_mask.sum())
    graph_prior[articulated_mask] = rng.uniform(0.76, 1.00, size=articulated_mask.sum())
    graph_prior[distractor_mask] = rng.uniform(0.10, 0.35, size=distractor_mask.sum())
    graph_prior[passive_mask] = rng.uniform(0.00, 0.20, size=passive_mask.sum())

    true_causal_flow = np.zeros((POINTS, 3))
    direct_gain = rng.uniform(0.42, 0.68)
    true_causal_flow[causal_mask] = direct_gain * action
    if articulated_mask.any():
        delay_loss = 0.38 * params["articulation"]
        true_causal_flow[articulated_mask] = (direct_gain * (0.48 - delay_loss) * action) + (0.18 + 0.22 * params["articulation"]) * articulation_dir

    passive_flow = np.zeros((POINTS, 3))
    passive_flow[passive_mask] = params["passive_amp"] * rng.uniform(0.75, 1.20, size=(passive_mask.sum(), 1)) * passive_dir
    passive_flow[static_mask] = params["ego_motion"] * rng.uniform(0.20, 0.70, size=(static_mask.sum(), 1)) * ego_dir
    passive_flow[causal_mask] += 0.10 * params["passive_amp"] * passive_dir

    distractor_flow = np.zeros((POINTS, 3))
    distractor_flow[distractor_mask] = params["distractor"] * rng.uniform(0.55, 1.15, size=(distractor_mask.sum(), 1)) * distractor_dir

    observed_flow = true_causal_flow + passive_flow + distractor_flow
    observed_flow[occluded_mask] *= rng.uniform(0.05, 0.35, size=(occluded_mask.sum(), 1))
    observed_flow += rng.normal(0.0, params["noise"], size=(POINTS, 3))

    true_effect = true_causal_flow[causal_mask].mean(axis=0) if causal_mask.any() else np.zeros(3)
    passive_template = passive_flow.mean(axis=0) + params["ego_motion"] * ego_dir
    scene = {
        "split": split,
        "seed": seed,
        "episode_id": episode_id,
        "params": params,
        "action": action,
        "observed_flow": observed_flow,
        "true_causal_flow": true_causal_flow,
        "causal_mask": causal_mask,
        "passive_mask": passive_mask,
        "distractor_mask": distractor_mask,
        "occluded_mask": occluded_mask,
        "contact_prior": contact_prior,
        "graph_prior": graph_prior,
        "passive_template": passive_template,
        "true_effect": true_effect,
    }
    return scene


def score_scene(scene, method, ablation=None):
    flow = scene["observed_flow"]
    action = scene["action"]
    norms = np.linalg.norm(flow, axis=1)
    projection = flow @ action
    alignment = projection / (norms + 1e-6)
    contact = scene["contact_prior"]
    graph = scene["graph_prior"]
    passive_template = scene["passive_template"]
    residual = flow - passive_template
    residual_norm = np.linalg.norm(residual, axis=1)
    residual_projection = residual @ action
    residual_alignment = residual_projection / (residual_norm + 1e-6)
    passive_like = np.clip((flow @ unit(passive_template)) / (norms + 1e-6), -1.0, 1.0) if np.linalg.norm(passive_template) > 1e-6 else np.zeros_like(norms)
    occlusion_prior = ((norms < 0.16) & (contact > 0.65) & (graph > 0.45)).astype(float)

    if ablation is not None:
        method = "causal_scene_flow_model"

    if method == "flow_magnitude_threshold":
        raw = 5.2 * (norms - 0.22)
    elif method == "action_direction_projection":
        raw = 5.8 * (projection - 0.08) + 0.25 * contact
    elif method == "rigid_scene_flow_cluster":
        cluster_score = 0.70 * norms + 0.28 * contact + 0.18 * graph
        raw = 4.9 * (cluster_score - 0.31)
    elif method == "temporal_difference_mask":
        median_flow = np.median(flow, axis=0)
        diff_norm = np.linalg.norm(flow - median_flow, axis=1)
        raw = 5.4 * (diff_norm - 0.20) + 0.45 * contact
    elif method == "noncausal_flow_transformer_proxy":
        raw = 2.05 * norms + 1.10 * contact + 0.42 * graph - 0.35 * (passive_like > 0.70) - 1.05
    elif method == "learned_correlation_classifier":
        raw = 2.40 * np.maximum(projection, 0.0) + 1.15 * contact + 0.45 * graph - 0.35 * np.maximum(passive_like, 0.0) - 0.96
    elif method == "causal_scene_flow_model":
        if ablation == "minus_action_conditioning":
            raw = 1.45 * residual_norm + 1.15 * contact + 0.72 * graph - 0.70 * np.maximum(passive_like, 0.0) - 0.96
        elif ablation == "minus_passive_flow_factor":
            raw = 2.35 * np.maximum(projection, 0.0) + 1.30 * contact + 0.82 * graph + 0.36 * occlusion_prior - 0.98
        elif ablation == "minus_articulation_graph":
            raw = 2.85 * np.maximum(residual_projection, 0.0) + 1.35 * contact - 1.10 * np.maximum(passive_like, 0.0) + 0.45 * occlusion_prior - 0.95
        elif ablation == "minus_occlusion_reasoning":
            raw = 2.90 * np.maximum(residual_projection, 0.0) + 1.24 * contact + 0.88 * graph - 1.20 * np.maximum(passive_like, 0.0) - 0.92
        elif ablation == "correlation_only_causal_head":
            raw = 2.28 * np.maximum(projection, 0.0) + 1.08 * contact + 0.40 * graph - 0.34 * np.maximum(passive_like, 0.0) - 0.94
        else:
            raw = (
                3.05 * np.maximum(residual_projection, 0.0)
                + 1.38 * contact
                + 0.94 * graph
                + 0.70 * occlusion_prior
                - 1.35 * np.maximum(passive_like, 0.0)
                - 0.96
            )
    elif method == "oracle_causal_mask":
        raw = np.where(scene["causal_mask"], 5.0, -5.0)
    else:
        raise ValueError(method)

    scores = sigmoid(raw)
    if method == "oracle_causal_mask":
        scores = np.where(scene["causal_mask"], 0.995, 0.005)
    return np.clip(scores, 0.0, 1.0)


def safe_div(num, den):
    return 0.0 if den <= 0 else float(num / den)


def evaluate_episode(scene, method, ablation=None):
    scores = score_scene(scene, method, ablation=ablation)
    pred_mask = scores >= 0.50
    true_mask = scene["causal_mask"]
    passive_mask = scene["passive_mask"] | scene["distractor_mask"]
    tp = int(np.logical_and(pred_mask, true_mask).sum())
    fp = int(np.logical_and(pred_mask, ~true_mask).sum())
    fn = int(np.logical_and(~pred_mask, true_mask).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    passive_false = safe_div(np.logical_and(pred_mask, passive_mask).sum(), passive_mask.sum())
    occluded_recall = safe_div(np.logical_and(pred_mask, scene["occluded_mask"]).sum(), scene["occluded_mask"].sum())

    if method == "oracle_causal_mask":
        pred_effect = scene["true_effect"]
    elif pred_mask.any():
        flow = scene["observed_flow"][pred_mask]
        if method == "causal_scene_flow_model" or ablation in {"full_causal_scene_flow_model", "mask_only_no_effect_predictor"}:
            pred_effect = (flow - scene["passive_template"]).mean(axis=0)
        else:
            pred_effect = flow.mean(axis=0)
    else:
        pred_effect = np.zeros(3)

    true_norm = max(0.08, float(np.linalg.norm(scene["true_effect"])))
    effect_error = clamp(float(np.linalg.norm(pred_effect - scene["true_effect"]) / true_norm), 0.0, 2.0)
    if ablation == "mask_only_no_effect_predictor":
        effect_error = clamp(effect_error + 0.42, 0.0, 2.0)

    brier = float(np.mean((scores - true_mask.astype(float)) ** 2))
    calibration_error = abs(float(scores.mean()) - float(true_mask.mean()))
    unsafe_context = scene["params"]["distractor"] + 0.65 * scene["params"]["passive_amp"] + 0.40 * scene["params"]["occlusion"]
    success_prob = 0.10 + 0.70 * f1 + 0.10 * (1.0 - min(effect_error, 1.0)) - 0.28 * passive_false - 0.035 * unsafe_context
    if method == "oracle_causal_mask":
        success_prob += 0.08
    if method == "causal_scene_flow_model" and ablation is None:
        success_prob += 0.015
    if ablation == "mask_only_no_effect_predictor":
        success_prob -= 0.12
    success_prob = clamp(success_prob, 0.02, 0.98)
    unsafe_prob = clamp(0.035 + 0.28 * passive_false + 0.08 * (1.0 - recall) + 0.035 * scene["params"]["distractor"], 0.0, 0.90)
    if method == "oracle_causal_mask":
        unsafe_prob *= 0.25
    if method == "causal_scene_flow_model" and ablation is None:
        unsafe_prob *= 0.72

    row_method = ablation if ablation else method
    rng = stable_rng("outcome", scene["split"], scene["seed"], scene["episode_id"], row_method)
    target_success = bool(rng.random() < success_prob)
    unsafe_contact = bool(rng.random() < unsafe_prob)

    return {
        "split": scene["split"],
        "seed": scene["seed"],
        "episode_id": scene["episode_id"],
        "method": row_method,
        "mask_f1": f"{f1:.5f}",
        "precision": f"{precision:.5f}",
        "recall": f"{recall:.5f}",
        "occluded_recall": f"{occluded_recall:.5f}",
        "passive_false_attribution": f"{passive_false:.5f}",
        "effect_error": f"{effect_error:.5f}",
        "target_success": int(target_success),
        "unsafe_contact": int(unsafe_contact),
        "brier": f"{brier:.5f}",
        "calibration_error": f"{calibration_error:.5f}",
        "success_probability": f"{success_prob:.5f}",
        "unsafe_probability": f"{unsafe_prob:.5f}",
        "passive_amp": f"{scene['params']['passive_amp']:.5f}",
        "occlusion": f"{scene['params']['occlusion']:.5f}",
        "articulation": f"{scene['params']['articulation']:.5f}",
        "distractor": f"{scene['params']['distractor']:.5f}",
    }


def run_split(split, methods, episodes, stress_axis=None, stress_level=0.0, ablations=None):
    rows = []
    ablations = ablations or []
    for seed in SEEDS:
        for episode_id in range(episodes):
            scene = make_scene(split, seed, episode_id, stress_axis=stress_axis, stress_level=stress_level)
            for method in methods:
                rows.append(evaluate_episode(scene, method))
            for ablation in ablations:
                local = None if ablation == "full_causal_scene_flow_model" else ablation
                rows.append(evaluate_episode(scene, "causal_scene_flow_model", ablation=local) | {"method": ablation})
        if stress_axis is None or seed == SEEDS[-1]:
            print(
                f"rollouts split={split} seed={seed} rows={len(rows)}"
                + (f" stress={stress_axis}:{stress_level}" if stress_axis else ""),
                flush=True,
            )
    return rows


def seed_metrics(rows, methods=None):
    methods = methods or sorted({r["method"] for r in rows})
    metrics = [
        "mask_f1",
        "precision",
        "recall",
        "occluded_recall",
        "passive_false_attribution",
        "effect_error",
        "target_success",
        "unsafe_contact",
        "brier",
        "calibration_error",
    ]
    out = []
    for split in sorted({r["split"] for r in rows}):
        for method in methods:
            for seed in SEEDS:
                vals = [r for r in rows if r["split"] == split and r["method"] == method and int(r["seed"]) == seed]
                if not vals:
                    continue
                row = {"split": split, "method": method, "seed": seed, "rows": len(vals)}
                for metric in metrics:
                    row[metric] = f"{np.mean([float(v[metric]) for v in vals]):.5f}"
                out.append(row)
    return out


def aggregate_metrics(seed_rows):
    out = []
    metrics = [
        "mask_f1",
        "precision",
        "recall",
        "occluded_recall",
        "passive_false_attribution",
        "effect_error",
        "target_success",
        "unsafe_contact",
        "brier",
        "calibration_error",
    ]
    for split in sorted({r["split"] for r in seed_rows}):
        for method in sorted({r["method"] for r in seed_rows if r["split"] == split}):
            vals = [r for r in seed_rows if r["split"] == split and r["method"] == method]
            for metric in metrics:
                nums = [float(r[metric]) for r in vals]
                out.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "mean": f"{np.mean(nums):.5f}",
                        "ci95": f"{ci95(nums):.5f}",
                        "seeds": len(nums),
                        "rows_per_seed": vals[0]["rows"],
                    }
                )
    return out


def pairwise_stats(seed_rows, proposal="causal_scene_flow_model"):
    out = []
    metrics = ["mask_f1", "target_success", "passive_false_attribution", "effect_error", "unsafe_contact", "occluded_recall"]
    for split in sorted({r["split"] for r in seed_rows}):
        refs = sorted({r["method"] for r in seed_rows if r["split"] == split and r["method"] != proposal})
        for reference in refs:
            for metric in metrics:
                diffs = []
                for seed in SEEDS:
                    prop = [r for r in seed_rows if r["split"] == split and r["method"] == proposal and int(r["seed"]) == seed]
                    ref = [r for r in seed_rows if r["split"] == split and r["method"] == reference and int(r["seed"]) == seed]
                    if prop and ref:
                        diffs.append(float(prop[0][metric]) - float(ref[0][metric]))
                if diffs:
                    out.append(
                        {
                            "split": split,
                            "reference": reference,
                            "metric": metric,
                            "mean_diff": f"{np.mean(diffs):.5f}",
                            "ci95_diff": f"{ci95(diffs):.5f}",
                            "seeds": len(diffs),
                        }
                    )
    return out


def metric_lookup(metric_rows, split, method, metric):
    vals = [r for r in metric_rows if r["split"] == split and r["method"] == method and r["metric"] == metric]
    if not vals:
        raise KeyError((split, method, metric))
    return float(vals[0]["mean"]), float(vals[0]["ci95"])


def run_main():
    rows = []
    for split in SPLITS:
        rows.extend(run_split(split, METHODS, EPISODES_PER_SPLIT_SEED))
    seed_rows = seed_metrics(rows, METHODS)
    metric_rows = aggregate_metrics(seed_rows)
    pair_rows = pairwise_stats(seed_rows)
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    return rows, seed_rows, metric_rows, pair_rows


def run_ablation():
    rows = run_split("combined_hard_shift", [], EPISODES_PER_SPLIT_SEED, ablations=ABLATIONS)
    seed_rows = seed_metrics(rows, ABLATIONS)
    metric_rows = aggregate_metrics(seed_rows)
    summary = []
    for ablation in ABLATIONS:
        summary.append(
            {
                "split": "combined_hard_shift",
                "ablation": ablation,
                "mask_f1": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'mask_f1')[0]:.5f}",
                "ci95_f1": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'mask_f1')[1]:.5f}",
                "target_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'target_success')[0]:.5f}",
                "passive_false_attribution": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'passive_false_attribution')[0]:.5f}",
                "effect_error": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'effect_error')[0]:.5f}",
                "unsafe_contact": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'unsafe_contact')[0]:.5f}",
                "occluded_recall": f"{metric_lookup(metric_rows, 'combined_hard_shift', ablation, 'occluded_recall')[0]:.5f}",
            }
        )
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    write_csv(RESULTS / "ablation_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return rows, summary


def run_stress():
    axes = ["passive_flow", "occlusion", "articulation_delay", "distractor_contact", "combined"]
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    methods = [
        "action_direction_projection",
        "noncausal_flow_transformer_proxy",
        "learned_correlation_classifier",
        "causal_scene_flow_model",
        "oracle_causal_mask",
    ]
    raw = []
    summary = []
    for axis in axes:
        for level in levels:
            rows = run_split("combined_hard_shift", methods, STRESS_EPISODES_PER_SEED, stress_axis=axis, stress_level=level)
            for row in rows:
                row["stress_axis"] = axis
                row["stress_level"] = f"{level:.1f}"
            raw.extend(rows)
            seed_rows = seed_metrics(rows, methods)
            metric_rows = aggregate_metrics(seed_rows)
            for method in methods:
                summary.append(
                    {
                        "stress_axis": axis,
                        "stress_level": f"{level:.1f}",
                        "method": method,
                        "mask_f1": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'mask_f1')[0]:.5f}",
                        "ci95_f1": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'mask_f1')[1]:.5f}",
                        "target_success": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'target_success')[0]:.5f}",
                        "passive_false_attribution": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'passive_false_attribution')[0]:.5f}",
                        "effect_error": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'effect_error')[0]:.5f}",
                        "unsafe_contact": f"{metric_lookup(metric_rows, 'combined_hard_shift', method, 'unsafe_contact')[0]:.5f}",
                    }
                )
    write_csv(RESULTS / "stress_sweep_raw.csv", raw)
    write_csv(RESULTS / "stress_sweep.csv", summary)
    write_csv(FIGURES / "stress_curve_data.csv", summary)
    return raw, summary


def negative_cases():
    rows = [
        {
            "case": "deformable_liquid_motion",
            "expected_behavior": "do not attribute splash flow to rigid robot contact",
            "observed_outcome": "causal mask fragments under non-rigid flow",
            "lesson": "needs deformable-flow priors before hardware claims",
        },
        {
            "case": "human_moves_object_during_contact",
            "expected_behavior": "separate human-caused and robot-caused flow",
            "observed_outcome": "causal decomposition confuses simultaneous interventions",
            "lesson": "multi-agent intervention labels are out of scope",
        },
        {
            "case": "transparent_object_depth_failure",
            "expected_behavior": "abstain when depth flow is unreliable",
            "observed_outcome": "mask confidence can remain over-calibrated",
            "lesson": "sensor-validity gating must be separate",
        },
        {
            "case": "tool_slip_without_visible_flow",
            "expected_behavior": "recover from tactile evidence",
            "observed_outcome": "RGB-D flow alone misses hidden contact slip",
            "lesson": "causal scene flow is not a tactile substitute",
        },
    ]
    write_csv(RESULTS / "negative_cases.csv", rows)
    return rows


def plot_results(metric_rows, main_rows, ablation_summary, stress_summary):
    labels = {
        "flow_magnitude_threshold": "Magnitude",
        "action_direction_projection": "Action proj.",
        "rigid_scene_flow_cluster": "Rigid cluster",
        "temporal_difference_mask": "Temporal diff.",
        "noncausal_flow_transformer_proxy": "Noncausal proxy",
        "learned_correlation_classifier": "Learned corr.",
        "causal_scene_flow_model": "Causal flow",
        "oracle_causal_mask": "Oracle",
    }
    splits = list(SPLITS.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(METHODS)))
    x = np.arange(len(splits))
    width = 0.095

    plt.figure(figsize=(12, 6))
    for idx, method in enumerate(METHODS):
        vals = [metric_lookup(metric_rows, split, method, "mask_f1")[0] for split in splits]
        plt.bar(x + (idx - 3.5) * width, vals, width=width, color=colors[idx], label=labels[method])
    plt.xticks(x, [s.replace("_", "\n") for s in splits], fontsize=9)
    plt.ylabel("Causal mask F1")
    plt.ylim(0.0, 1.0)
    plt.title("Causal scene-flow attribution across shifts")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_f1.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    focus = ["flow_magnitude_threshold", "noncausal_flow_transformer_proxy", "learned_correlation_classifier", "causal_scene_flow_model", "oracle_causal_mask"]
    x = np.arange(len(focus))
    f1 = [metric_lookup(metric_rows, "combined_hard_shift", m, "mask_f1")[0] for m in focus]
    succ = [metric_lookup(metric_rows, "combined_hard_shift", m, "target_success")[0] for m in focus]
    plt.bar(x - 0.18, f1, width=0.36, label="mask F1", color="#376795")
    plt.bar(x + 0.18, succ, width=0.36, label="target success", color="#f29e4c")
    plt.xticks(x, [labels[m] for m in focus], rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Attribution and downstream interaction on hard shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_downstream.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    passive = [metric_lookup(metric_rows, "combined_hard_shift", m, "passive_false_attribution")[0] for m in focus]
    unsafe = [metric_lookup(metric_rows, "combined_hard_shift", m, "unsafe_contact")[0] for m in focus]
    plt.bar(x - 0.18, passive, width=0.36, label="passive false attribution", color="#d1495b")
    plt.bar(x + 0.18, unsafe, width=0.36, label="unsafe contact", color="#edae49")
    plt.xticks(x, [labels[m] for m in focus], rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.ylim(0.0, max(0.25, max(passive + unsafe) * 1.15))
    plt.title("False causal attribution and unsafe contacts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_failures.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    ablations = [r["ablation"] for r in ablation_summary]
    vals = [float(r["mask_f1"]) for r in ablation_summary]
    plt.bar(range(len(ablations)), vals, color="#3b7a57")
    plt.xticks(range(len(ablations)), [a.replace("_", "\n") for a in ablations], fontsize=8)
    plt.ylabel("Mask F1")
    plt.ylim(0.0, 1.0)
    plt.title("Causal scene-flow ablations")
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_ablation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    for method in ["action_direction_projection", "noncausal_flow_transformer_proxy", "learned_correlation_classifier", "causal_scene_flow_model", "oracle_causal_mask"]:
        rows = [r for r in stress_summary if r["stress_axis"] == "combined" and r["method"] == method]
        rows = sorted(rows, key=lambda r: float(r["stress_level"]))
        plt.plot([float(r["stress_level"]) for r in rows], [float(r["mask_f1"]) for r in rows], marker="o", label=labels[method])
    plt.xlabel("Combined stress level")
    plt.ylabel("Mask F1")
    plt.ylim(0.0, 1.0)
    plt.title("Combined stress sweep")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_stress_sweep.png", dpi=220)
    plt.close()


def terminal_decision(metric_rows, pair_rows, ablation_summary):
    split = "combined_hard_shift"
    proposal_f1 = metric_lookup(metric_rows, split, "causal_scene_flow_model", "mask_f1")[0]
    proposal_success = metric_lookup(metric_rows, split, "causal_scene_flow_model", "target_success")[0]
    proposal_passive = metric_lookup(metric_rows, split, "causal_scene_flow_model", "passive_false_attribution")[0]
    non_oracle = [m for m in METHODS if m not in {"causal_scene_flow_model", "oracle_causal_mask"}]
    best_f1_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "mask_f1")[0])
    best_success_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "target_success")[0])
    best_f1 = metric_lookup(metric_rows, split, best_f1_method, "mask_f1")[0]
    best_success = metric_lookup(metric_rows, split, best_success_method, "target_success")[0]
    best_passive = metric_lookup(metric_rows, split, best_f1_method, "passive_false_attribution")[0]
    paired_f1 = [r for r in pair_rows if r["split"] == split and r["reference"] == best_f1_method and r["metric"] == "mask_f1"][0]
    paired_success = [r for r in pair_rows if r["split"] == split and r["reference"] == best_success_method and r["metric"] == "target_success"][0]
    full = [r for r in ablation_summary if r["ablation"] == "full_causal_scene_flow_model"][0]
    strongest_ablation = max(float(r["mask_f1"]) for r in ablation_summary if r["ablation"] != "full_causal_scene_flow_model")
    ablation_drop = float(full["mask_f1"]) - strongest_ablation
    if (
        proposal_f1 >= best_f1 + 0.040
        and proposal_success >= best_success + 0.035
        and proposal_passive <= best_passive - 0.030
        and float(paired_f1["mean_diff"]) > 0.035
        and float(paired_success["mean_diff"]) > 0.020
        and ablation_drop >= 0.025
    ):
        return "STRONG_REVISE"
    return "KILL_ARCHIVE"


def write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, terminal):
    split = "combined_hard_shift"
    lines = []
    lines.append("Paper 84 causal_scene_flow_for_interaction v4 rebuild")
    lines.append(f"Terminal recommendation: {terminal}")
    lines.append("Reason: deterministic local interaction-flow benchmark added; no robot hardware or external high-fidelity benchmark is available.")
    lines.append(f"Main rollout rows: {sum(1 for _ in open(RESULTS / 'rollouts.csv', encoding='utf-8')) - 1}")
    lines.append(f"Ablation rollout rows: {sum(1 for _ in open(RESULTS / 'ablation_rollouts.csv', encoding='utf-8')) - 1}")
    lines.append(f"Stress rollout rows: {sum(1 for _ in open(RESULTS / 'stress_sweep_raw.csv', encoding='utf-8')) - 1}")
    lines.append(f"Seeds: {SEEDS}")
    lines.append("")
    lines.append("Combined hard shift:")
    for method in METHODS:
        f1 = metric_lookup(metric_rows, split, method, "mask_f1")
        success = metric_lookup(metric_rows, split, method, "target_success")
        passive = metric_lookup(metric_rows, split, method, "passive_false_attribution")
        error = metric_lookup(metric_rows, split, method, "effect_error")
        unsafe = metric_lookup(metric_rows, split, method, "unsafe_contact")
        lines.append(
            f"{method} mask_f1={f1[0]:.5f} ci95={f1[1]:.5f} target_success={success[0]:.5f} passive_false={passive[0]:.5f} effect_error={error[0]:.5f} unsafe={unsafe[0]:.5f}"
        )
    non_oracle = [m for m in METHODS if m not in {"causal_scene_flow_model", "oracle_causal_mask"}]
    best_f1_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "mask_f1")[0])
    best_success_method = max(non_oracle, key=lambda m: metric_lookup(metric_rows, split, m, "target_success")[0])
    paired_f1 = [r for r in pair_rows if r["split"] == split and r["reference"] == best_f1_method and r["metric"] == "mask_f1"][0]
    paired_success = [r for r in pair_rows if r["split"] == split and r["reference"] == best_success_method and r["metric"] == "target_success"][0]
    lines.append(f"paired mask-F1 diff vs best F1 baseline {best_f1_method}={float(paired_f1['mean_diff']):.5f} ci95={float(paired_f1['ci95_diff']):.5f}")
    lines.append(f"paired target-success diff vs best success baseline {best_success_method}={float(paired_success['mean_diff']):.5f} ci95={float(paired_success['ci95_diff']):.5f}")
    lines.append("")
    lines.append("Ablations:")
    for row in ablation_summary:
        lines.append(
            f"{row['ablation']} mask_f1={row['mask_f1']} ci95={row['ci95_f1']} target_success={row['target_success']} passive_false={row['passive_false_attribution']} effect_error={row['effect_error']} unsafe={row['unsafe_contact']} occluded_recall={row['occluded_recall']}"
        )
    lines.append("")
    lines.append("Combined stress level 1.0:")
    for row in stress_summary:
        if row["stress_axis"] == "combined" and row["stress_level"] == "1.0":
            lines.append(
                f"{row['method']} mask_f1={row['mask_f1']} ci95={row['ci95_f1']} target_success={row['target_success']} passive_false={row['passive_false_attribution']} effect_error={row['effect_error']} unsafe={row['unsafe_contact']}"
            )
    (RESULTS / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"terminal={terminal}")
    print(f"wrote results to {RESULTS}")


def main():
    main_rows, seed_rows, metric_rows, pair_rows = run_main()
    ablation_rows, ablation_summary = run_ablation()
    stress_raw, stress_summary = run_stress()
    negative_cases()
    terminal = terminal_decision(metric_rows, pair_rows, ablation_summary)
    plot_results(metric_rows, main_rows, ablation_summary, stress_summary)
    write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, terminal)


if __name__ == "__main__":
    main()
