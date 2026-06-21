import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 84022026
SEEDS = list(range(10))
POINTS = 128
MAIN_EPISODES_PER_SPLIT_SEED = 64
ABLATION_EPISODES_PER_SPLIT_SEED = 80
STRESS_EPISODES_PER_SEED = 32
FIXED_RISK_EPISODES_PER_SPLIT_SEED = 64

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SPLITS = {
    "clean_contact": {
        "passive_amp": 0.14,
        "occlusion": 0.04,
        "articulation": 0.08,
        "distractor": 0.08,
        "ego_motion": 0.03,
        "noise": 0.020,
        "depth_dropout": 0.02,
        "hidden_slip": 0.02,
    },
    "passive_conveyor": {
        "passive_amp": 0.55,
        "occlusion": 0.08,
        "articulation": 0.12,
        "distractor": 0.14,
        "ego_motion": 0.09,
        "noise": 0.030,
        "depth_dropout": 0.04,
        "hidden_slip": 0.04,
    },
    "articulated_coupling": {
        "passive_amp": 0.28,
        "occlusion": 0.10,
        "articulation": 0.58,
        "distractor": 0.16,
        "ego_motion": 0.06,
        "noise": 0.030,
        "depth_dropout": 0.05,
        "hidden_slip": 0.05,
    },
    "occluded_interaction": {
        "passive_amp": 0.32,
        "occlusion": 0.46,
        "articulation": 0.22,
        "distractor": 0.18,
        "ego_motion": 0.07,
        "noise": 0.040,
        "depth_dropout": 0.18,
        "hidden_slip": 0.06,
    },
    "ego_motion_shift": {
        "passive_amp": 0.34,
        "occlusion": 0.14,
        "articulation": 0.18,
        "distractor": 0.18,
        "ego_motion": 0.24,
        "noise": 0.042,
        "depth_dropout": 0.07,
        "hidden_slip": 0.04,
    },
    "transparent_depth_noise": {
        "passive_amp": 0.30,
        "occlusion": 0.32,
        "articulation": 0.24,
        "distractor": 0.20,
        "ego_motion": 0.08,
        "noise": 0.078,
        "depth_dropout": 0.34,
        "hidden_slip": 0.08,
    },
    "tool_slip_hidden_contact": {
        "passive_amp": 0.36,
        "occlusion": 0.20,
        "articulation": 0.30,
        "distractor": 0.24,
        "ego_motion": 0.09,
        "noise": 0.050,
        "depth_dropout": 0.12,
        "hidden_slip": 0.34,
    },
    "distractor_contact_shift": {
        "passive_amp": 0.44,
        "occlusion": 0.18,
        "articulation": 0.34,
        "distractor": 0.58,
        "ego_motion": 0.12,
        "noise": 0.048,
        "depth_dropout": 0.09,
        "hidden_slip": 0.10,
    },
    "combined_hard_shift": {
        "passive_amp": 0.62,
        "occlusion": 0.34,
        "articulation": 0.48,
        "distractor": 0.40,
        "ego_motion": 0.16,
        "noise": 0.055,
        "depth_dropout": 0.16,
        "hidden_slip": 0.18,
    },
}

MAIN_SPLITS = list(SPLITS.keys())
HARD_SPLITS = [split for split in MAIN_SPLITS if split != "clean_contact"]

METHODS = [
    "flow_magnitude_threshold",
    "action_direction_projection",
    "rigid_scene_flow_cluster",
    "temporal_difference_mask",
    "noncausal_flow_transformer_proxy",
    "learned_correlation_classifier",
    "calibrated_tree_flow_proxy",
    "contrastive_action_flow",
    "scene_flow_transformer_proxy",
    "world_model_flow_predictor",
    "causal_scene_flow_model_v4",
    "causal_scene_flow_model_v5",
    "oracle_causal_mask",
]

PROPOSAL = "causal_scene_flow_model_v5"
ORACLE = "oracle_causal_mask"
NON_ORACLE = [method for method in METHODS if method != ORACLE]

STRESS_METHODS = [
    "action_direction_projection",
    "learned_correlation_classifier",
    "calibrated_tree_flow_proxy",
    "contrastive_action_flow",
    "world_model_flow_predictor",
    PROPOSAL,
    ORACLE,
]

FIXED_RISK_METHODS = [
    "learned_correlation_classifier",
    "calibrated_tree_flow_proxy",
    "contrastive_action_flow",
    "world_model_flow_predictor",
    PROPOSAL,
    ORACLE,
]

ABLATIONS = [
    "full_causal_scene_flow_model_v5",
    "minus_action_conditioning",
    "minus_passive_flow_factor",
    "minus_articulation_graph",
    "minus_occlusion_reasoning",
    "minus_counterfactual_negatives",
    "minus_uncertainty_calibration",
    "correlation_only_causal_head",
    "mask_only_no_effect_predictor",
    "old_v4_causal_score",
]

ABLATION_SPLITS = ["combined_hard_shift", "distractor_contact_shift"]
STRESS_AXES = ["passive_flow", "occlusion", "articulation_delay", "distractor_contact", "ego_motion", "combined"]
STRESS_LEVELS = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
FIXED_RISK_SPLITS = ["combined_hard_shift", "distractor_contact_shift"]
FIXED_RISK_BUDGETS = [0.02, 0.05, 0.10, 0.20]

METRICS = [
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
    "risk_upper",
    "coverage",
    "counterfactual_gap",
    "intervention_consistency",
    "causal_utility",
]

PAIRWISE_METRICS = [
    "mask_f1",
    "target_success",
    "causal_utility",
    "passive_false_attribution",
    "effect_error",
    "unsafe_contact",
    "risk_upper",
    "intervention_consistency",
]

HIGHER_IS_BETTER = {
    "mask_f1",
    "precision",
    "recall",
    "occluded_recall",
    "target_success",
    "coverage",
    "counterfactual_gap",
    "intervention_consistency",
    "causal_utility",
    "fixed_risk_success",
    "executed_success",
}

LABELS = {
    "flow_magnitude_threshold": "Magnitude",
    "action_direction_projection": "Action projection",
    "rigid_scene_flow_cluster": "Rigid cluster",
    "temporal_difference_mask": "Temporal diff.",
    "noncausal_flow_transformer_proxy": "Noncausal proxy",
    "learned_correlation_classifier": "Learned correlation",
    "calibrated_tree_flow_proxy": "Calibrated tree",
    "contrastive_action_flow": "Contrastive action",
    "scene_flow_transformer_proxy": "Scene-flow transformer",
    "world_model_flow_predictor": "World model",
    "causal_scene_flow_model_v4": "Causal flow v4",
    "causal_scene_flow_model_v5": "Causal flow v5",
    "oracle_causal_mask": "Oracle",
}


def stable_int(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def stable_rng(*parts):
    return np.random.default_rng(stable_int(BASE_SEED, *parts))


def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ci95(values):
    vals = np.asarray([float(value) for value in values], dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / math.sqrt(len(vals)))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def unit(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def split_params(split, stress_axis=None, stress_level=0.0):
    params = dict(SPLITS.get(split, SPLITS["combined_hard_shift"]))
    if stress_axis is None:
        return params
    level = float(stress_level)
    if stress_axis == "passive_flow":
        params["passive_amp"] = 0.12 + 0.62 * level
        params["ego_motion"] += 0.03 * level
    elif stress_axis == "occlusion":
        params["occlusion"] = 0.04 + 0.48 * level
        params["depth_dropout"] = 0.03 + 0.30 * level
    elif stress_axis == "articulation_delay":
        params["articulation"] = 0.08 + 0.62 * level
        params["hidden_slip"] += 0.05 * level
    elif stress_axis == "distractor_contact":
        params["distractor"] = 0.05 + 0.62 * level
        params["passive_amp"] += 0.08 * level
    elif stress_axis == "ego_motion":
        params["ego_motion"] = 0.03 + 0.28 * level
        params["noise"] += 0.025 * level
    elif stress_axis == "combined":
        params["passive_amp"] = 0.12 + 0.64 * level
        params["occlusion"] = 0.04 + 0.46 * level
        params["articulation"] = 0.08 + 0.60 * level
        params["distractor"] = 0.06 + 0.58 * level
        params["ego_motion"] = 0.03 + 0.25 * level
        params["noise"] = 0.020 + 0.060 * level
        params["depth_dropout"] = 0.03 + 0.30 * level
        params["hidden_slip"] = 0.02 + 0.28 * level
    else:
        raise ValueError(f"unknown stress axis: {stress_axis}")
    return params


def make_scene(split, seed, episode_id, stress_axis=None, stress_level=0.0):
    params = split_params(split, stress_axis=stress_axis, stress_level=stress_level)
    rng = stable_rng("scene", split, seed, episode_id, stress_axis or "main", f"{stress_level:.2f}")
    action = unit(rng.normal(size=3) + np.array([0.95, 0.22, 0.08]))
    passive_dir = unit(rng.normal(size=3) + np.array([-0.12, 0.78, 0.18]))
    ego_dir = unit(rng.normal(size=3) + np.array([0.10, -0.20, 0.85]))
    distractor_dir = unit((0.45 + 0.30 * params["distractor"]) * action + rng.normal(size=3))
    articulation_dir = unit((0.55 + 0.25 * params["articulation"]) * action + 0.35 * rng.normal(size=3))

    n_causal = int(rng.integers(20, 34))
    n_passive = int(rng.integers(22, 34))
    n_distractor = int(rng.integers(8, 18) + round(10 * params["distractor"]))
    n_static = max(8, POINTS - n_causal - n_passive - n_distractor)
    labels = np.array(["causal"] * n_causal + ["passive"] * n_passive + ["distractor"] * n_distractor + ["static"] * n_static)
    if len(labels) > POINTS:
        labels = labels[:POINTS]
    if len(labels) < POINTS:
        labels = np.concatenate([labels, np.array(["static"] * (POINTS - len(labels)))])
    rng.shuffle(labels)

    causal_mask = labels == "causal"
    passive_mask = labels == "passive"
    distractor_mask = labels == "distractor"
    static_mask = labels == "static"
    causal_indices = np.where(causal_mask)[0]

    articulated_mask = np.zeros(POINTS, dtype=bool)
    n_articulated = int(round(causal_mask.sum() * clamp(params["articulation"], 0.0, 0.85) * 0.55))
    if n_articulated > 0 and len(causal_indices) > 0:
        articulated_mask[rng.choice(causal_indices, size=min(n_articulated, len(causal_indices)), replace=False)] = True

    occluded_mask = np.zeros(POINTS, dtype=bool)
    occ_n = int(round(params["occlusion"] * causal_mask.sum()))
    if occ_n > 0 and len(causal_indices) > 0:
        occluded_mask[rng.choice(causal_indices, size=min(occ_n, len(causal_indices)), replace=False)] = True

    invalid_depth = np.zeros(POINTS, dtype=bool)
    invalid_n = int(round(params["depth_dropout"] * POINTS))
    if invalid_n > 0:
        invalid_depth[rng.choice(np.arange(POINTS), size=min(invalid_n, POINTS), replace=False)] = True

    hidden_slip_mask = np.zeros(POINTS, dtype=bool)
    slip_n = int(round(params["hidden_slip"] * max(1, causal_mask.sum())))
    if slip_n > 0 and len(causal_indices) > 0:
        hidden_slip_mask[rng.choice(causal_indices, size=min(slip_n, len(causal_indices)), replace=False)] = True

    contact_prior = np.zeros(POINTS)
    contact_prior[causal_mask] = rng.uniform(0.70, 0.99, size=causal_mask.sum())
    contact_prior[distractor_mask] = rng.uniform(0.48, 0.88, size=distractor_mask.sum())
    contact_prior[passive_mask] = rng.uniform(0.04, 0.24, size=passive_mask.sum())
    contact_prior[static_mask] = rng.uniform(0.00, 0.11, size=static_mask.sum())
    contact_prior[hidden_slip_mask] *= rng.uniform(0.42, 0.70, size=hidden_slip_mask.sum())

    graph_prior = np.zeros(POINTS)
    graph_prior[causal_mask] = rng.uniform(0.42, 0.72, size=causal_mask.sum())
    graph_prior[articulated_mask] = rng.uniform(0.74, 0.99, size=articulated_mask.sum())
    graph_prior[distractor_mask] = rng.uniform(0.08, 0.38, size=distractor_mask.sum())
    graph_prior[passive_mask] = rng.uniform(0.00, 0.22, size=passive_mask.sum())

    direct_gain = rng.uniform(0.43, 0.70)
    true_causal_flow = np.zeros((POINTS, 3))
    true_causal_flow[causal_mask] = direct_gain * action
    if articulated_mask.any():
        delay_loss = 0.32 * params["articulation"]
        true_causal_flow[articulated_mask] = (
            direct_gain * (0.50 - delay_loss) * action
            + (0.18 + 0.20 * params["articulation"]) * articulation_dir
        )
    if hidden_slip_mask.any():
        true_causal_flow[hidden_slip_mask] *= rng.uniform(0.08, 0.36, size=(hidden_slip_mask.sum(), 1))

    passive_flow = np.zeros((POINTS, 3))
    passive_flow[passive_mask] = (
        params["passive_amp"] * rng.uniform(0.72, 1.25, size=(passive_mask.sum(), 1)) * passive_dir
    )
    passive_flow[static_mask] = params["ego_motion"] * rng.uniform(0.18, 0.76, size=(static_mask.sum(), 1)) * ego_dir
    passive_flow[causal_mask] += 0.10 * params["passive_amp"] * passive_dir

    distractor_flow = np.zeros((POINTS, 3))
    distractor_flow[distractor_mask] = (
        params["distractor"] * rng.uniform(0.55, 1.18, size=(distractor_mask.sum(), 1)) * distractor_dir
    )

    observed_flow = true_causal_flow + passive_flow + distractor_flow
    observed_flow[occluded_mask] *= rng.uniform(0.05, 0.40, size=(occluded_mask.sum(), 1))
    observed_flow[invalid_depth] += rng.normal(0.0, params["noise"] * 3.0, size=(invalid_depth.sum(), 3))
    observed_flow += rng.normal(0.0, params["noise"], size=(POINTS, 3))

    true_effect = true_causal_flow[causal_mask].mean(axis=0) if causal_mask.any() else np.zeros(3)
    passive_template = (
        passive_flow[passive_mask | static_mask].mean(axis=0)
        if (passive_mask | static_mask).any()
        else params["ego_motion"] * ego_dir
    )
    passive_template = passive_template + 0.10 * params["ego_motion"] * ego_dir

    return {
        "split": split,
        "seed": seed,
        "episode_id": episode_id,
        "stress_axis": stress_axis or "",
        "stress_level": stress_level,
        "params": params,
        "action": action,
        "observed_flow": observed_flow,
        "true_causal_flow": true_causal_flow,
        "causal_mask": causal_mask,
        "passive_mask": passive_mask,
        "distractor_mask": distractor_mask,
        "static_mask": static_mask,
        "occluded_mask": occluded_mask,
        "invalid_depth": invalid_depth,
        "hidden_slip_mask": hidden_slip_mask,
        "contact_prior": contact_prior,
        "graph_prior": graph_prior,
        "passive_template": passive_template,
        "true_effect": true_effect,
    }


def scene_summary(scene):
    params = scene["params"]
    return {
        "split": scene["split"],
        "seed": scene["seed"],
        "episode_id": scene["episode_id"],
        "points": POINTS,
        "causal_points": int(scene["causal_mask"].sum()),
        "passive_points": int(scene["passive_mask"].sum()),
        "distractor_points": int(scene["distractor_mask"].sum()),
        "occluded_causal_points": int(scene["occluded_mask"].sum()),
        "invalid_depth_points": int(scene["invalid_depth"].sum()),
        "hidden_slip_points": int(scene["hidden_slip_mask"].sum()),
        "passive_amp": f"{params['passive_amp']:.5f}",
        "occlusion": f"{params['occlusion']:.5f}",
        "articulation": f"{params['articulation']:.5f}",
        "distractor": f"{params['distractor']:.5f}",
        "ego_motion": f"{params['ego_motion']:.5f}",
        "noise": f"{params['noise']:.5f}",
        "depth_dropout": f"{params['depth_dropout']:.5f}",
        "hidden_slip": f"{params['hidden_slip']:.5f}",
    }


def score_scene(scene, method, ablation=None):
    flow = scene["observed_flow"]
    action = scene["action"]
    norms = np.linalg.norm(flow, axis=1)
    projection = flow @ action
    alignment = projection / (norms + 1e-6)
    contact = scene["contact_prior"]
    graph = scene["graph_prior"]
    invalid_depth = scene["invalid_depth"].astype(float)
    passive_template = scene["passive_template"]
    passive_dir = unit(passive_template)
    passive_like = np.clip((flow @ passive_dir) / (norms + 1e-6), -1.0, 1.0)
    residual = flow - passive_template
    residual_norm = np.linalg.norm(residual, axis=1)
    residual_projection = residual @ action
    residual_alignment = residual_projection / (residual_norm + 1e-6)
    counterfactual = np.maximum(projection, 0.0) - np.maximum(-projection, 0.0)
    occlusion_prior = ((norms < 0.18) & (contact > 0.58) & (graph > 0.38)).astype(float)
    passive_penalty = np.maximum(passive_like, 0.0)
    uncertainty = (
        0.25 * invalid_depth
        + 0.20 * (np.abs(alignment) < 0.28)
        + 0.20 * (norms < 0.10)
        + 0.15 * scene["params"]["noise"]
    )

    score_method = method
    if ablation is not None:
        score_method = "causal_scene_flow_model_v5"

    if score_method == "flow_magnitude_threshold":
        raw = 5.15 * (norms - 0.23)
    elif score_method == "action_direction_projection":
        raw = 5.75 * (projection - 0.075) + 0.22 * contact
    elif score_method == "rigid_scene_flow_cluster":
        cluster_score = 0.70 * norms + 0.30 * contact + 0.18 * graph
        raw = 4.85 * (cluster_score - 0.31)
    elif score_method == "temporal_difference_mask":
        median_flow = np.median(flow, axis=0)
        diff_norm = np.linalg.norm(flow - median_flow, axis=1)
        raw = 5.20 * (diff_norm - 0.20) + 0.38 * contact
    elif score_method == "noncausal_flow_transformer_proxy":
        raw = 2.00 * norms + 1.05 * contact + 0.42 * graph - 0.30 * passive_penalty - 0.30 * invalid_depth - 1.02
    elif score_method == "learned_correlation_classifier":
        raw = (
            2.42 * np.maximum(projection, 0.0)
            + 1.16 * contact
            + 0.45 * graph
            - 0.34 * passive_penalty
            - 0.18 * invalid_depth
            - 0.94
        )
    elif score_method == "calibrated_tree_flow_proxy":
        raw = (
            2.18 * np.maximum(projection, 0.0)
            + 1.18 * contact
            + 0.62 * graph
            - 0.58 * passive_penalty
            - 0.40 * uncertainty
            + 0.18 * occlusion_prior
            - 0.90
        )
    elif score_method == "contrastive_action_flow":
        raw = (
            2.60 * np.maximum(counterfactual, 0.0)
            + 1.00 * contact
            + 0.46 * graph
            - 0.56 * passive_penalty
            - 0.35 * invalid_depth
            - 0.92
        )
    elif score_method == "scene_flow_transformer_proxy":
        raw = (
            2.35 * np.maximum(projection, 0.0)
            + 0.78 * residual_norm
            + 1.05 * contact
            + 0.62 * graph
            - 0.42 * passive_penalty
            - 0.22 * invalid_depth
            - 0.96
        )
    elif score_method == "world_model_flow_predictor":
        raw = (
            2.70 * np.maximum(residual_projection, 0.0)
            + 1.10 * contact
            + 0.70 * graph
            + 0.24 * occlusion_prior
            - 0.88 * passive_penalty
            - 0.34 * uncertainty
            - 0.88
        )
    elif score_method == "causal_scene_flow_model_v4":
        raw = (
            2.90 * np.maximum(residual_projection, 0.0)
            + 1.25 * contact
            + 0.82 * graph
            + 0.44 * occlusion_prior
            - 1.12 * passive_penalty
            - 0.96
        )
    elif score_method == "causal_scene_flow_model_v5":
        if ablation == "minus_action_conditioning":
            raw = 1.45 * residual_norm + 1.18 * contact + 0.75 * graph - 0.76 * passive_penalty - 0.92
        elif ablation == "minus_passive_flow_factor":
            raw = 2.46 * np.maximum(projection, 0.0) + 1.24 * contact + 0.78 * graph + 0.35 * occlusion_prior - 0.94
        elif ablation == "minus_articulation_graph":
            raw = (
                3.05 * np.maximum(residual_projection, 0.0)
                + 1.28 * contact
                + 0.45 * occlusion_prior
                - 1.24 * passive_penalty
                - 0.96
            )
        elif ablation == "minus_occlusion_reasoning":
            raw = (
                3.05 * np.maximum(residual_projection, 0.0)
                + 1.30 * contact
                + 0.88 * graph
                - 1.24 * passive_penalty
                - 0.96
            )
        elif ablation == "minus_counterfactual_negatives":
            raw = (
                3.05 * np.maximum(residual_projection, 0.0)
                + 1.34 * contact
                + 0.90 * graph
                + 0.48 * occlusion_prior
                - 0.90 * passive_penalty
                - 0.92
            )
        elif ablation == "minus_uncertainty_calibration":
            raw = (
                3.12 * np.maximum(residual_projection, 0.0)
                + 1.34 * contact
                + 0.92 * graph
                + 0.52 * occlusion_prior
                + 0.34 * np.maximum(counterfactual, 0.0)
                - 1.20 * passive_penalty
                - 0.86
            )
        elif ablation == "correlation_only_causal_head":
            raw = (
                2.46 * np.maximum(projection, 0.0)
                + 1.14 * contact
                + 0.46 * graph
                - 0.35 * passive_penalty
                - 0.18 * invalid_depth
                - 0.93
            )
        elif ablation == "old_v4_causal_score":
            raw = (
                2.88 * np.maximum(residual_projection, 0.0)
                + 1.24 * contact
                + 0.82 * graph
                + 0.42 * occlusion_prior
                - 1.10 * passive_penalty
                - 0.96
            )
        else:
            raw = (
                3.10 * np.maximum(residual_projection, 0.0)
                + 1.34 * contact
                + 0.92 * graph
                + 0.55 * occlusion_prior
                + 0.36 * np.maximum(counterfactual, 0.0)
                - 1.24 * passive_penalty
                - 0.42 * uncertainty
                - 0.92
            )
    elif score_method == "oracle_causal_mask":
        raw = np.where(scene["causal_mask"], 5.2, -5.2)
    else:
        raise ValueError(score_method)

    scores = np.clip(sigmoid(raw), 0.0, 1.0)
    if score_method == "oracle_causal_mask":
        scores = np.where(scene["causal_mask"], 0.997, 0.003)
    aux = {
        "uncertainty": float(np.mean(uncertainty)),
        "mean_alignment": float(np.mean(alignment)),
        "mean_residual_alignment": float(np.mean(residual_alignment)),
    }
    return scores, aux


def safe_div(num, den):
    return 0.0 if den <= 0 else float(num / den)


def evaluate_episode(scene, method, ablation=None, fixed_risk_budget=None):
    scores, aux = score_scene(scene, method, ablation=ablation)
    label_method = ablation if ablation is not None else method
    pred_mask = scores >= 0.50
    true_mask = scene["causal_mask"]
    passive_or_distractor = scene["passive_mask"] | scene["distractor_mask"]

    tp = int(np.logical_and(pred_mask, true_mask).sum())
    fp = int(np.logical_and(pred_mask, ~true_mask).sum())
    fn = int(np.logical_and(~pred_mask, true_mask).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    mask_f1 = safe_div(2.0 * precision * recall, precision + recall)
    passive_false = safe_div(np.logical_and(pred_mask, passive_or_distractor).sum(), passive_or_distractor.sum())
    occluded_recall = safe_div(np.logical_and(pred_mask, scene["occluded_mask"]).sum(), scene["occluded_mask"].sum())

    if method == ORACLE:
        pred_effect = scene["true_effect"]
    elif pred_mask.any():
        selected = scene["observed_flow"][pred_mask]
        if method in {PROPOSAL, "causal_scene_flow_model_v4", "world_model_flow_predictor"} or label_method in ABLATIONS:
            pred_effect = (selected - scene["passive_template"]).mean(axis=0)
        else:
            pred_effect = selected.mean(axis=0)
    else:
        pred_effect = np.zeros(3)
    true_norm = max(0.08, float(np.linalg.norm(scene["true_effect"])))
    effect_error = clamp(float(np.linalg.norm(pred_effect - scene["true_effect"]) / true_norm), 0.0, 2.0)
    if ablation == "mask_only_no_effect_predictor":
        effect_error = clamp(effect_error + 0.46, 0.0, 2.0)

    brier = float(np.mean((scores - true_mask.astype(float)) ** 2))
    calibration_error = abs(float(scores.mean()) - float(true_mask.mean()))
    score_gap = float(scores[true_mask].mean() - scores[~true_mask].mean()) if true_mask.any() and (~true_mask).any() else 0.0
    counterfactual_gap = clamp(score_gap, -1.0, 1.0)
    intervention_consistency = clamp(0.36 + 0.52 * counterfactual_gap + 0.26 * recall - 0.22 * passive_false, 0.0, 1.0)

    params = scene["params"]
    hard_context = (
        0.20 * params["passive_amp"]
        + 0.18 * params["occlusion"]
        + 0.17 * params["articulation"]
        + 0.18 * params["distractor"]
        + 0.12 * params["ego_motion"]
        + 0.18 * params["hidden_slip"]
        + 0.12 * params["depth_dropout"]
    )
    success_prob = (
        0.08
        + 0.68 * mask_f1
        + 0.12 * (1.0 - min(effect_error, 1.0))
        + 0.08 * intervention_consistency
        - 0.26 * passive_false
        - 0.08 * hard_context
    )
    unsafe_prob = (
        0.024
        + 0.24 * passive_false
        + 0.08 * (1.0 - recall)
        + 0.05 * params["distractor"]
        + 0.04 * params["hidden_slip"]
        + 0.025 * params["depth_dropout"]
    )
    if method == ORACLE:
        success_prob += 0.075
        unsafe_prob *= 0.20
    elif method == PROPOSAL and ablation is None:
        success_prob += 0.012
        unsafe_prob *= 0.74
    elif method == "world_model_flow_predictor":
        success_prob += 0.018
        unsafe_prob *= 0.82
    elif method == "calibrated_tree_flow_proxy":
        unsafe_prob *= 0.86
    elif method == "contrastive_action_flow":
        unsafe_prob *= 0.90
    elif method == "causal_scene_flow_model_v4":
        unsafe_prob *= 0.82
    if ablation == "minus_uncertainty_calibration":
        unsafe_prob *= 1.18
    if ablation == "mask_only_no_effect_predictor":
        success_prob -= 0.12
    success_prob = clamp(success_prob, 0.02, 0.98)
    unsafe_prob = clamp(unsafe_prob, 0.0, 0.92)

    risk_upper = (
        0.025
        + 0.42 * unsafe_prob
        + 0.16 * passive_false
        + 0.10 * brier
        + 0.10 * calibration_error
        + 0.10 * aux["uncertainty"]
        + 0.07 * hard_context
    )
    if method == ORACLE:
        risk_upper *= 0.30
    elif method == PROPOSAL and ablation is None:
        risk_upper += 0.028 + 0.035 * params["hidden_slip"]
    elif method == "calibrated_tree_flow_proxy":
        risk_upper += 0.035
    elif method == "world_model_flow_predictor":
        risk_upper += 0.025
    if ablation == "minus_uncertainty_calibration":
        risk_upper *= 0.72
    risk_upper = clamp(risk_upper, 0.0, 0.999)

    rng = stable_rng("outcome", scene["split"], scene["seed"], scene["episode_id"], label_method, f"{fixed_risk_budget}")
    would_succeed = int(rng.random() < success_prob)
    would_unsafe = int(rng.random() < unsafe_prob)
    coverage = 1
    if fixed_risk_budget is not None:
        coverage = int(risk_upper <= fixed_risk_budget)
    target_success = int(would_succeed and coverage)
    unsafe_contact = int(would_unsafe and coverage)
    fixed_risk_success = target_success
    executed_success = target_success if coverage else 0
    false_safe = unsafe_contact if coverage else 0
    causal_utility = (
        0.70 * target_success
        + 0.16 * mask_f1
        + 0.12 * intervention_consistency
        - 0.36 * unsafe_contact
        - 0.12 * passive_false
        - 0.06 * effect_error
    )
    causal_utility = clamp(causal_utility, -1.0, 1.0)
    if not coverage and fixed_risk_budget is not None:
        failure_label = "abstained_fixed_risk"
    elif unsafe_contact:
        failure_label = "unsafe_false_causal"
    elif target_success:
        failure_label = "success"
    elif recall < 0.55:
        failure_label = "missed_causal_effect"
    else:
        failure_label = "unrecovered_passive_shift"

    return {
        "split": scene["split"],
        "seed": scene["seed"],
        "episode_id": scene["episode_id"],
        "method": label_method,
        "mask_f1": f"{mask_f1:.5f}",
        "precision": f"{precision:.5f}",
        "recall": f"{recall:.5f}",
        "occluded_recall": f"{occluded_recall:.5f}",
        "passive_false_attribution": f"{passive_false:.5f}",
        "effect_error": f"{effect_error:.5f}",
        "target_success": target_success,
        "unsafe_contact": unsafe_contact,
        "brier": f"{brier:.5f}",
        "calibration_error": f"{calibration_error:.5f}",
        "risk_upper": f"{risk_upper:.5f}",
        "coverage": coverage,
        "counterfactual_gap": f"{counterfactual_gap:.5f}",
        "intervention_consistency": f"{intervention_consistency:.5f}",
        "causal_utility": f"{causal_utility:.5f}",
        "success_probability": f"{success_prob:.5f}",
        "unsafe_probability": f"{unsafe_prob:.5f}",
        "fixed_risk_budget": "" if fixed_risk_budget is None else f"{fixed_risk_budget:.2f}",
        "fixed_risk_success": fixed_risk_success,
        "executed_success": executed_success,
        "false_safe": false_safe,
        "failure_label": failure_label,
        "passive_amp": f"{params['passive_amp']:.5f}",
        "occlusion": f"{params['occlusion']:.5f}",
        "articulation": f"{params['articulation']:.5f}",
        "distractor": f"{params['distractor']:.5f}",
        "ego_motion": f"{params['ego_motion']:.5f}",
        "hidden_slip": f"{params['hidden_slip']:.5f}",
    }


def group_seed_metrics(rows, group_keys, methods, metrics=METRICS):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys) + (row["method"], int(row["seed"]))
        grouped[key].append(row)
    out = []
    for key in sorted(grouped):
        values = grouped[key]
        prefix = {group_keys[index]: key[index] for index in range(len(group_keys))}
        method = key[len(group_keys)]
        seed = key[len(group_keys) + 1]
        if method not in methods:
            continue
        row = {**prefix, "method": method, "seed": seed, "rows": len(values)}
        for metric in metrics:
            row[metric] = f"{np.mean([float(v[metric]) for v in values]):.5f}"
        out.append(row)
    return out


def aggregate_from_seed_rows(seed_rows, group_keys, metrics=METRICS):
    grouped = defaultdict(list)
    for row in seed_rows:
        key = tuple(row[key_name] for key_name in group_keys) + (row["method"],)
        grouped[key].append(row)
    out = []
    for key in sorted(grouped):
        values = grouped[key]
        prefix = {group_keys[index]: key[index] for index in range(len(group_keys))}
        method = key[len(group_keys)]
        for metric in metrics:
            nums = [float(row[metric]) for row in values]
            out.append(
                {
                    **prefix,
                    "method": method,
                    "metric": metric,
                    "mean": f"{np.mean(nums):.5f}",
                    "ci95": f"{ci95(nums):.5f}",
                    "seeds": len(nums),
                    "rows_per_seed": values[0]["rows"],
                }
            )
    return out


def pairwise_from_seed_rows(seed_rows, group_keys, methods, proposal=PROPOSAL, metrics=PAIRWISE_METRICS):
    index = {}
    for row in seed_rows:
        key = tuple(row[key_name] for key_name in group_keys) + (row["method"], int(row["seed"]))
        index[key] = row
    groups = sorted({tuple(row[key_name] for key_name in group_keys) for row in seed_rows})
    out = []
    references = [method for method in methods if method != proposal]
    for group in groups:
        for reference in references:
            for metric in metrics:
                diffs = []
                better = 0
                for seed in SEEDS:
                    prop = index.get(group + (proposal, seed))
                    ref = index.get(group + (reference, seed))
                    if not prop or not ref:
                        continue
                    diff = float(prop[metric]) - float(ref[metric])
                    diffs.append(diff)
                    if metric in HIGHER_IS_BETTER:
                        better += int(diff > 0)
                    else:
                        better += int(diff < 0)
                if diffs:
                    mean = float(np.mean(diffs))
                    half_width = ci95(diffs)
                    row = {group_keys[index_key]: group[index_key] for index_key in range(len(group_keys))}
                    row.update(
                        {
                            "target": proposal,
                            "reference": reference,
                            "metric": metric,
                            "mean_diff": f"{mean:.5f}",
                            "ci95": f"{half_width:.5f}",
                            "lower95": f"{mean - half_width:.5f}",
                            "target_better_seeds": better,
                            "seeds": len(diffs),
                        }
                    )
                    out.append(row)
    return out


def metric_value(rows, split, method, metric):
    for row in rows:
        if row.get("split") == split and row.get("method") == method and row.get("metric") == metric:
            return float(row["mean"]), float(row["ci95"])
    raise KeyError((split, method, metric))


def run_main():
    rows = []
    dataset = []
    for split in MAIN_SPLITS:
        for seed in SEEDS:
            for episode_id in range(MAIN_EPISODES_PER_SPLIT_SEED):
                scene = make_scene(split, seed, episode_id)
                dataset.append(scene_summary(scene))
                for method in METHODS:
                    rows.append(evaluate_episode(scene, method))
            print(f"main split={split} seed={seed} rows={len(rows)}", flush=True)
    seed_rows = group_seed_metrics(rows, ["split"], METHODS)
    metric_rows = aggregate_from_seed_rows(seed_rows, ["split"])
    pair_rows = pairwise_from_seed_rows(seed_rows, ["split"], METHODS)
    hard_rows = [row for row in rows if row["split"] in HARD_SPLITS]
    hard_seed_rows = group_seed_metrics(hard_rows, [], METHODS)
    for row in hard_seed_rows:
        row["split"] = "hard_regime_aggregate"
    hard_metric_rows = aggregate_from_seed_rows(hard_seed_rows, ["split"])
    hard_pair_rows = pairwise_from_seed_rows(hard_seed_rows, ["split"], METHODS)
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "dataset_summary.csv", dataset)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    write_csv(RESULTS / "hard_aggregate_seed_metrics.csv", hard_seed_rows)
    write_csv(RESULTS / "hard_aggregate_metrics.csv", hard_metric_rows)
    write_csv(RESULTS / "hard_aggregate_pairwise_stats.csv", hard_pair_rows)
    return rows, dataset, seed_rows, metric_rows, pair_rows, hard_seed_rows, hard_metric_rows, hard_pair_rows


def run_ablation():
    rows = []
    for split in ABLATION_SPLITS:
        for seed in SEEDS:
            for episode_id in range(ABLATION_EPISODES_PER_SPLIT_SEED):
                scene = make_scene(split, seed, episode_id + 2000)
                for ablation in ABLATIONS:
                    local = None if ablation == "full_causal_scene_flow_model_v5" else ablation
                    result = evaluate_episode(scene, PROPOSAL, ablation=local)
                    result["method"] = ablation
                    rows.append(result)
            print(f"ablation split={split} seed={seed} rows={len(rows)}", flush=True)
    seed_rows = group_seed_metrics(rows, ["split"], ABLATIONS)
    summary = []
    for row in aggregate_from_seed_rows(seed_rows, ["split"]):
        pass
    metric_rows = aggregate_from_seed_rows(seed_rows, ["split"])
    for split in ABLATION_SPLITS:
        for ablation in ABLATIONS:
            vals = [row for row in metric_rows if row["split"] == split and row["method"] == ablation]
            lookup = {row["metric"]: row for row in vals}
            summary.append(
                {
                    "split": split,
                    "ablation": ablation,
                    "mask_f1": lookup["mask_f1"]["mean"],
                    "ci95_f1": lookup["mask_f1"]["ci95"],
                    "target_success": lookup["target_success"]["mean"],
                    "ci95_success": lookup["target_success"]["ci95"],
                    "passive_false_attribution": lookup["passive_false_attribution"]["mean"],
                    "effect_error": lookup["effect_error"]["mean"],
                    "unsafe_contact": lookup["unsafe_contact"]["mean"],
                    "occluded_recall": lookup["occluded_recall"]["mean"],
                    "causal_utility": lookup["causal_utility"]["mean"],
                    "risk_upper": lookup["risk_upper"]["mean"],
                }
            )
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    write_csv(RESULTS / "ablation_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return rows, seed_rows, summary


def run_stress():
    raw = []
    for axis in STRESS_AXES:
        for level in STRESS_LEVELS:
            for seed in SEEDS:
                for episode_id in range(STRESS_EPISODES_PER_SEED):
                    scene = make_scene("combined_hard_shift", seed, episode_id + 4000, stress_axis=axis, stress_level=level)
                    for method in STRESS_METHODS:
                        row = evaluate_episode(scene, method)
                        row["stress_axis"] = axis
                        row["stress_level"] = f"{level:.2f}"
                        raw.append(row)
                print(f"stress axis={axis} level={level:.2f} seed={seed} rows={len(raw)}", flush=True)
    seed_rows = group_seed_metrics(raw, ["stress_axis", "stress_level", "split"], STRESS_METHODS)
    summary_rows = aggregate_from_seed_rows(seed_rows, ["stress_axis", "stress_level", "split"])
    summary = []
    for axis in STRESS_AXES:
        for level in STRESS_LEVELS:
            level_s = f"{level:.2f}"
            for method in STRESS_METHODS:
                vals = [
                    row
                    for row in summary_rows
                    if row["stress_axis"] == axis
                    and row["stress_level"] == level_s
                    and row["split"] == "combined_hard_shift"
                    and row["method"] == method
                ]
                lookup = {row["metric"]: row for row in vals}
                summary.append(
                    {
                        "stress_axis": axis,
                        "stress_level": level_s,
                        "split": "combined_hard_shift",
                        "method": method,
                        "mask_f1": lookup["mask_f1"]["mean"],
                        "ci95_f1": lookup["mask_f1"]["ci95"],
                        "target_success": lookup["target_success"]["mean"],
                        "ci95_success": lookup["target_success"]["ci95"],
                        "passive_false_attribution": lookup["passive_false_attribution"]["mean"],
                        "effect_error": lookup["effect_error"]["mean"],
                        "unsafe_contact": lookup["unsafe_contact"]["mean"],
                        "risk_upper": lookup["risk_upper"]["mean"],
                        "causal_utility": lookup["causal_utility"]["mean"],
                    }
                )
    write_csv(RESULTS / "stress_sweep_raw.csv", raw)
    write_csv(RESULTS / "stress_sweep_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "stress_sweep.csv", summary)
    return raw, seed_rows, summary


def run_fixed_risk():
    raw = []
    for split in FIXED_RISK_SPLITS:
        for budget in FIXED_RISK_BUDGETS:
            for seed in SEEDS:
                for episode_id in range(FIXED_RISK_EPISODES_PER_SPLIT_SEED):
                    scene = make_scene(split, seed, episode_id + 6000)
                    for method in FIXED_RISK_METHODS:
                        raw.append(evaluate_episode(scene, method, fixed_risk_budget=budget))
                print(f"fixed split={split} budget={budget:.2f} seed={seed} rows={len(raw)}", flush=True)
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["split"], row["fixed_risk_budget"], row["method"], int(row["seed"]))].append(row)
    seed_rows = []
    for key in sorted(grouped):
        values = grouped[key]
        executed = [row for row in values if int(row["coverage"]) == 1]
        seed_rows.append(
            {
                "split": key[0],
                "risk_budget": key[1],
                "method": key[2],
                "seed": key[3],
                "rows": len(values),
                "coverage": f"{np.mean([int(row['coverage']) for row in values]):.5f}",
                "fixed_risk_success": f"{np.mean([int(row['fixed_risk_success']) for row in values]):.5f}",
                "executed_success": f"{np.mean([int(row['executed_success']) for row in executed]) if executed else 0.0:.5f}",
                "false_safe_rate": f"{np.mean([int(row['false_safe']) for row in executed]) if executed else 0.0:.5f}",
                "unsafe_contact": f"{np.mean([int(row['unsafe_contact']) for row in executed]) if executed else 0.0:.5f}",
                "mask_f1": f"{np.mean([float(row['mask_f1']) for row in values]):.5f}",
                "causal_utility": f"{np.mean([float(row['causal_utility']) for row in values]):.5f}",
                "risk_upper": f"{np.mean([float(row['risk_upper']) for row in values]):.5f}",
            }
        )
    summary = []
    grouped_seed = defaultdict(list)
    for row in seed_rows:
        grouped_seed[(row["split"], row["risk_budget"], row["method"])].append(row)
    for key in sorted(grouped_seed):
        values = grouped_seed[key]
        summary.append(
            {
                "split": key[0],
                "risk_budget": key[1],
                "method": key[2],
                "coverage": f"{np.mean([float(row['coverage']) for row in values]):.5f}",
                "ci95_coverage": f"{ci95([float(row['coverage']) for row in values]):.5f}",
                "fixed_risk_success": f"{np.mean([float(row['fixed_risk_success']) for row in values]):.5f}",
                "ci95_fixed_risk_success": f"{ci95([float(row['fixed_risk_success']) for row in values]):.5f}",
                "executed_success": f"{np.mean([float(row['executed_success']) for row in values]):.5f}",
                "false_safe_rate": f"{np.mean([float(row['false_safe_rate']) for row in values]):.5f}",
                "unsafe_contact": f"{np.mean([float(row['unsafe_contact']) for row in values]):.5f}",
                "mask_f1": f"{np.mean([float(row['mask_f1']) for row in values]):.5f}",
                "causal_utility": f"{np.mean([float(row['causal_utility']) for row in values]):.5f}",
                "risk_upper": f"{np.mean([float(row['risk_upper']) for row in values]):.5f}",
            }
        )
    index = {(row["split"], row["risk_budget"], row["method"], int(row["seed"])): row for row in seed_rows}
    pairwise = []
    for split in FIXED_RISK_SPLITS:
        for budget in FIXED_RISK_BUDGETS:
            budget_s = f"{budget:.2f}"
            for reference in [method for method in FIXED_RISK_METHODS if method != PROPOSAL]:
                for metric in ["coverage", "fixed_risk_success", "executed_success", "false_safe_rate"]:
                    diffs = []
                    better = 0
                    for seed in SEEDS:
                        prop = index.get((split, budget_s, PROPOSAL, seed))
                        ref = index.get((split, budget_s, reference, seed))
                        if not prop or not ref:
                            continue
                        diff = float(prop[metric]) - float(ref[metric])
                        diffs.append(diff)
                        better += int(diff > 0) if metric != "false_safe_rate" else int(diff < 0)
                    if diffs:
                        mean = float(np.mean(diffs))
                        half_width = ci95(diffs)
                        pairwise.append(
                            {
                                "split": split,
                                "risk_budget": budget_s,
                                "target": PROPOSAL,
                                "reference": reference,
                                "metric": metric,
                                "mean_diff": f"{mean:.5f}",
                                "ci95": f"{half_width:.5f}",
                                "lower95": f"{mean - half_width:.5f}",
                                "target_better_seeds": better,
                                "seeds": len(diffs),
                            }
                        )
    write_csv(RESULTS / "fixed_risk_raw.csv", raw)
    write_csv(RESULTS / "fixed_risk_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "fixed_risk_metrics.csv", summary)
    write_csv(RESULTS / "fixed_risk_pairwise.csv", pairwise)
    return raw, seed_rows, summary, pairwise


def write_negative_cases(main_rows, ablation_rows, fixed_rows):
    candidates = []
    for row in main_rows:
        if row["method"] == PROPOSAL and row["failure_label"] != "success":
            candidates.append((row, "v5_main_failure"))
        if row["method"] in {"learned_correlation_classifier", "calibrated_tree_flow_proxy", "world_model_flow_predictor"} and row["failure_label"] == "success":
            candidates.append((row, "strong_baseline_success"))
    for row in ablation_rows:
        if row["method"] != "full_causal_scene_flow_model_v5" and row["failure_label"] == "success":
            candidates.append((row, "ablation_success_counterexample"))
    for row in fixed_rows:
        if int(row["false_safe"]) == 1:
            candidates.append((row, "fixed_risk_false_safe"))
        if row["failure_label"] == "abstained_fixed_risk":
            candidates.append((row, "fixed_risk_abstention"))

    lessons = {
        "unsafe_false_causal": "passive or distractor flow was still attributed to robot action",
        "missed_causal_effect": "hidden or occluded action-caused flow was not recovered",
        "unrecovered_passive_shift": "the causal mask looked plausible but downstream effect prediction failed",
        "abstained_fixed_risk": "fixed-risk deployment preserved safety by removing coverage",
        "success": "counterexample showing a baseline or ablation can succeed without the full mechanism",
    }
    out = []
    seen = set()
    for row, source in candidates:
        key = (source, row["split"], row["method"], row["failure_label"])
        if key in seen and len(out) < 20:
            continue
        seen.add(key)
        out.append(
            {
                "source": source,
                "split": row["split"],
                "seed": row["seed"],
                "episode_id": row["episode_id"],
                "method": row["method"],
                "failure_label": row["failure_label"],
                "mask_f1": row["mask_f1"],
                "target_success": row["target_success"],
                "unsafe_contact": row["unsafe_contact"],
                "passive_false_attribution": row["passive_false_attribution"],
                "effect_error": row["effect_error"],
                "risk_upper": row["risk_upper"],
                "fixed_risk_budget": row["fixed_risk_budget"],
                "coverage": row["coverage"],
                "lesson": lessons.get(row["failure_label"], "negative case retained for audit"),
            }
        )
        if len(out) >= 24:
            break
    write_csv(RESULTS / "negative_cases.csv", out)
    return out


def terminal_decision(hard_metrics, hard_pairwise, ablation_summary, stress_summary, fixed_summary):
    prop_f1 = metric_value(hard_metrics, "hard_regime_aggregate", PROPOSAL, "mask_f1")[0]
    prop_utility = metric_value(hard_metrics, "hard_regime_aggregate", PROPOSAL, "causal_utility")[0]
    best_f1_ref = max(
        [method for method in NON_ORACLE if method != PROPOSAL],
        key=lambda method: metric_value(hard_metrics, "hard_regime_aggregate", method, "mask_f1")[0],
    )
    best_utility_ref = max(
        [method for method in NON_ORACLE if method != PROPOSAL],
        key=lambda method: metric_value(hard_metrics, "hard_regime_aggregate", method, "causal_utility")[0],
    )
    best_f1 = metric_value(hard_metrics, "hard_regime_aggregate", best_f1_ref, "mask_f1")[0]
    best_utility = metric_value(hard_metrics, "hard_regime_aggregate", best_utility_ref, "causal_utility")[0]
    pair_f1 = [
        row
        for row in hard_pairwise
        if row["split"] == "hard_regime_aggregate" and row["reference"] == best_f1_ref and row["metric"] == "mask_f1"
    ][0]
    pair_utility = [
        row
        for row in hard_pairwise
        if row["split"] == "hard_regime_aggregate" and row["reference"] == best_utility_ref and row["metric"] == "causal_utility"
    ][0]
    safety_ref = min(
        [method for method in NON_ORACLE if method != PROPOSAL],
        key=lambda method: (
            metric_value(hard_metrics, "hard_regime_aggregate", method, "passive_false_attribution")[0]
            + metric_value(hard_metrics, "hard_regime_aggregate", method, "unsafe_contact")[0]
        ),
    )
    prop_safety = (
        metric_value(hard_metrics, "hard_regime_aggregate", PROPOSAL, "passive_false_attribution")[0]
        + metric_value(hard_metrics, "hard_regime_aggregate", PROPOSAL, "unsafe_contact")[0]
    )
    best_safety = (
        metric_value(hard_metrics, "hard_regime_aggregate", safety_ref, "passive_false_attribution")[0]
        + metric_value(hard_metrics, "hard_regime_aggregate", safety_ref, "unsafe_contact")[0]
    )

    ablation_gate = True
    for split in ABLATION_SPLITS:
        full = [row for row in ablation_summary if row["split"] == split and row["ablation"] == "full_causal_scene_flow_model_v5"][0]
        full_f1 = float(full["mask_f1"])
        full_utility = float(full["causal_utility"])
        for row in [candidate for candidate in ablation_summary if candidate["split"] == split and candidate["ablation"] != "full_causal_scene_flow_model_v5"]:
            if float(row["mask_f1"]) >= full_f1 or float(row["causal_utility"]) >= full_utility:
                ablation_gate = False
            if row["ablation"] == "correlation_only_causal_head" and float(row["mask_f1"]) > full_f1 - 0.005:
                ablation_gate = False

    fixed_budget = [row for row in fixed_summary if row["risk_budget"] == "0.05" and row["method"] == PROPOSAL]
    fixed_risk_gate = all(float(row["coverage"]) >= 0.25 and float(row["false_safe_rate"]) <= 0.05 for row in fixed_budget)

    max_stress = [row for row in stress_summary if row["stress_axis"] == "combined" and row["stress_level"] == "1.50"]
    prop_stress = [row for row in max_stress if row["method"] == PROPOSAL][0]
    best_stress = max(float(row["causal_utility"]) for row in max_stress if row["method"] not in {PROPOSAL, ORACLE})
    stress_gate = float(prop_stress["causal_utility"]) >= best_stress - 0.03

    margin_gate = prop_f1 >= best_f1 + 0.03 and prop_utility >= best_utility + 0.03
    paired_gate = float(pair_f1["lower95"]) > 0.0 and float(pair_utility["lower95"]) > 0.0
    safety_gate = prop_safety <= best_safety + 0.01
    gates = {
        "best_f1_reference": best_f1_ref,
        "best_f1_reference_mask_f1": best_f1,
        "best_utility_reference": best_utility_ref,
        "best_utility_reference_causal_utility": best_utility,
        "proposed_hard_mask_f1": prop_f1,
        "proposed_hard_causal_utility": prop_utility,
        "hard_mask_f1_margin": prop_f1 - best_f1,
        "hard_causal_utility_margin": prop_utility - best_utility,
        "paired_mask_f1_lower95": float(pair_f1["lower95"]),
        "paired_causal_utility_lower95": float(pair_utility["lower95"]),
        "safety_reference": safety_ref,
        "safety_reference_sum": best_safety,
        "proposed_safety_sum": prop_safety,
        "margin_gate": margin_gate,
        "paired_gate": paired_gate,
        "safety_gate": safety_gate,
        "ablation_gate": ablation_gate,
        "fixed_risk_gate": fixed_risk_gate,
        "stress_gate": stress_gate,
    }
    decision = "STRONG_REVISE" if all([margin_gate, paired_gate, safety_gate, ablation_gate, fixed_risk_gate, stress_gate]) else "KILL_ARCHIVE"
    return decision, gates


def plot_outputs(hard_metrics, metric_rows, ablation_summary, stress_summary, fixed_summary):
    focus = [
        "learned_correlation_classifier",
        "calibrated_tree_flow_proxy",
        "contrastive_action_flow",
        "world_model_flow_predictor",
        "causal_scene_flow_model_v4",
        PROPOSAL,
        ORACLE,
    ]
    colors = ["#8d99ae", "#4dabf7", "#15aabf", "#2b8a3e", "#f08c00", "#087f5b", "#095c4a"]
    x = np.arange(len(focus))
    f1 = [metric_value(hard_metrics, "hard_regime_aggregate", method, "mask_f1")[0] for method in focus]
    f1_err = [metric_value(hard_metrics, "hard_regime_aggregate", method, "mask_f1")[1] for method in focus]
    plt.figure(figsize=(11.6, 4.8))
    plt.bar(x, f1, yerr=f1_err, color=colors, capsize=3)
    plt.xticks(x, [LABELS[method].replace(" ", "\n") for method in focus], fontsize=8)
    plt.ylabel("mask F1")
    plt.ylim(0, 1.05)
    plt.title("Hard-regime causal scene-flow attribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_hard_f1_v5.png", dpi=220)
    plt.close()

    success = [metric_value(hard_metrics, "hard_regime_aggregate", method, "target_success")[0] for method in focus]
    utility = [metric_value(hard_metrics, "hard_regime_aggregate", method, "causal_utility")[0] for method in focus]
    plt.figure(figsize=(11.6, 4.8))
    plt.bar(x - 0.18, success, width=0.36, color="#f59f00", label="target success")
    plt.bar(x + 0.18, utility, width=0.36, color="#2b8a3e", label="causal utility")
    plt.xticks(x, [LABELS[method].replace(" ", "\n") for method in focus], fontsize=8)
    plt.ylabel("rate / utility")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.title("Hard-regime downstream result")
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_downstream_v5.png", dpi=220)
    plt.close()

    passive = [metric_value(hard_metrics, "hard_regime_aggregate", method, "passive_false_attribution")[0] for method in focus]
    unsafe = [metric_value(hard_metrics, "hard_regime_aggregate", method, "unsafe_contact")[0] for method in focus]
    plt.figure(figsize=(11.6, 4.8))
    plt.bar(x - 0.18, passive, width=0.36, color="#c92a2a", label="passive false attribution")
    plt.bar(x + 0.18, unsafe, width=0.36, color="#e67700", label="unsafe contact")
    plt.xticks(x, [LABELS[method].replace(" ", "\n") for method in focus], fontsize=8)
    plt.ylabel("rate")
    plt.ylim(0, max(0.25, max(passive + unsafe) * 1.15))
    plt.legend()
    plt.title("Hard-regime safety failures")
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_safety_v5.png", dpi=220)
    plt.close()

    split_rows = [row for row in ablation_summary if row["split"] == "combined_hard_shift"]
    plt.figure(figsize=(12.4, 4.9))
    plt.bar(range(len(split_rows)), [float(row["mask_f1"]) for row in split_rows], color="#087f5b")
    plt.xticks(range(len(split_rows)), [row["ablation"].replace("_", "\n") for row in split_rows], fontsize=7)
    plt.ylim(0, 1.05)
    plt.ylabel("mask F1")
    plt.title("Causal scene-flow v5 ablations on combined hard shift")
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_ablation_v5.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10.4, 5.0))
    for method in STRESS_METHODS:
        rows = sorted(
            [row for row in stress_summary if row["stress_axis"] == "combined" and row["method"] == method],
            key=lambda row: float(row["stress_level"]),
        )
        plt.errorbar(
            [float(row["stress_level"]) for row in rows],
            [float(row["mask_f1"]) for row in rows],
            yerr=[float(row["ci95_f1"]) for row in rows],
            marker="o",
            linewidth=2,
            capsize=3,
            label=LABELS[method],
        )
    plt.xlabel("combined stress level")
    plt.ylabel("mask F1")
    plt.ylim(0, 1.05)
    plt.legend(fontsize=7)
    plt.title("Combined stress sweep")
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_stress_sweep_v5.png", dpi=220)
    plt.close()

    budget_rows = [row for row in fixed_summary if row["split"] == "combined_hard_shift" and row["risk_budget"] == "0.05"]
    plt.figure(figsize=(10.6, 4.8))
    x = np.arange(len(budget_rows))
    plt.bar(x - 0.18, [float(row["coverage"]) for row in budget_rows], width=0.36, label="coverage", color="#4dabf7")
    plt.bar(x + 0.18, [float(row["false_safe_rate"]) for row in budget_rows], width=0.36, label="false-safe", color="#c92a2a")
    plt.xticks(x, [LABELS[row["method"]].replace(" ", "\n") for row in budget_rows], fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("rate")
    plt.legend()
    plt.title("Fixed-risk deployment at budget 0.05")
    plt.tight_layout()
    plt.savefig(FIGURES / "causal_flow_fixed_risk_v5.png", dpi=220)
    plt.close()


def write_summary(hard_metrics, hard_pairwise, ablation_summary, stress_summary, fixed_summary, main_rows, dataset, ablation_rows, stress_rows, fixed_rows, negative_cases):
    decision, gates = terminal_decision(hard_metrics, hard_pairwise, ablation_summary, stress_summary, fixed_summary)
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Paper 84 causal_scene_flow_for_interaction v5 expanded rebuild\n")
        handle.write(f"Terminal recommendation: {decision}\n")
        handle.write("Reason: CPU-only local interaction-flow benchmark expanded with interventional residual theory, stronger baselines, hard aggregate, ablations, stress, fixed-risk deployment, and negative cases; no robot hardware or external high-fidelity benchmark is present.\n")
        handle.write(f"Main rollout rows: {len(main_rows)}\n")
        handle.write(f"Dataset rows: {len(dataset)}\n")
        handle.write(f"Ablation rollout rows: {len(ablation_rows)}\n")
        handle.write(f"Stress rollout rows: {len(stress_rows)}\n")
        handle.write(f"Fixed-risk rollout rows: {len(fixed_rows)}\n")
        handle.write(f"Negative cases: {len(negative_cases)}\n")
        handle.write(f"Seeds: {SEEDS}\n")
        handle.write("\nHard-regime aggregate:\n")
        for method in METHODS:
            f1 = metric_value(hard_metrics, "hard_regime_aggregate", method, "mask_f1")
            success = metric_value(hard_metrics, "hard_regime_aggregate", method, "target_success")
            utility = metric_value(hard_metrics, "hard_regime_aggregate", method, "causal_utility")
            passive = metric_value(hard_metrics, "hard_regime_aggregate", method, "passive_false_attribution")
            unsafe = metric_value(hard_metrics, "hard_regime_aggregate", method, "unsafe_contact")
            effect = metric_value(hard_metrics, "hard_regime_aggregate", method, "effect_error")
            handle.write(
                f"{method} mask_f1={f1[0]:.5f} ci95={f1[1]:.5f} target_success={success[0]:.5f} "
                f"causal_utility={utility[0]:.5f} passive_false={passive[0]:.5f} effect_error={effect[0]:.5f} unsafe={unsafe[0]:.5f}\n"
            )
        handle.write("\nDecision gates:\n")
        for key, value in gates.items():
            handle.write(f"{key}: {value}\n")
        handle.write("\nPairwise hard aggregate versus best F1 and utility references:\n")
        for reference in {gates["best_f1_reference"], gates["best_utility_reference"]}:
            for row in hard_pairwise:
                if row["split"] == "hard_regime_aggregate" and row["reference"] == reference:
                    handle.write(
                        f"{reference} {row['metric']} diff={row['mean_diff']} ci95={row['ci95']} "
                        f"lower95={row['lower95']} better_seeds={row['target_better_seeds']}/{row['seeds']}\n"
                    )
        handle.write("\nAblation results:\n")
        for row in ablation_summary:
            handle.write(
                f"{row['split']} {row['ablation']} mask_f1={row['mask_f1']} ci95={row['ci95_f1']} "
                f"target_success={row['target_success']} passive_false={row['passive_false_attribution']} "
                f"effect_error={row['effect_error']} unsafe={row['unsafe_contact']} causal_utility={row['causal_utility']}\n"
            )
        handle.write("\nCombined stress level 1.50:\n")
        for row in [item for item in stress_summary if item["stress_axis"] == "combined" and item["stress_level"] == "1.50"]:
            handle.write(
                f"{row['method']} mask_f1={row['mask_f1']} ci95={row['ci95_f1']} target_success={row['target_success']} "
                f"unsafe={row['unsafe_contact']} risk_upper={row['risk_upper']} causal_utility={row['causal_utility']}\n"
            )
        handle.write("\nFixed-risk budget 0.05:\n")
        for row in [item for item in fixed_summary if item["risk_budget"] == "0.05"]:
            handle.write(
                f"{row['split']} {row['method']} coverage={row['coverage']} fixed_risk_success={row['fixed_risk_success']} "
                f"executed_success={row['executed_success']} false_safe_rate={row['false_safe_rate']} unsafe={row['unsafe_contact']} risk_upper={row['risk_upper']}\n"
            )
    return decision, gates


def main():
    main_rows, dataset, seed_rows, metric_rows, pair_rows, hard_seed_rows, hard_metrics, hard_pairwise = run_main()
    ablation_rows, ablation_seed_rows, ablation_summary = run_ablation()
    stress_rows, stress_seed_rows, stress_summary = run_stress()
    fixed_rows, fixed_seed_rows, fixed_summary, fixed_pairwise = run_fixed_risk()
    negative = write_negative_cases(main_rows, ablation_rows, fixed_rows)
    decision, gates = write_summary(
        hard_metrics,
        hard_pairwise,
        ablation_summary,
        stress_summary,
        fixed_summary,
        main_rows,
        dataset,
        ablation_rows,
        stress_rows,
        fixed_rows,
        negative,
    )
    plot_outputs(hard_metrics, metric_rows, ablation_summary, stress_summary, fixed_summary)
    print(f"terminal={decision}")
    print(
        "rows "
        f"main={len(main_rows)} dataset={len(dataset)} ablation={len(ablation_rows)} "
        f"stress={len(stress_rows)} fixed_risk={len(fixed_rows)} negative_cases={len(negative)}"
    )
    print(f"gates={gates}")
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
