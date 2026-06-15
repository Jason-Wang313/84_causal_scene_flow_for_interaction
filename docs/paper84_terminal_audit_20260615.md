# Paper 84 Terminal Audit

Date: 2026-06-15 09:17:18 +01:00

Paper: `84_causal_scene_flow_for_interaction`

Terminal decision: `KILL_ARCHIVE`

## Rerun Command

```powershell
python -m py_compile src\run_experiment.py
python src\run_experiment.py
```

The experiment runner completed successfully and printed `terminal=KILL_ARCHIVE`.

## Evidence Coverage

- `rollouts.csv`: 11,760 rows, 20 columns.
- `raw_seed_metrics.csv`: 280 rows, 14 columns.
- `metrics.csv`: 400 rows, 7 columns.
- `pairwise_stats.csv`: 210 rows, 6 columns.
- `ablation_rollouts.csv`: 2,058 rows, 20 columns.
- `ablation_seed_metrics.csv`: 49 rows, 14 columns.
- `ablation_metrics.csv`: 7 rows, 9 columns.
- `stress_sweep_raw.csv`: 25,200 rows, 22 columns.
- `stress_sweep.csv`: 150 rows, 9 columns.
- `negative_cases.csv`: 4 rows, 4 columns.

Verified seeds: `0, 1, 2, 3, 4, 5, 6`.

Verified splits: `clean_contact`, `passive_conveyor`, `articulated_coupling`, `occluded_interaction`, `combined_hard_shift`.

Verified methods: `flow_magnitude_threshold`, `action_direction_projection`, `rigid_scene_flow_cluster`, `temporal_difference_mask`, `noncausal_flow_transformer_proxy`, `learned_correlation_classifier`, `causal_scene_flow_model`, `oracle_causal_mask`.

Verified ablations: `full_causal_scene_flow_model`, `minus_action_conditioning`, `minus_passive_flow_factor`, `minus_articulation_graph`, `minus_occlusion_reasoning`, `correlation_only_causal_head`, `mask_only_no_effect_predictor`.

Verified stress axes: `passive_flow`, `occlusion`, `articulation_delay`, `distractor_contact`, `combined`.

## Main Gate

Combined hard-shift mask F1:

- `causal_scene_flow_model`: `0.76087 +/- 0.01126`.
- `learned_correlation_classifier`: `0.77230 +/- 0.01102`.
- Paired mask-F1 difference versus learned correlation: `-0.01144 +/- 0.01838`.
- Paired target-success difference versus learned correlation: `0.01701 +/- 0.08624`.
- `oracle_causal_mask`: `1.00000 +/- 0.00000`.

The proposed model is safer on unsafe contact (`0.12245` vs `0.14626`) and slightly better on mean target success (`0.56463` vs `0.54762`), but those observations do not rescue the primary attribution claim.

## Ablation Gate

- Full causal scene-flow model: `0.76087 +/- 0.01126` mask F1.
- `correlation_only_causal_head`: `0.78781 +/- 0.01137` mask F1.
- The correlation-only ablation also improves target success (`0.58844`), passive false attribution (`0.32462`), effect error (`0.55097`), and unsafe contact (`0.10204`).

This contradicts the structured causal-decomposition mechanism.

## Stress Gate

At maximum combined stress:

- `causal_scene_flow_model`: `0.74491 +/- 0.02253` mask F1, `0.50000` target success.
- `learned_correlation_classifier`: `0.77801 +/- 0.02427` mask F1, `0.48810` target success.
- `oracle_causal_mask`: `1.00000 +/- 0.00000` mask F1, `0.92262` target success.

Stress evidence does not rescue the central attribution claim.

## Submission Decision

Paper 84 is not ICLR-main ready. It should remain an archived negative result unless future work adds robot or recognized high-fidelity interaction-flow evidence, a learned causal scene-flow model trained on intervention-labeled data, strong modern scene-flow/world-model baselines, and decisive paired gains with mechanism-validating ablations.

## PDF Artifact

- Canonical PDF: `C:/Users/wangz/Downloads/84.pdf`.
- SHA256: `80E7B60BC6CD7455A8C46ECE6C8BB21BA49F94501D4CB9F451936A9FF4BB8A1E`.
- Desktop copy: absent.
