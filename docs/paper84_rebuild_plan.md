# Paper 84 Rebuild Plan

Last update: 2026-06-14 10:46:22 +01:00

## Target Claim

Interactive robot perception should separate scene-flow components caused by the robot action from passive, exogenous, or self-motion flow. A causal scene-flow representation should improve action-effect attribution and downstream manipulation decisions under distractor motion, occlusion, and articulated coupling.

## Evidence To Build

Replace the v3 template scaffold with a deterministic local 3D interaction-flow benchmark.

### Splits

- `clean_contact`: direct manipulation with mild passive background flow.
- `passive_conveyor`: strong passive object/background motion unrelated to the robot action.
- `articulated_coupling`: robot action moves linked parts with delayed/attenuated causal flow.
- `occluded_interaction`: action-caused flow is partially hidden by sensor dropout.
- `combined_hard_shift`: passive motion, occlusion, articulation, and distractor contact occur together.

### Methods

- `flow_magnitude_threshold`
- `action_direction_projection`
- `rigid_scene_flow_cluster`
- `temporal_difference_mask`
- `noncausal_flow_transformer_proxy`
- `learned_correlation_classifier`
- `causal_scene_flow_model` (proposed)
- `oracle_causal_mask`

### Main Metrics

- causal mask F1.
- passive false-attribution rate.
- action-effect vector error.
- interaction target success.
- calibration error.
- downstream unsafe contact rate.
- paired seed-level differences against strongest baselines.

### Ablations

- full causal scene-flow model.
- minus action conditioning.
- minus passive-flow factor.
- minus articulation graph.
- minus occlusion reasoning.
- correlation-only causal head.
- mask-only no effect predictor.

### Stress Tests

- passive-flow amplitude.
- occlusion fraction.
- articulation delay.
- distractor contact strength.
- combined stress.

### Terminal Gate

Mark `STRONG_REVISE` only if the proposed model beats the strongest non-oracle baseline on combined hard-shift mask F1 and downstream interaction success, reduces passive false attribution, and ablations degrade the mechanism. Otherwise mark `KILL_ARCHIVE`.

Even a `STRONG_REVISE` outcome is not ICLR-main ready without robot or accepted high-fidelity benchmark validation.
