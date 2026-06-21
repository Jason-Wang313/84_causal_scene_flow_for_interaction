# Paper 84 Expanded Submission-Readiness Plan

Date: 2026-06-21

Target paper: `84_causal_scene_flow_for_interaction`

Frozen objective: rebuild Paper 84 into a 25+ page ICLR-main-target manuscript using only substantive new evidence, theory, method analysis, and failure analysis. Keep the entire workflow CPU-only and RAM-light. Produce the numbered PDF only at `C:\Users\wangz\Downloads\84.pdf`, never on Desktop. Push the matching public GitHub repository and update the root batch ledgers after validation.

## Starting Evidence

The v4 terminal decision is `KILL_ARCHIVE`.

- The old proposed `causal_scene_flow_model` loses the combined-hard-shift primary mask-F1 comparison to `learned_correlation_classifier`.
- The correlation-only ablation improves over the structured causal model, directly challenging the claimed mechanism.
- Evidence is local synthetic interaction-flow only, with no real robot dataset, external accepted benchmark, or high-fidelity simulator.
- Therefore the v5 rebuild must not present cosmetic gains as submission readiness. It must expose the weakness, improve the method only before the protocol is frozen, and report every predefined result honestly.

## Frozen Method Upgrade

The new tested method is `causal_scene_flow_model_v5`, an interventional residual scene-flow estimator with:

- action-conditioned counterfactual residualization;
- passive and ego-flow factor subtraction;
- contact graph and articulation-lag priors;
- occlusion-aware causal imputation;
- uncertainty calibration for fixed-risk deployment;
- explicit negative-action and distractor-contact consistency checks.

The previous v4 method remains in the comparison set as `causal_scene_flow_model_v4`, not as the final claim.

## Theory Additions

The manuscript must include:

- A formal problem setup for action-caused point flow, passive flow, observed flow, causal masks, and downstream interaction selection.
- A definition of passive false attribution, interventional effect error, causal utility, fixed-risk coverage, and false-safe rate.
- A residual-separation lemma: if passive-flow estimation error is bounded below the causal/passive residual margin, thresholded residual causal attribution has bounded passive false attribution.
- A dominance proposition: action-conditioned residualization can dominate magnitude or correlation scoring when passive flow is action-aligned but distinguishable by intervention features.
- A non-identifiability theorem: without intervention labels or negative actions, passive flow that shares action-correlated features can make causal flow unidentifiable, so correlation-only predictors can match or beat structured causal models.
- A hostile-review interpretation explaining when each theorem supports or undermines the method.

## Frozen Main Evaluation

Main evaluation constants:

- Seeds: 10 (`0` through `9`).
- Points per synthetic RGB-D point scene: 128.
- Episodes per split per seed: 64.
- Splits: `clean_contact`, `passive_conveyor`, `articulated_coupling`, `occluded_interaction`, `ego_motion_shift`, `transparent_depth_noise`, `tool_slip_hidden_contact`, `distractor_contact_shift`, `combined_hard_shift`.
- Methods: `flow_magnitude_threshold`, `action_direction_projection`, `rigid_scene_flow_cluster`, `temporal_difference_mask`, `noncausal_flow_transformer_proxy`, `learned_correlation_classifier`, `calibrated_tree_flow_proxy`, `contrastive_action_flow`, `scene_flow_transformer_proxy`, `world_model_flow_predictor`, `causal_scene_flow_model_v4`, `causal_scene_flow_model_v5`, `oracle_causal_mask`.

Expected main rollouts: `9 * 10 * 64 * 13 = 74,880`.

Main metrics:

- `mask_f1`
- `precision`
- `recall`
- `occluded_recall`
- `passive_false_attribution`
- `effect_error`
- `target_success`
- `unsafe_contact`
- `brier`
- `calibration_error`
- `risk_upper`
- `coverage`
- `counterfactual_gap`
- `intervention_consistency`
- `causal_utility`

The hard aggregate is predefined as all non-clean splits.

## Frozen Ablation Evaluation

Ablation constants:

- Splits: `combined_hard_shift`, `distractor_contact_shift`.
- Seeds: 10.
- Episodes per split per seed: 80.
- Ablations: `full_causal_scene_flow_model_v5`, `minus_action_conditioning`, `minus_passive_flow_factor`, `minus_articulation_graph`, `minus_occlusion_reasoning`, `minus_counterfactual_negatives`, `minus_uncertainty_calibration`, `correlation_only_causal_head`, `mask_only_no_effect_predictor`, `old_v4_causal_score`.

Expected ablation rollouts: `2 * 10 * 80 * 10 = 16,000`.

The full model passes the ablation gate only if removing each core mechanism reduces hard aggregate causal utility or mask F1, and if the correlation-only ablation does not beat the full model.

## Frozen Stress Evaluation

Stress constants:

- Stress axes: `passive_flow`, `occlusion`, `articulation_delay`, `distractor_contact`, `ego_motion`, `combined`.
- Stress levels: `0.00`, `0.25`, `0.50`, `0.75`, `1.00`, `1.25`, `1.50`.
- Seeds: 10.
- Episodes per axis-level-seed: 32.
- Methods: `action_direction_projection`, `learned_correlation_classifier`, `calibrated_tree_flow_proxy`, `contrastive_action_flow`, `world_model_flow_predictor`, `causal_scene_flow_model_v5`, `oracle_causal_mask`.

Expected stress rollouts: `6 * 7 * 10 * 32 * 7 = 94,080`.

## Frozen Fixed-Risk Evaluation

Fixed-risk constants:

- Splits: `combined_hard_shift`, `distractor_contact_shift`.
- Risk budgets: `0.02`, `0.05`, `0.10`, `0.20`.
- Seeds: 10.
- Episodes per split per seed: 64.
- Methods: `learned_correlation_classifier`, `calibrated_tree_flow_proxy`, `contrastive_action_flow`, `world_model_flow_predictor`, `causal_scene_flow_model_v5`, `oracle_causal_mask`.

Expected fixed-risk rollouts: `2 * 4 * 10 * 64 * 6 = 30,720`.

The fixed-risk gate passes only if `causal_scene_flow_model_v5` has coverage at risk budget `0.05` of at least `0.25` while keeping false-safe rate at or below `0.05`.

## Frozen Decision Gates

The paper may receive `STRONG_REVISE` only if all of the following pass:

- Hard aggregate mask-F1 margin over the strongest non-oracle baseline is at least `0.03`.
- Hard aggregate causal-utility or target-success margin over the strongest non-oracle baseline is at least `0.03`.
- Paired seed-level lower 95 percent confidence bounds are above zero for mask F1 and causal utility.
- Passive false attribution and unsafe contact are not worse than the safest strong non-oracle baseline by more than `0.01`.
- The ablation gate passes.
- The fixed-risk gate passes.
- The max combined-stress result is not dominated by a non-oracle baseline.

If any gate fails, the terminal decision remains `KILL_ARCHIVE`.

Even if all local gates pass, the manuscript must state that ICLR-main readiness still requires real robot RGB-D interaction data or a recognized high-fidelity benchmark. Local synthetic evidence alone cannot honestly certify acceptance-level empirical strength.

## Manuscript And Artifact Requirements

- Produce a 25+ page ICLR-style PDF.
- Use bright clickable citation boxes for in-text citations, routing to bibliography entries.
- Include the frozen protocol, theory, main results, hard aggregate, paired statistics, ablations, stress tests, fixed-risk deployment results, negative cases, limitations, and reproducibility details.
- Generate all figures and tables from local CSV outputs.
- Validate that `C:\Users\wangz\Downloads\84.pdf` exists, has at least 25 pages, and that `C:\Users\wangz\Desktop\84.pdf` does not exist.
- Validate the LaTeX log for unresolved citations, unresolved references, rerun warnings, natbib warnings, and LaTeX errors.
- Update `README.md`, `child_status.md`, local docs, root `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, `MASTER_SUBMISSION_REPORT.md`, and `SUBMISSION_AUDIT_MATRIX.csv`.
- Commit and push the child repository to the public GitHub repo.
