# Submission Version Log

## v1 - Generated Draft
- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening
- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed metrics, stronger baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Recompiled canonical PDF at `C:/Users/wangz/Downloads/84.pdf`.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive
- Applied the stricter ICLR-main-conference standard.
- Re-read local paper, docs, experiments, prior-work artifacts, PDF state, and repo state.
- Determined that missing real-robot/high-fidelity evidence, template-generated experiments, and unresolved novelty threats are not recoverable from local artifacts.
- Recompiled the canonical PDF with `Submission-hardening version: v3`.
- Terminal decision: KILL_ARCHIVE.

## v4 - Causal Scene-Flow Evidence Audit
- Replaced the template scaffold with a deterministic local 3D interaction-flow benchmark.
- Added point-level causal masks, eight methods, five physical-shift splits, ablations, stress sweeps, negative cases, and figures.
- Main result: causal scene-flow model loses to learned correlation classifier on combined hard-shift mask F1.
- Ablation result: correlation-only causal head improves over the full structured mechanism.
- Recompiled the canonical PDF with `Submission-hardening version: v4`.
- Terminal decision: KILL_ARCHIVE.

## v4.1 - 2026-06-15 Rerun Audit
- Added the paper-specific ICLR-main execution plan before running any new evidence.
- Re-ran `python src\run_experiment.py` from source and reproduced `terminal=KILL_ARCHIVE`.
- Verified 11,760 main rollouts, 2,058 ablation rollouts, 25,200 stress rollouts, seven seeds, eight methods, seven ablations, five stress axes, and four negative cases.
- Preserved the terminal decision because the proposed model loses the primary mask-F1 gate and the correlation-only ablation remains stronger than the full mechanism.

## v5 - 2026-06-21 Expanded Submission-Readiness Audit
- Wrote `docs/paper84_expanded_submission_plan_20260621.md` before editing or running the new protocol.
- Replaced the v4 runner with a CPU-only, RAM-light frozen protocol: 10 seeds, 128-point scenes, nine splits, 13 methods, hard-regime aggregate, paired tests, two-split ablations, six stress axes, four fixed-risk budgets, and 24 negative cases.
- Added `causal_scene_flow_model_v5`, `causal_scene_flow_model_v4`, `calibrated_tree_flow_proxy`, `contrastive_action_flow`, `scene_flow_transformer_proxy`, and `world_model_flow_predictor` comparisons.
- Regenerated 74,880 main rollouts, 5,760 scene records, 16,000 ablation rollouts, 94,080 stress rollouts, 30,720 fixed-risk rollouts, and 24 negative cases.
- Generated a 39-page ICLR-style PDF with bright clickable citation boxes at `C:/Users/wangz/Downloads/84.pdf`.
- Added `scripts/generate_manuscript.py` and `scripts/validate_submission_artifacts.py`.
- Terminal decision remains `KILL_ARCHIVE`: v5 loses to `contrastive_action_flow` on hard mask F1 and causal utility, paired lower bounds are negative, safety and ablation gates fail, and strict fixed-risk coverage collapses.
