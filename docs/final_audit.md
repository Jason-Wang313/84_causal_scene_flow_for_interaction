# Final Audit

1. Chosen thesis: Causal Scene Flow for Interaction explores `Infer which scene-flow components are caused by robot action versus passive motion.` for 3D perception for manipulation.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4.
4. Reason: a local interaction-flow benchmark was added, but the causal scene-flow model loses to learned correlation on mask F1 and the correlation-only ablation improves over the full mechanism.
5. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`.
6. Reproducibility: v4 benchmark code runs and regenerates metrics/figures, but no real robot or high-fidelity benchmark is reproduced.
7. Claim-validity status: positive main-conference claims killed; v4 negative evidence audit retained.
8. Exact Downloads PDF path: `C:/Users/wangz/Downloads/84.pdf`
9. GitHub URL: https://github.com/Jason-Wang313/84_causal_scene_flow_for_interaction
10. Confirmation: no visible Desktop copy was requested or made.
11. 2026-06-15 rerun: 11,760 main rollouts, 2,058 ablation rollouts, and 25,200 stress rollouts reproduced `KILL_ARCHIVE`.
12. Hard-split gate: `causal_scene_flow_model` vs `learned_correlation_classifier` paired mask-F1 difference is `-0.01144 +/- 0.01838`.
13. Mechanism gate: `correlation_only_causal_head` improves mask F1 to `0.78781`, above the full model at `0.76087`.
