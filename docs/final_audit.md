# Final Audit

1. Chosen thesis: Causal Scene Flow for Interaction explores `Infer which scene-flow components are caused by robot action versus passive motion.` for 3D perception for manipulation.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v5 expanded.
4. Reason: the expanded frozen protocol shows that `causal_scene_flow_model_v5` loses to `contrastive_action_flow` on the hard-regime aggregate, has negative paired lower bounds, worsens the safety sum, fails ablation necessity, and has zero non-oracle coverage under strict fixed-risk deployment.
5. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, `docs/hostile_reviewer_response.md`, and the new prior-work pressure table in `paper/main.tex`.
6. Reproducibility: `python src\run_experiment.py` regenerates all v5 CSVs and figures. `python scripts\validate_submission_artifacts.py` validates row counts, PDF page count, citation-box settings, LaTeX resolution, Downloads-only placement, and Desktop absence.
7. Claim-validity status: positive main-conference claims killed; v5 negative evidence audit retained.
8. Exact Downloads PDF path: `C:/Users/wangz/Downloads/84.pdf`
9. GitHub URL: https://github.com/Jason-Wang313/84_causal_scene_flow_for_interaction
10. Confirmation: no visible Desktop copy was requested or made.
11. 2026-06-21 v5 rerun: 74,880 main rollouts, 5,760 scene records, 16,000 ablation rollouts, 94,080 stress rollouts, 30,720 fixed-risk rollouts, and 24 negative cases.
12. Hard-regime gate: `causal_scene_flow_model_v5` mask F1 is `0.81699 +/- 0.00175`, while `contrastive_action_flow` is `0.83774 +/- 0.00186`.
13. Paired gate: lower 95 percent bounds are `-0.02228` for mask F1 and `-0.03869` for causal utility.
14. Fixed-risk gate: at budget `0.05`, every non-oracle method has zero coverage on both fixed-risk splits.
15. Canonical PDF: 39 pages, SHA256 `03414EF5F41537EF71C9159B20370D3D859AB525C98369C47E2392959362EAF9`.
