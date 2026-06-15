# Paper 84 ICLR-Main Submission-Readiness Execution Plan

Date: 2026-06-15

Paper: `84_causal_scene_flow_for_interaction`

Target venue standard: ICLR main conference, with an evidence-first gate. The paper can advance only if the rebuilt evidence shows that structured causal scene-flow decomposition beats strong learned correlation baselines, survives ablation, improves downstream interaction, and is not merely a local synthetic artifact.

## Current State

The repository currently reports a v4 terminal decision of `KILL_ARCHIVE`. The existing claim is that robot interaction perception should separate action-caused scene flow from passive flow, distractor flow, occlusion artifacts, articulation, and ego-motion noise. The prior audit found that `causal_scene_flow_model` loses to `learned_correlation_classifier` on combined hard-shift mask F1 and that `correlation_only_causal_head` improves over the full structured decomposition. The evidence remains local synthetic simulation, not robot hardware or an accepted high-fidelity benchmark.

## Execution Order

1. Verify repository hygiene before touching evidence.
   - Confirm the worktree is clean except for this plan.
   - Record the pre-audit commit.
   - Confirm the GitHub remote exists and is public.

2. Re-run the full evidence generator from source.
   - Compile-check `src/run_experiment.py`.
   - Run `python src/run_experiment.py`.
   - Preserve all generated CSVs, figures, and `results/summary.txt`.

3. Audit evidence completeness.
   - Confirm seven seeds are present.
   - Confirm all splits, methods, ablations, stress axes, and negative cases are represented.
   - Confirm row counts and schemas for rollout, seed metric, aggregate metric, pairwise, ablation, stress, and negative-case files.

4. Apply the ICLR-main decision gate.
   - Require the proposed model to beat the strongest non-oracle baseline on combined hard-shift mask F1.
   - Require downstream target success to improve without hiding unsafe contacts or passive false attribution.
   - Require paired seed-level effects that are not swallowed by uncertainty.
   - Require ablations to degrade when action conditioning, passive-flow factoring, articulation structure, occlusion reasoning, or effect prediction is removed.
   - Require stress tests to support the same conclusion under passive-flow amplitude, occlusion, articulation delay, distractor contact, and combined stress.

5. Decide honestly.
   - If all gates pass but evidence remains local synthetic only, mark at most `STRONG_REVISE`.
   - If mask F1 loses to learned correlation or a correlation-only ablation improves over the full mechanism, preserve `KILL_ARCHIVE`.
   - Do not claim ICLR-main readiness without robot or recognized high-fidelity benchmark evidence.

6. Update the paper and child documentation.
   - Make `README.md`, `child_status.md`, `plan.md`, audit docs, attack log, readiness decision, hostile reviewer response, and version log match the rerun.
   - Add a terminal audit document with exact row counts, seed coverage, metric conclusions, and PDF hash.

7. Build and verify the PDF.
   - Build `paper/main.pdf` with LaTeX.
   - Copy only the numbered PDF to `C:/Users/wangz/Downloads/84.pdf`.
   - Do not copy any PDF to the visible Desktop.
   - Scan logs for LaTeX/BibTeX warnings that affect submission quality.

8. Update root reports.
   - Update `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, and `MASTER_SUBMISSION_REPORT.md`.
   - Mark Paper 84 with the final terminal decision, commit hash, PDF hash, GitHub URL, and concise evidence.

9. Commit, push, and verify.
   - Commit only Paper 84 files inside its child repo.
   - Push `main` to the public GitHub repo.
   - Verify local `HEAD` equals `origin/main`.
   - Verify `C:/Users/wangz/Downloads/84.pdf` exists and `C:/Users/wangz/Desktop/84.pdf` does not.

## Expected Outcome Risk

The likely outcome is `KILL_ARCHIVE`, because the prior v4 evidence reports a negative mask-F1 gap against the learned correlation classifier and a core ablation that improves over the full structured mechanism. The rerun will still be performed end-to-end; the decision will be evidence-bound, not assumed.
