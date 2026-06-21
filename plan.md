# Plan

Paper 84 `causal_scene_flow_for_interaction` completed the 2026-06-21 v5 expanded ICLR-main submission-readiness audit.

Frozen execution plan used:

1. Write the paper-specific v5 plan before editing or running new evidence.
2. Rebuild the deterministic interaction-flow benchmark with stronger baselines and hard aggregate tests.
3. Add theory for residual separation, dominance under intervention features, and non-identifiability without interventions.
4. Run the full CPU-only protocol: main evaluation, ablations, stress sweeps, fixed-risk deployment, and negative cases.
5. Generate a 25+ page ICLR-style manuscript with bright clickable citation boxes.
6. Validate the numbered PDF in Downloads only and ensure no visible Desktop copy exists.
7. Preserve `KILL_ARCHIVE` unless every predefined evidence gate passes.

Outcome: `KILL_ARCHIVE`. The strongest non-oracle baseline beats v5 on the hard aggregate, paired lower bounds are negative, safety and ablation gates fail, and strict fixed-risk deployment has zero non-oracle coverage.
