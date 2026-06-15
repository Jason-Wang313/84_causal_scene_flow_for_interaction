# Child Status 84

Current stage: ICLR main v4 evidence audit terminal
Last update: 2026-06-15 09:17:18 +01:00
PDF: C:/Users/wangz/Downloads/84.pdf
GitHub: https://github.com/Jason-Wang313/84_causal_scene_flow_for_interaction
Submission-hardening version: v4
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

Reason: the 2026-06-15 full rerun regenerated 11,760 main rollouts, 2,058 ablation rollouts, and 25,200 stress rollouts. The causal scene-flow model still loses to the learned correlation classifier on combined hard-shift mask F1 (`-0.01144 +/- 0.01838` paired difference), the downstream target-success gain is non-decisive (`0.01701 +/- 0.08624`), and the correlation-only ablation improves over the full structured mechanism. No robot hardware or high-fidelity simulator validation is available.
