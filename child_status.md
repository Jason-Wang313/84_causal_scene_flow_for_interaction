# Child Status 84

Current stage: ICLR main v4 evidence audit terminal
Last update: 2026-06-14 10:56:09 +01:00
PDF: C:/Users/wangz/Downloads/84.pdf
GitHub: https://github.com/Jason-Wang313/84_causal_scene_flow_for_interaction
Submission-hardening version: v4
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

Reason: the repo now contains a deterministic local interaction-flow benchmark, but the causal scene-flow model loses to a learned correlation classifier on mask F1 and the correlation-only ablation improves over the full structured mechanism. No robot hardware or high-fidelity simulator validation is available.
