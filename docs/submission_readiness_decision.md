# Submission Readiness Decision

Decision: KILL_ARCHIVE

Last update: 2026-06-15 09:17:18 +01:00

ICLR main-conference readiness: NO.

Reason: The 2026-06-15 v4 rerun confirms the negative result. The causal scene-flow model loses to the learned correlation classifier on mask F1 (`-0.01144 +/- 0.01838` paired difference), its target-success gain is non-decisive (`0.01701 +/- 0.08624`), and the correlation-only ablation improves over the full structured mechanism (`0.78781` versus `0.76087` mask F1). The paper also still lacks real-robot or high-fidelity simulator validation, intervention-labeled training data, and manual full-paper related-work depth.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: rebuild as a real empirical robotics paper with robot or accepted high-fidelity interaction-flow data, a trained causal scene-flow model, modern scene-flow/world-model baselines, manual related work, and decisive paired gains in attribution and downstream safety.
