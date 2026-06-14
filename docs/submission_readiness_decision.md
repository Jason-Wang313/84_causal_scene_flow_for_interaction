# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: The v4 rebuild adds a paper-specific local interaction-flow benchmark, but the result is negative. The causal scene-flow model loses to a learned correlation classifier on mask F1, and the correlation-only ablation improves over the full structured mechanism. The paper also still lacks real-robot or high-fidelity simulator validation, intervention-labeled training data, and manual full-paper related-work depth.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: rebuild as a real empirical robotics paper with robot or accepted high-fidelity interaction-flow data, a trained causal scene-flow model, modern scene-flow/world-model baselines, manual related work, and decisive paired gains in attribution and downstream safety.
