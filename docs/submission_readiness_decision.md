# Submission Readiness Decision

Decision: KILL_ARCHIVE

Last update: 2026-06-21

ICLR main-conference readiness: NO.

Reason: The 2026-06-21 v5 expanded rebuild confirms the negative result under a much stronger protocol. The proposed `causal_scene_flow_model_v5` loses to `contrastive_action_flow` on the hard-regime aggregate mask-F1 gate (`0.81699` versus `0.83774`) and causal-utility gate (`0.64823` versus `0.67778`). Paired lower bounds are negative, safety is worse than the best reference, ablation necessity fails, and fixed-risk coverage at budget `0.05` is zero for all non-oracle methods.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: rebuild as a real empirical robotics paper with robot or accepted high-fidelity RGB-D interaction-flow data, a trained causal scene-flow model, modern scene-flow/world-model/action-conditioned perception baselines, manual related work, and decisive paired gains in attribution, downstream success, safety, and fixed-risk coverage.
