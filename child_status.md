# Child Status 84

Current stage: ICLR main v5 expanded evidence audit terminal
Last update: 2026-06-21
PDF: C:/Users/wangz/Downloads/84.pdf
GitHub: https://github.com/Jason-Wang313/84_causal_scene_flow_for_interaction
Submission-hardening version: v5 expanded
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

Reason: the 2026-06-21 frozen v5 rebuild regenerated 74,880 main rollouts, 5,760 scene records, 16,000 ablation rollouts, 94,080 stress rollouts, 30,720 fixed-risk rollouts, and 24 negative cases. The proposed `causal_scene_flow_model_v5` loses the hard-regime aggregate to `contrastive_action_flow` on mask F1 (`0.81699` vs `0.83774`) and causal utility (`0.64823` vs `0.67778`). Paired lower bounds are negative, the safety sum is worse, ablation necessity fails, and fixed-risk coverage at budget `0.05` is zero for all non-oracle methods. The paper remains a local synthetic diagnostic and lacks robot hardware or accepted high-fidelity benchmark validation.
