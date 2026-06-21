# Claims

- Mechanism claim tested: causal scene flow should separate robot-action-caused point motion from passive, exogenous, ego-motion, distractor-contact, articulation, occlusion, transparent-depth, and hidden-slip motion.
- Evidence claim: the v5 local benchmark tests point-level causal masks, action-effect prediction, downstream target success, hard aggregate metrics, paired seed statistics, ablations, stress sweeps, fixed-risk deployment, and negative cases.
- Negative result: `causal_scene_flow_model_v5` loses to `contrastive_action_flow` on hard-regime mask F1 and causal utility, has negative paired lower bounds, fails safety and ablation gates, and has zero non-oracle coverage at fixed-risk budget `0.05`.
- Scope claim: results support archive-quality negative evidence, not real-robot deployment.
- Unsupported claim explicitly avoided: no claim of SOTA robot performance or ICLR-main readiness.
