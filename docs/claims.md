# Claims

- Mechanism claim tested: causal scene flow should separate robot-action-caused point motion from passive, exogenous, and occlusion-driven motion.
- Evidence claim: the v4 local benchmark tests point-level causal masks, action-effect prediction, downstream target success, ablations, and stress sweeps.
- Negative result: the full causal scene-flow model loses to a learned correlation classifier on mask F1, and a correlation-only ablation improves over the full mechanism.
- Scope claim: results support archive-quality negative evidence, not real-robot deployment.
- Unsupported claim explicitly avoided: no claim of SOTA robot performance or ICLR-main readiness.
