# Experiment Rigor Checklist

## v2 Synthetic Rigor
- [x] Multiple seeds.
- [x] Error bars.
- [x] Stronger synthetic baselines.
- [x] Ablations.
- [x] Stress tests.
- [x] Negative cases.

## v4 Local Interaction-Flow Rigor
- [x] Paper-specific point-level scene-flow benchmark.
- [x] Five physical-shift splits.
- [x] Eight methods including learned correlation and oracle upper bound.
- [x] Seed-level paired comparisons.
- [x] Ablations for action conditioning, passive-flow factoring, articulation, occlusion, and effect prediction.
- [x] Stress sweeps for passive flow, occlusion, articulation delay, distractor contact, and combined stress.
- [x] Negative cases documented.

## ICLR Main Bar
- [ ] Real-robot validation.
- [ ] High-fidelity simulator benchmark.
- [ ] Implemented learned model.
- [ ] Implemented real competing baselines.
- [ ] Manual related-work synthesis.
- [ ] Paper-specific qualitative figures.

Decision: fail ICLR main empirical-rigor gate because the v4 result is negative and still local-only; archive.
