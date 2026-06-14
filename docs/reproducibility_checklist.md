# Reproducibility Checklist

## What Reproduces
- [x] `python src/run_experiment.py`
- [x] `results/metrics.csv`
- [x] `results/raw_seed_metrics.csv`
- [x] `results/ablation_metrics.csv`
- [x] `results/stress_sweep.csv`
- [x] `results/negative_cases.csv`
- [x] `results/rollouts.csv`
- [x] `results/pairwise_stats.csv`
- [x] `results/ablation_rollouts.csv`
- [x] `results/stress_sweep_raw.csv`
- [x] `paper/main.tex`
- [x] Canonical PDF: `C:/Users/wangz/Downloads/84.pdf`

## What Does Not Reproduce
- [ ] Real robot results.
- [ ] High-fidelity benchmark runs.
- [ ] Trained intervention-labeled causal scene-flow checkpoints.
- [ ] External modern scene-flow and manipulation-world-model baselines.

This is reproducible as a local negative evidence audit, not as an ICLR-main robotics system paper.
