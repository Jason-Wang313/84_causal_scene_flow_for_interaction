# 84 Causal Scene Flow for Interaction

Submission-hardening version: v4

Terminal decision: **KILL_ARCHIVE** for ICLR main conference.

This repository contains a reproducible local evidence audit for the research bet:

> Infer which scene-flow components are caused by robot action versus passive motion.

The v4 rebuild replaces the template scaffold with a deterministic local 3D interaction-flow benchmark covering action-caused flow, passive conveyor/background flow, distractor contact, articulated coupling, occlusion, and ego-motion noise.

## Why This Is Archived

- On the combined hard-shift split, `causal_scene_flow_model` reaches `0.76087 +/- 0.01126` mask F1.
- The strongest non-oracle baseline, `learned_correlation_classifier`, reaches `0.77230 +/- 0.01102` mask F1.
- The paired mask-F1 difference versus the learned baseline is `-0.01144 +/- 0.01838`.
- The `correlation_only_causal_head` ablation improves mask F1 to `0.78781 +/- 0.01137`, contradicting the structured causal-decomposition mechanism.
- The evidence is local and synthetic, not hardware or accepted high-fidelity benchmark validation.

## Reproduce

```powershell
python src\run_experiment.py
```

The runner writes:

- `results/rollouts.csv`
- `results/raw_seed_metrics.csv`
- `results/metrics.csv`
- `results/pairwise_stats.csv`
- `results/ablation_metrics.csv`
- `results/stress_sweep.csv`
- `results/negative_cases.csv`
- `results/summary.txt`
- `figures/causal_flow_*.png`

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/84.pdf`
