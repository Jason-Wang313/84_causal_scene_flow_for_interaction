# 84 Causal Scene Flow for Interaction

Submission-hardening version: v5 expanded submission-readiness audit

Terminal decision: **KILL_ARCHIVE** for ICLR main conference.

Latest audit rerun: 2026-06-21.

This repository contains a reproducible local evidence audit for the research bet:

> Infer which scene-flow components are caused by robot action versus passive motion.

The v5 rebuild expands the old five-split v4 benchmark into a frozen 25+ page ICLR-style negative audit with theory, stronger baselines, hard aggregate tests, two-split ablations, stress sweeps, fixed-risk deployment, and retained negative cases.

## Why This Is Archived

- The frozen v5 protocol regenerated 74,880 main rollouts, 5,760 scene records, 16,000 ablation rollouts, 94,080 stress rollouts, 30,720 fixed-risk rollouts, and 24 negative cases.
- On the hard-regime aggregate, `causal_scene_flow_model_v5` reaches `0.81699 +/- 0.00175` mask F1 and `0.64823 +/- 0.014` causal utility.
- The strongest non-oracle baseline, `contrastive_action_flow`, reaches `0.83774 +/- 0.00186` mask F1 and `0.67778` causal utility.
- The paired lower 95 percent bounds are negative: `-0.02228` for mask F1 and `-0.03869` for causal utility.
- The safety sum is worse than the safest strong reference: v5 passive false plus unsafe is `0.30425`, while `contrastive_action_flow` is `0.24069`.
- Ablation necessity fails: simpler or older variants beat the full method on key hard-split metrics.
- Fixed-risk deployment at budget `0.05` has zero non-oracle coverage on both fixed-risk splits.
- The evidence remains local synthetic evidence, not robot hardware, accepted high-fidelity simulation, or external benchmark validation.

## Reproduce

```powershell
python src\run_experiment.py
python scripts\generate_manuscript.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item main.pdf C:\Users\wangz\Downloads\84.pdf -Force
cd ..
python scripts\validate_submission_artifacts.py
```

The validator checks row counts, clean citation/reference resolution, bright clickable citation-box settings, page count, Downloads-only placement, and Desktop absence.

Canonical local PDF: `C:/Users/wangz/Downloads/84.pdf`

Public GitHub repo: https://github.com/Jason-Wang313/84_causal_scene_flow_for_interaction
