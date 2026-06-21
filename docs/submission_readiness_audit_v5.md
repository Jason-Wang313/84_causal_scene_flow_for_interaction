# Paper 84 Submission Readiness Audit v5

Last update: 2026-06-21

## Terminal Decision

KILL_ARCHIVE for ICLR main.

## Evidence Added In v5

- Frozen plan written before execution: `docs/paper84_expanded_submission_plan_20260621.md`.
- Main evaluation: 74,880 rollouts across nine splits, 13 methods, and 10 seeds.
- Dataset summary: 5,760 generated 128-point RGB-D interaction-flow scenes.
- Ablation evaluation: 16,000 rollouts across `combined_hard_shift` and `distractor_contact_shift`.
- Stress evaluation: 94,080 rollouts over passive flow, occlusion, articulation delay, distractor contact, ego motion, and combined stress.
- Fixed-risk evaluation: 30,720 rollouts across two hard splits, four risk budgets, six methods, and 10 seeds.
- Negative cases: 24 retained cases.
- Manuscript: 39-page ICLR-style PDF with bright clickable citation boxes.

## Main Hard-Regime Evidence

- `causal_scene_flow_model_v5`: mask F1 `0.81699 +/- 0.00175`; causal utility `0.64823`; target success `0.70918`; passive false `0.22730`; unsafe `0.07695`.
- Strongest non-oracle baseline `contrastive_action_flow`: mask F1 `0.83774 +/- 0.00186`; causal utility `0.67778`; target success `0.73555`; passive false `0.15339`; unsafe `0.08730`.
- Paired mask-F1 lower 95 percent bound versus `contrastive_action_flow`: `-0.02228`.
- Paired causal-utility lower 95 percent bound versus `contrastive_action_flow`: `-0.03869`.

## Gate Outcomes

- Margin gate: false.
- Paired gate: false.
- Safety gate: false.
- Ablation gate: false.
- Fixed-risk gate: false.
- Stress gate: true.

The single stress-gate pass is not enough to rescue the paper because all core main-readiness gates fail.

## Submission Readiness Blockers

- The proposed method loses to a strong contrastive action-flow baseline on the hard aggregate.
- Paired seed-level lower bounds are negative for the primary attribution and utility metrics.
- The proposed method's passive-false-plus-unsafe safety sum is worse than the best safety reference.
- Ablations do not consistently support the full mechanism.
- Strict fixed-risk deployment at budget `0.05` has zero non-oracle coverage.
- Evidence is still local synthetic evidence only, with no real robot or accepted high-fidelity benchmark validation.

## Validator Result

`python scripts\validate_submission_artifacts.py` passed.

- Downloads PDF: `C:/Users/wangz/Downloads/84.pdf`
- Pages: 39
- SHA256: `03414EF5F41537EF71C9159B20370D3D859AB525C98369C47E2392959362EAF9`
- Desktop copy: absent.
