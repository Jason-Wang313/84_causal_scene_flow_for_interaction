import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
DOCS = ROOT / "docs"


METHOD_LABELS = {
    "flow_magnitude_threshold": "Magnitude",
    "action_direction_projection": "Action projection",
    "rigid_scene_flow_cluster": "Rigid cluster",
    "temporal_difference_mask": "Temporal difference",
    "noncausal_flow_transformer_proxy": "Noncausal proxy",
    "learned_correlation_classifier": "Learned correlation",
    "calibrated_tree_flow_proxy": "Calibrated tree",
    "contrastive_action_flow": "Contrastive action",
    "scene_flow_transformer_proxy": "Scene-flow transformer",
    "world_model_flow_predictor": "World model",
    "causal_scene_flow_model_v4": "Causal flow v4",
    "causal_scene_flow_model_v5": "Causal flow v5",
    "oracle_causal_mask": "Oracle",
}

PLOT_METHODS = list(METHOD_LABELS.keys())
FOCUS_METHODS = [
    "learned_correlation_classifier",
    "calibrated_tree_flow_proxy",
    "contrastive_action_flow",
    "world_model_flow_predictor",
    "causal_scene_flow_model_v4",
    "causal_scene_flow_model_v5",
    "oracle_causal_mask",
]


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ascii_clean(text):
    text = str(text or "")
    text = text.replace("–", "-").replace("—", "-").replace("“", '"').replace("”", '"').replace("’", "'")
    return text.encode("ascii", "ignore").decode("ascii")


def tex_escape(text):
    text = ascii_clean(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def metric_lookup(rows, split, method, metric):
    for row in rows:
        if row.get("split") == split and row.get("method") == method and row.get("metric") == metric:
            return float(row["mean"]), float(row["ci95"])
    raise KeyError((split, method, metric))


def fmt_pm(mean, ci):
    return f"{mean:.3f} +/- {ci:.3f}"


def count_rows(name):
    return len(read_csv(RESULTS / name))


def parse_gates(summary_text):
    gates = {}
    active = False
    for line in summary_text.splitlines():
        if line.strip() == "Decision gates:":
            active = True
            continue
        if active and not line.strip():
            break
        if active and ":" in line:
            key, value = line.split(":", 1)
            gates[key.strip()] = value.strip()
    return gates


def bib_key(i):
    return f"pool84_{i:02d}"


def write_references():
    prior_rows = read_csv(DOCS / "deep_read_250.csv")[:42]
    entries = []
    for i, row in enumerate(prior_rows, start=1):
        title = tex_escape(row.get("title") or "Untitled prior work")
        authors_raw = ascii_clean(row.get("authors") or "Local Prior Work Pool")
        authors = " and ".join([tex_escape(part.strip()) for part in re.split(r";", authors_raw) if part.strip()])
        if not authors:
            authors = "Local Prior Work Pool"
        year_raw = ascii_clean(row.get("year") or "")
        match = re.search(r"(19|20)\d{2}", year_raw)
        year = match.group(0) if match else "2026"
        venue = tex_escape(row.get("venue") or row.get("source") or "prior-work pool")
        link = tex_escape(row.get("doi") or row.get("url") or row.get("arxiv_id") or row.get("uid") or "local pool record")
        entries.append(
            "\n".join(
                [
                    f"@misc{{{bib_key(i)},",
                    f"  author={{{authors}}},",
                    f"  title={{{title}}},",
                    f"  year={{{year}}},",
                    f"  note={{{venue}; {link}}}",
                    "}",
                ]
            )
        )
    (PAPER / "references.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return [bib_key(i) for i in range(1, len(prior_rows) + 1)], prior_rows


def longtable(header, rows, spec, caption, label, fontsize=r"\scriptsize"):
    out = [
        r"\begin{center}",
        fontsize,
        f"\\begin{{longtable}}{{{spec}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        r"\toprule",
        header + r"\\",
        r"\midrule",
        r"\endfirsthead",
        f"\\caption[]{{{caption} (continued)}}\\\\",
        r"\toprule",
        header + r"\\",
        r"\midrule",
        r"\endhead",
    ]
    out.extend(rows)
    out.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize", r"\end{center}"])
    return "\n".join(out)


def method_name(method):
    return tex_escape(METHOD_LABELS.get(method, method))


def main():
    PAPER.mkdir(exist_ok=True)
    cite_keys, prior_rows = write_references()
    metrics = read_csv(RESULTS / "metrics.csv")
    hard_metrics = read_csv(RESULTS / "hard_aggregate_metrics.csv")
    hard_pairs = read_csv(RESULTS / "hard_aggregate_pairwise_stats.csv")
    ablations = read_csv(RESULTS / "ablation_metrics.csv")
    ablation_seed = read_csv(RESULTS / "ablation_seed_metrics.csv")
    stress = read_csv(RESULTS / "stress_sweep.csv")
    stress_seed = read_csv(RESULTS / "stress_sweep_seed_metrics.csv")
    fixed = read_csv(RESULTS / "fixed_risk_metrics.csv")
    fixed_seed = read_csv(RESULTS / "fixed_risk_seed_metrics.csv")
    fixed_pairs = read_csv(RESULTS / "fixed_risk_pairwise.csv")
    negative = read_csv(RESULTS / "negative_cases.csv")
    summary_text = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    gates = parse_gates(summary_text)

    proposal = "causal_scene_flow_model_v5"
    oracle = "oracle_causal_mask"
    best_ref = gates.get("best_f1_reference", "contrastive_action_flow")
    best_util_ref = gates.get("best_utility_reference", best_ref)
    prop_f1 = metric_lookup(hard_metrics, "hard_regime_aggregate", proposal, "mask_f1")
    best_f1 = metric_lookup(hard_metrics, "hard_regime_aggregate", best_ref, "mask_f1")
    prop_utility = metric_lookup(hard_metrics, "hard_regime_aggregate", proposal, "causal_utility")
    best_utility = metric_lookup(hard_metrics, "hard_regime_aggregate", best_util_ref, "causal_utility")

    lines = []
    lines.extend(
        [
            r"\documentclass{article}",
            r"\usepackage{iclr2026_conference,times}",
            r"\input{math_commands.tex}",
            r"\usepackage{hyperref}",
            r"\usepackage{url}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage{array}",
            r"\usepackage{longtable}",
            r"\usepackage{xcolor}",
            r"\usepackage{amsmath,amssymb}",
            r"\hypersetup{colorlinks=false,pdfborder={0 0 1.6},citebordercolor={0 1 0},linkbordercolor={1 0.55 0},urlbordercolor={0 0.45 1}}",
            r"\graphicspath{{../figures/}}",
            r"\newcommand{\methodname}{causal scene flow v5}",
            r"\title{Causal Scene Flow for Interaction:\\An Expanded Negative Submission-Readiness Audit}",
            r"\author{Anonymous Authors}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            (
                "Interaction requires deciding which observed 3D scene-flow components were caused by a robot action rather than by passive conveyors, ego motion, distractors, articulation, occlusion, transparent-depth failures, or hidden tool slip. "
                "This paper rebuilds a small archived seed into a hostile-review audit of causal scene flow for robot interaction. "
                f"The frozen v5 protocol contains {count_rows('rollouts.csv'):,} main rollouts, {count_rows('dataset_summary.csv'):,} generated scene records, {count_rows('ablation_rollouts.csv'):,} ablation rollouts, "
                f"{count_rows('stress_sweep_raw.csv'):,} stress rollouts, {count_rows('fixed_risk_raw.csv'):,} fixed-risk rollouts, and {count_rows('negative_cases.csv')} retained negative cases. "
                f"On the predefined hard-regime aggregate, \\methodname{{}} reaches {fmt_pm(*prop_f1)} mask F1 and {fmt_pm(*prop_utility)} causal utility. "
                f"The strongest non-oracle baseline, {method_name(best_ref)}, reaches {fmt_pm(*best_f1)} mask F1 and {fmt_pm(*best_utility)} causal utility. "
                "The paired lower confidence bounds are negative, the safety sum is worse, ablation necessity fails, and strict fixed-risk coverage collapses. "
                "The honest terminal decision is therefore \\textbf{KILL/ARCHIVE}, not ICLR-main submission."
            ),
            r"\end{abstract}",
        ]
    )

    lines.extend(
        [
            r"\section{Terminal Decision}",
            (
                "\\textbf{Decision: KILL/ARCHIVE for ICLR main.} "
                "The rebuild improves the archive in scale, baselines, theory, stress testing, and reporting, but it does not produce a defensible positive submission. "
                f"The proposal trails {method_name(best_ref)} by {float(gates.get('hard_mask_f1_margin', '0')):.3f} hard-aggregate mask F1 and trails {method_name(best_util_ref)} by {float(gates.get('hard_causal_utility_margin', '0')):.3f} causal utility. "
                f"The paired lower bounds are {tex_escape(gates.get('paired_mask_f1_lower95', 'unknown'))} for mask F1 and {tex_escape(gates.get('paired_causal_utility_lower95', 'unknown'))} for causal utility."
            ),
            (
                "This is not a formatting failure. "
                "It is a mechanism failure under a stronger protocol. "
                "The v5 model is useful as a diagnostic because it reduces some unsafe contacts relative to weaker methods, but it does not beat the strongest non-oracle alternatives and it cannot certify fixed-risk deployment."
            ),
            (
                "The paper is still worth preserving because negative evidence is informative: it says that interventional residualization alone is not enough when contrastive action features and world-model-like flow prediction already explain most local synthetic variation."
            ),
            r"\section{Problem And Prior-Work Pressure}",
            (
                "A robot interacting with a scene observes point motion after an action. "
                "Only part of that motion is action-caused. "
                "The rest may come from a passive conveyor, camera ego motion, another moving object, articulated coupling, depth dropout, or hidden tool slip. "
                "A manipulation policy that grasps the largest moving cluster or the strongest action-aligned flow can therefore select a passive object and make an unsafe contact."
            ),
            (
                "The local prior-work pool places this problem near scene-motion teaching, contact-aware onboard perception, optical/scene flow, flow parsing, ego-motion learning, RGB-D world models, and flow-based manipulation policies "
                f"\\citep{{{cite_keys[0]},{cite_keys[1]},{cite_keys[2]},{cite_keys[5]},{cite_keys[6]},{cite_keys[10]},{cite_keys[11]}}}. "
                "That proximity creates a hostile novelty boundary: a positive paper must show that causal source separation is more than a relabeled action-conditioned flow baseline."
            ),
            (
                "The v4 archive failed this boundary. "
                "It tested a small deterministic benchmark and found that the full structured causal model lost mask F1 to a learned correlation classifier, while a correlation-only ablation improved over the full method. "
                "The v5 rebuild therefore starts from a negative burden of proof rather than from optimism."
            ),
            r"\section{Formal Setup}",
            (
                "Let $p_i\\in\\mathbb{R}^3$ be a point in an RGB-D scene, $a\\in\\mathbb{R}^3$ a robot action direction, and $y_i\\in\\{0,1\\}$ the latent indicator that point $i$ moved because of the robot action. "
                "The observed flow is decomposed as"
            ),
            "\n".join(
                [
                    r"\begin{equation}",
                    r"f_i^{obs} = y_i f_i^{act}(a) + f_i^{passive} + f_i^{ego} + f_i^{dist} + \epsilon_i.",
                    r"\end{equation}",
                ]
            ),
            (
                "The method predicts scores $s_i\\in[0,1]$ and a mask $\\hat y_i=\\mathbb{1}[s_i\\geq 0.5]$. "
                "Downstream interaction succeeds when the selected action-caused effect is accurate enough and does not attribute passive or distractor flow to the robot."
            ),
            (
                "\\textbf{Metric definitions.} "
                "Mask F1 evaluates point-level causal attribution. "
                "Passive false attribution is the fraction of passive or distractor points selected as causal. "
                "Effect error is normalized error in the mean action-caused flow. "
                "Target success is the downstream binary interaction outcome. "
                "Unsafe contact is a binary failure caused by false causal selection. "
                "Causal utility combines target success, mask F1, intervention consistency, unsafe contact, passive false attribution, and effect error."
            ),
            (
                "Fixed-risk deployment uses an upper risk estimate $\\hat r$ and executes only when $\\hat r\\leq \\rho$. "
                "Coverage is the executed fraction, fixed-risk success is success over all requested episodes including abstentions, and false-safe rate is unsafe contact among executed episodes."
            ),
            r"\section{Theory: What Would Make Causal Flow Identifiable?}",
            (
                "\\textbf{Residual-separation lemma.} "
                "Assume $\\|\\hat f_i^{passive}-f_i^{passive}\\|\\leq \\epsilon$ for all non-action points, and action-caused points satisfy $\\langle f_i^{act},a\\rangle-\\langle f_i^{passive},a\\rangle\\geq \\gamma$. "
                "If $\\epsilon<\\gamma/2$, thresholding an action-conditioned residual score separates passive from action-caused flow up to the residual estimator's error set."
            ),
            (
                "The lemma explains why residualization is plausible: when passive flow can be estimated and the action margin is real, a causal mask can reduce false passive attribution. "
                "It also exposes the weak point: transparent depth, hidden slip, and action-aligned distractors directly attack the residual estimator."
            ),
            (
                "\\textbf{Dominance proposition.} "
                "When passive flow is action-aligned but predictable from intervention features, a residual scorer with negative-action consistency can dominate a magnitude scorer and a simple correlation scorer on passive false attribution. "
                "The proposition is conditional on separability; it is not a universal guarantee."
            ),
            (
                "\\textbf{Non-identifiability theorem.} "
                "If intervention labels and negative actions are absent, and passive flow shares the same action-conditioned sufficient statistics as action-caused flow, then the causal mask is not identifiable from observed flow alone. "
                "In that regime a correlation classifier, contrastive action feature, or world-model predictor can match or beat a structured causal decomposition."
            ),
            (
                "The empirical result below lands in the theorem's warning zone. "
                "The v5 residual model is more structured than v4, yet contrastive action flow and world-model flow remain stronger on the frozen hard aggregate."
            ),
            r"\section{Method Under Test}",
            (
                "\\methodname{} estimates an action-conditioned residual score. "
                "It subtracts a passive/ego-flow template, adds contact-graph and articulation-lag priors, imputes occluded causal points when contact and graph evidence agree, penalizes passive-like flow, and uses an uncertainty term for depth dropout and weak alignment. "
                "A counterfactual feature compares positive action projection with negative-action consistency."
            ),
            (
                "The key comparison set is deliberately uncomfortable: a learned correlation classifier, a calibrated tree proxy, a contrastive action-flow baseline, a scene-flow-transformer proxy, a world-model flow predictor, the old v4 causal score, and an oracle causal mask. "
                "These are not neural training runs; they are deterministic CPU-light analytic proxies designed to test whether the mechanism has signal after obvious strong alternatives are included."
            ),
            r"\section{Frozen Experimental Protocol}",
            (
                "The protocol was frozen before execution in \\texttt{docs/paper84\\_expanded\\_submission\\_plan\\_20260621.md}. "
                "Main evaluation uses 10 seeds, 128 points per generated scene, 64 episodes per split and seed, nine splits, and 13 methods. "
                "The hard-regime aggregate excludes only the clean-contact split."
            ),
            (
                "The nine splits are clean contact, passive conveyor, articulated coupling, occluded interaction, ego-motion shift, transparent-depth noise, tool-slip hidden contact, distractor-contact shift, and combined hard shift. "
                "Stress evaluation sweeps passive flow, occlusion, articulation delay, distractor contact, ego motion, and combined stress over seven levels. "
                "Fixed-risk evaluation tests budgets 0.02, 0.05, 0.10, and 0.20 on the two hardest deployment splits."
            ),
            (
                "The terminal gate requires at least 0.03 hard-aggregate margin in both mask F1 and causal utility over the strongest non-oracle baseline, positive paired lower confidence bounds, no safety regression beyond 0.01, ablation necessity, fixed-risk coverage of at least 0.25 at budget 0.05 with false-safe rate at most 0.05, and no domination at maximum combined stress."
            ),
        ]
    )

    hard_rows = []
    for method in PLOT_METHODS:
        f1 = metric_lookup(hard_metrics, "hard_regime_aggregate", method, "mask_f1")
        success = metric_lookup(hard_metrics, "hard_regime_aggregate", method, "target_success")
        utility = metric_lookup(hard_metrics, "hard_regime_aggregate", method, "causal_utility")
        passive = metric_lookup(hard_metrics, "hard_regime_aggregate", method, "passive_false_attribution")
        unsafe = metric_lookup(hard_metrics, "hard_regime_aggregate", method, "unsafe_contact")
        hard_rows.append(
            f"{method_name(method)} & {fmt_pm(*f1)} & {success[0]:.3f} & {utility[0]:.3f} & {passive[0]:.3f} & {unsafe[0]:.3f}\\\\"
        )
    lines.extend(
        [
            r"\section{Main Hard-Regime Results}",
            (
                "Table~\\ref{tab:hard-main} is the central result. "
                f"\\methodname{{}} is competitive with the old causal model, but it loses to {method_name(best_ref)} and the world-model baseline. "
                "The oracle gap remains large, showing that the synthetic task is not saturated."
            ),
            longtable(
                r"Method & Mask F1 & Target success & Causal utility & Passive false & Unsafe",
                hard_rows,
                r"p{0.28\linewidth}ccccc",
                "Predefined hard-regime aggregate over eight non-clean splits.",
                "tab:hard-main",
            ),
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.98\linewidth]{causal_flow_hard_f1_v5.png}",
            r"\caption{Hard-regime mask F1. The v5 causal method does not beat the strongest non-oracle baseline.}",
            r"\label{fig:hard-f1}",
            r"\end{figure}",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.98\linewidth]{causal_flow_downstream_v5.png}",
            r"\caption{Target success and causal utility on the hard aggregate. The downstream result follows the attribution failure.}",
            r"\label{fig:downstream}",
            r"\end{figure}",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.98\linewidth]{causal_flow_safety_v5.png}",
            r"\caption{Passive false attribution and unsafe contact. v5 is safer than some weak baselines but worse than the strongest contrastive reference in the combined safety sum.}",
            r"\label{fig:safety}",
            r"\end{figure}",
        ]
    )

    pair_rows = []
    for row in hard_pairs:
        if row["reference"] in {best_ref, best_util_ref}:
            pair_rows.append(
                f"{method_name(row['reference'])} & {tex_escape(row['metric'])} & {row['mean_diff']} & {row['ci95']} & {row['lower95']} & {row['target_better_seeds']}/{row['seeds']}\\\\"
            )
    lines.extend(
        [
            r"\section{Paired Seed-Level Test}",
            (
                "The paired comparison is more damaging than the aggregate mean. "
                "Against the strongest reference, v5 has negative lower confidence bounds for mask F1 and causal utility and wins zero of 10 seeds on those primary comparisons. "
                "That is not a close positive result; it is a failed gate."
            ),
            longtable(
                r"Reference & Metric & Mean diff & CI95 & Lower95 & Better seeds",
                pair_rows,
                r"p{0.24\linewidth}p{0.23\linewidth}cccc",
                "Hard-regime paired seed differences for v5 minus the strongest non-oracle references.",
                "tab:paired",
            ),
        ]
    )

    ablation_rows = []
    for row in ablations:
        ablation_rows.append(
            f"{tex_escape(row['split'])} & {tex_escape(row['ablation'])} & {row['mask_f1']} & {row['target_success']} & {row['passive_false_attribution']} & {row['effect_error']} & {row['unsafe_contact']} & {row['causal_utility']}\\\\"
        )
    lines.extend(
        [
            r"\section{Ablations}",
            (
                "Ablation necessity fails. "
                "On combined hard shift, the old v4 causal score beats v5 on mask F1, and on distractor-contact shift the old v4 score also beats v5 on mask F1. "
                "Removing the articulation graph improves causal utility on distractor-contact shift. "
                "This contradicts the claim that the full decomposition is necessary in the current benchmark."
            ),
            longtable(
                r"Split & Ablation & F1 & Success & Passive false & Effect err. & Unsafe & Utility",
                ablation_rows,
                r"p{0.17\linewidth}p{0.27\linewidth}cccccc",
                "Ablation results on the two predefined hard splits.",
                "tab:ablations",
            ),
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.98\linewidth]{causal_flow_ablation_v5.png}",
            r"\caption{Combined hard-shift ablations. Some removals are weaker, but the full mechanism is not consistently necessary across the frozen ablation suite.}",
            r"\label{fig:ablation}",
            r"\end{figure}",
        ]
    )

    stress_rows = []
    for row in [item for item in stress if item["stress_axis"] == "combined"]:
        stress_rows.append(
            f"{row['stress_level']} & {method_name(row['method'])} & {row['mask_f1']} & {row['target_success']} & {row['unsafe_contact']} & {row['risk_upper']} & {row['causal_utility']}\\\\"
        )
    lines.extend(
        [
            r"\section{Stress Tests}",
            (
                "Stress testing gives the only partially favorable gate: at maximum combined stress, v5 has slightly higher causal utility than the best non-oracle reference because it trades success for lower unsafe contacts. "
                "That isolated advantage does not overcome the main hard aggregate, paired test, safety aggregate, ablation, or fixed-risk failures."
            ),
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.98\linewidth]{causal_flow_stress_sweep_v5.png}",
            r"\caption{Combined stress sweep. v5 is not uniformly dominated under maximum stress, but this does not rescue the submission gate.}",
            r"\label{fig:stress}",
            r"\end{figure}",
            longtable(
                r"Level & Method & F1 & Success & Unsafe & Risk upper & Utility",
                stress_rows,
                r"cp{0.26\linewidth}ccccc",
                "Combined stress sweep over seven levels.",
                "tab:stress-combined",
            ),
        ]
    )

    fixed_rows = []
    for row in fixed:
        fixed_rows.append(
            f"{tex_escape(row['split'])} & {row['risk_budget']} & {method_name(row['method'])} & {row['coverage']} & {row['fixed_risk_success']} & {row['executed_success']} & {row['false_safe_rate']} & {row['risk_upper']}\\\\"
        )
    lines.extend(
        [
            r"\section{Fixed-Risk Deployment}",
            (
                "Fixed-risk deployment is the cleanest submission blocker. "
                "At risk budget 0.05, every non-oracle method has zero coverage on both fixed-risk splits. "
                "This is safer than reckless execution, but a method that cannot execute under a strict risk budget is not deployment-ready."
            ),
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=0.98\linewidth]{causal_flow_fixed_risk_v5.png}",
            r"\caption{Coverage and false-safe rate at budget 0.05 on combined hard shift. Non-oracle coverage collapses to zero.}",
            r"\label{fig:fixed-risk}",
            r"\end{figure}",
            longtable(
                r"Split & Budget & Method & Coverage & Fixed success & Exec. success & False-safe & Risk",
                fixed_rows,
                r"p{0.18\linewidth}cp{0.19\linewidth}ccccc",
                "Fixed-risk deployment over two splits and four risk budgets.",
                "tab:fixed-risk",
            ),
        ]
    )

    neg_rows = []
    for row in negative:
        neg_rows.append(
            f"{tex_escape(row['source'])} & {tex_escape(row['split'])} & {tex_escape(row['method'])} & {tex_escape(row['failure_label'])} & {row['mask_f1']} & {row['risk_upper']} & {tex_escape(row['lesson'])}\\\\"
        )
    lines.extend(
        [
            r"\section{Negative Cases}",
            (
                "The negative cases are part of the evidence, not an appendix dumping ground. "
                "They include v5 main failures, strong-baseline successes, ablation counterexamples, and fixed-risk abstentions. "
                "Keeping them visible prevents the paper from turning an archive-quality diagnostic into a false positive claim."
            ),
            longtable(
                r"Source & Split & Method & Outcome & F1 & Risk & Lesson",
                neg_rows,
                r"p{0.16\linewidth}p{0.17\linewidth}p{0.18\linewidth}p{0.16\linewidth}ccp{0.22\linewidth}",
                "Retained negative cases.",
                "tab:negative-cases",
            ),
            r"\section{Reviewer Attack Surface}",
            (
                "A hostile reviewer has several fair attacks. "
                "First, the strongest non-oracle baseline wins the main hard aggregate. "
                "Second, the safety reference is also the same contrastive baseline, so v5 cannot claim a clean safety trade. "
                "Third, ablations show that older or simpler components sometimes beat the full decomposition. "
                "Fourth, fixed-risk coverage is zero at the strict budget. "
                "Fifth, all evidence is local synthetic evidence rather than real RGB-D robot data or a recognized high-fidelity benchmark."
            ),
            (
                "The strongest defense is modest: the benchmark is reproducible, the protocol is frozen, the baselines are uncomfortable, and the negative result identifies exactly what a future real submission must fix. "
                "That is valuable research hygiene, but it is not acceptance-level evidence."
            ),
            r"\section{Reproducibility}",
            r"\begin{verbatim}",
            r"python src\run_experiment.py",
            r"python scripts\generate_manuscript.py",
            r"cd paper",
            r"pdflatex -interaction=nonstopmode -halt-on-error main.tex",
            r"bibtex main",
            r"pdflatex -interaction=nonstopmode -halt-on-error main.tex",
            r"pdflatex -interaction=nonstopmode -halt-on-error main.tex",
            r"Copy-Item main.pdf C:\Users\wangz\Downloads\84.pdf",
            r"python ..\scripts\validate_submission_artifacts.py",
            r"\end{verbatim}",
            (
                "The canonical numbered PDF is \\texttt{C:/Users/wangz/Downloads/84.pdf}. "
                "No numbered Paper 84 PDF should be copied to the visible Desktop."
            ),
            r"\section{Limitations}",
            (
                "The limitations are severe. "
                "The benchmark is deterministic and local. "
                "The baselines are analytic proxies, not trained neural systems. "
                "The prior-work pool is useful for pressure testing but must be manually verified before a real submission. "
                "Most importantly, no real robot RGB-D interaction data, external high-fidelity simulator, or accepted benchmark is present. "
                "Even a positive local result would therefore require external validation before an ICLR-main claim."
            ),
            r"\section{Conclusion}",
            (
                "Causal scene flow remains a good research question. "
                "This v5 rebuild shows that action-conditioned residualization can be competitive and can sometimes lower unsafe contacts under severe stress. "
                "It also shows that contrastive action features and world-model flow prediction remain stronger on the predefined hard aggregate, and that fixed-risk deployment is not usable. "
                "The honest terminal state is \\textbf{KILL/ARCHIVE}; future work needs real interaction data, trained baselines, and stronger identifiability evidence."
            ),
        ]
    )

    split_rows = []
    for split in sorted({row["split"] for row in metrics}):
        for method in PLOT_METHODS:
            f1 = metric_lookup(metrics, split, method, "mask_f1")
            success = metric_lookup(metrics, split, method, "target_success")
            passive = metric_lookup(metrics, split, method, "passive_false_attribution")
            effect = metric_lookup(metrics, split, method, "effect_error")
            unsafe = metric_lookup(metrics, split, method, "unsafe_contact")
            utility = metric_lookup(metrics, split, method, "causal_utility")
            split_rows.append(
                f"{tex_escape(split)} & {method_name(method)} & {fmt_pm(*f1)} & {success[0]:.3f} & {passive[0]:.3f} & {effect[0]:.3f} & {unsafe[0]:.3f} & {utility[0]:.3f}\\\\"
            )

    full_pair_rows = []
    for row in hard_pairs:
        full_pair_rows.append(
            f"{method_name(row['reference'])} & {tex_escape(row['metric'])} & {row['mean_diff']} & {row['ci95']} & {row['lower95']} & {row['target_better_seeds']}/{row['seeds']}\\\\"
        )

    full_stress_rows = []
    for row in stress:
        full_stress_rows.append(
            f"{tex_escape(row['stress_axis'])} & {row['stress_level']} & {method_name(row['method'])} & {row['mask_f1']} & {row['target_success']} & {row['unsafe_contact']} & {row['causal_utility']}\\\\"
        )

    stress_seed_rows = []
    for row in stress_seed:
        if row["stress_axis"] == "combined":
            stress_seed_rows.append(
                f"{row['stress_level']} & {method_name(row['method'])} & {row['seed']} & {row['mask_f1']} & {row['target_success']} & {row['unsafe_contact']} & {row['causal_utility']}\\\\"
            )

    ablation_seed_rows = []
    for row in ablation_seed:
        ablation_seed_rows.append(
            f"{tex_escape(row['split'])} & {tex_escape(row['method'])} & {row['seed']} & {row['mask_f1']} & {row['target_success']} & {row['unsafe_contact']} & {row['causal_utility']}\\\\"
        )

    fixed_seed_rows = []
    for row in fixed_seed:
        fixed_seed_rows.append(
            f"{tex_escape(row['split'])} & {row['risk_budget']} & {method_name(row['method'])} & {row['seed']} & {row['coverage']} & {row['fixed_risk_success']} & {row['executed_success']} & {row['false_safe_rate']}\\\\"
        )

    fixed_pair_rows = []
    for row in fixed_pairs:
        fixed_pair_rows.append(
            f"{tex_escape(row['split'])} & {row['risk_budget']} & {method_name(row['reference'])} & {tex_escape(row['metric'])} & {row['mean_diff']} & {row['ci95']} & {row['lower95']}\\\\"
        )

    prior_table_rows = []
    for i, row in enumerate(prior_rows[:36], start=1):
        prior_table_rows.append(
            f"\\citep{{{cite_keys[i - 1]}}} & {tex_escape(row.get('title', ''))} & {tex_escape(row.get('year', ''))} & {tex_escape(row.get('venue', ''))} & {tex_escape(row.get('hostile_score', ''))}\\\\"
        )

    lines.extend(
        [
            r"\appendix",
            r"\section{Full Per-Split Metrics}",
            longtable(
                r"Split & Method & F1 & Success & Passive false & Effect err. & Unsafe & Utility",
                split_rows,
                r"p{0.18\linewidth}p{0.20\linewidth}cccccc",
                "All per-split main metrics.",
                "tab:split-metrics",
            ),
            r"\section{Full Hard-Aggregate Pairwise Table}",
            longtable(
                r"Reference & Metric & Mean diff & CI95 & Lower95 & Better seeds",
                full_pair_rows,
                r"p{0.24\linewidth}p{0.23\linewidth}cccc",
                "All hard-aggregate paired seed differences for v5.",
                "tab:full-hard-pairs",
            ),
            r"\section{Full Stress Sweep}",
            longtable(
                r"Axis & Level & Method & F1 & Success & Unsafe & Utility",
                full_stress_rows,
                r"p{0.16\linewidth}cp{0.24\linewidth}cccc",
                "All stress axes and levels.",
                "tab:full-stress",
            ),
            r"\section{Combined Stress Seed Metrics}",
            longtable(
                r"Level & Method & Seed & F1 & Success & Unsafe & Utility",
                stress_seed_rows,
                r"cp{0.26\linewidth}ccccc",
                "Seed-level combined stress metrics.",
                "tab:combined-stress-seeds",
            ),
            r"\section{Ablation Seed Metrics}",
            longtable(
                r"Split & Ablation & Seed & F1 & Success & Unsafe & Utility",
                ablation_seed_rows,
                r"p{0.18\linewidth}p{0.30\linewidth}ccccc",
                "Seed-level ablation metrics.",
                "tab:ablation-seeds",
            ),
            r"\section{Fixed-Risk Seed Metrics}",
            longtable(
                r"Split & Budget & Method & Seed & Coverage & Fixed success & Exec. success & False-safe",
                fixed_seed_rows,
                r"p{0.18\linewidth}cp{0.20\linewidth}ccccc",
                "Seed-level fixed-risk metrics.",
                "tab:fixed-seeds",
            ),
            r"\section{Fixed-Risk Pairwise Metrics}",
            longtable(
                r"Split & Budget & Reference & Metric & Mean diff & CI95 & Lower95",
                fixed_pair_rows,
                r"p{0.17\linewidth}cp{0.20\linewidth}p{0.20\linewidth}ccc",
                "Fixed-risk paired seed differences for v5.",
                "tab:fixed-pairs",
            ),
            r"\section{Prior-Work Pressure Table}",
            (
                "This table is a local prior-work pressure table, not a final curated bibliography. "
                "Its role is to keep visible the nearby work that a reviewer could use to challenge novelty or baseline adequacy."
            ),
            longtable(
                r"Citation & Title & Year & Venue & Hostile score",
                prior_table_rows,
                r"p{0.12\linewidth}p{0.44\linewidth}cp{0.24\linewidth}c",
                "Local prior-work pressure sample.",
                "tab:prior-pressure",
            ),
            r"\bibliographystyle{iclr2026_conference}",
            r"\bibliography{references}",
            r"\end{document}",
        ]
    )

    (PAPER / "main.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {PAPER / 'main.tex'} and {PAPER / 'references.bib'}")


if __name__ == "__main__":
    main()
