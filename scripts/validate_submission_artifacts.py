import csv
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
DOWNLOADS_PDF = Path.home() / "Downloads" / "84.pdf"
DESKTOP_PDF = Path.home() / "Desktop" / "84.pdf"

EXPECTED_ROWS = {
    "rollouts.csv": 74880,
    "dataset_summary.csv": 5760,
    "raw_seed_metrics.csv": 1170,
    "metrics.csv": 1755,
    "pairwise_stats.csv": 864,
    "hard_aggregate_seed_metrics.csv": 130,
    "hard_aggregate_metrics.csv": 195,
    "hard_aggregate_pairwise_stats.csv": 96,
    "ablation_rollouts.csv": 16000,
    "ablation_seed_metrics.csv": 200,
    "ablation_metrics.csv": 20,
    "stress_sweep_raw.csv": 94080,
    "stress_sweep_seed_metrics.csv": 2940,
    "stress_sweep.csv": 294,
    "fixed_risk_raw.csv": 30720,
    "fixed_risk_seed_metrics.csv": 480,
    "fixed_risk_metrics.csv": 48,
    "fixed_risk_pairwise.csv": 160,
    "negative_cases.csv": 24,
}


def csv_count(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def pdf_pages(path):
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        pass
    proc = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True)
    match = re.search(r"^Pages:\s+(\d+)", proc.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("could not determine PDF page count")
    return int(match.group(1))


def main():
    for name, expected in EXPECTED_ROWS.items():
        path = RESULTS / name
        if not path.exists():
            raise SystemExit(f"missing expected CSV: {path}")
        observed = csv_count(path)
        if observed != expected:
            raise SystemExit(f"row-count mismatch for {name}: expected {expected}, observed {observed}")

    summary = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    required_summary = [
        "Terminal recommendation: KILL_ARCHIVE",
        "Main rollout rows: 74880",
        "Dataset rows: 5760",
        "Ablation rollout rows: 16000",
        "Stress rollout rows: 94080",
        "Fixed-risk rollout rows: 30720",
        "Negative cases: 24",
        "margin_gate: False",
        "paired_gate: False",
        "safety_gate: False",
        "ablation_gate: False",
        "fixed_risk_gate: False",
    ]
    for token in required_summary:
        if token not in summary:
            raise SystemExit(f"missing summary token: {token}")

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    required_tex = [
        r"colorlinks=false",
        r"pdfborder={0 0 1.6}",
        r"citebordercolor={0 1 0}",
        r"linkbordercolor={1 0.55 0}",
        r"urlbordercolor={0 0.45 1}",
        r"\bibliography{references}",
    ]
    for token in required_tex:
        if token not in tex:
            raise SystemExit(f"missing LaTeX token: {token}")

    log_path = PAPER / "main.log"
    if not log_path.exists():
        raise SystemExit("missing LaTeX log")
    log = log_path.read_text(encoding="utf-8", errors="ignore")
    forbidden = [
        "Citation `",
        "Reference `",
        "undefined references",
        "Rerun to get cross-references right",
        "Package natbib Warning",
        "LaTeX Error",
    ]
    for token in forbidden:
        if token in log:
            raise SystemExit(f"LaTeX log still contains forbidden token: {token}")

    if not DOWNLOADS_PDF.exists():
        raise SystemExit(f"missing Downloads PDF: {DOWNLOADS_PDF}")
    if DESKTOP_PDF.exists():
        raise SystemExit(f"Desktop PDF must be absent: {DESKTOP_PDF}")
    pages = pdf_pages(DOWNLOADS_PDF)
    if pages < 25:
        raise SystemExit(f"PDF too short: {pages} pages")
    digest = sha256(DOWNLOADS_PDF)
    print(f"validated Paper 84 artifacts: pages={pages}, sha256={digest}")


if __name__ == "__main__":
    main()
