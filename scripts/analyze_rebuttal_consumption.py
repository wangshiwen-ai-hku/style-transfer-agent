"""
Analyze token consumption and generated image count for result_for_rebuttal.

For each experiment directory, this script:
  - Reads all LLM call report JSONs from the `logs/` subfolder.
  - Separates stats by stage: plan (system_orchestration), execute, reflection.
  - Records input_tokens, output_tokens, total_tokens per stage separately.
  - Counts generated images (all images except image_1.* and image_2.*).
  - Outputs a per-experiment CSV and an aggregate summary CSV.

Usage:
    cd /path/to/style-transfer-agent
    python scripts/analyze_rebuttal_consumption.py
"""

import os
import json
import csv
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def classify_stage(filename: str) -> str:
    """Map a log JSON filename to a stage label."""
    name = filename.lower()
    if "system_orchestration" in name:
        return "plan"
    if "reflection" in name:
        return "reflection"
    return "execute"


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
INPUT_IMAGES = {"image_1", "image_2"}


# ──────────────────────────────────────────────────────────────────────────────
# Per-experiment analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_experiment(exp_dir: Path, base_dir: Path) -> dict | None:
    """Return a stats dict for a single experiment directory, or None if invalid."""
    logs_dir = exp_dir / "logs"
    if not logs_dir.exists():
        return None

    stages = ("plan", "execute", "reflection")
    stats = {
        s: {
            "input_tokens":  0,
            "output_tokens": 0,
            "total_tokens":  0,
            "calls":         0,
        }
        for s in stages
    }

    json_files = [p for p in logs_dir.glob("*.json")
                  if p.name != "resource_constraints.json"]

    if not json_files:
        return None

    for jf in json_files:
        data = load_json(jf)
        if not data or "input_tokens" not in data:
            continue

        stage = classify_stage(jf.name)
        stats[stage]["input_tokens"]  += data["input_tokens"].get("total", 0)
        stats[stage]["output_tokens"] += data["output_tokens"].get("total", 0)
        stats[stage]["total_tokens"]  += data.get("total_tokens", 0)
        stats[stage]["calls"]         += 1

    total_calls = sum(stats[s]["calls"] for s in stages)
    if total_calls == 0:
        return None

    # Count generated images (everything except input images)
    gen_images = [
        item.name
        for item in exp_dir.iterdir()
        if item.is_file()
        and item.suffix.lower() in IMAGE_EXTS
        and item.stem not in INPUT_IMAGES
    ]

    p, e, r = stats["plan"], stats["execute"], stats["reflection"]
    return {
        "style_id":           "",  # filled by caller
        "exp_path":           str(exp_dir.relative_to(base_dir)),
        # ── plan stage ──────────────────────────────────────────────────────
        "plan_input_tokens":  p["input_tokens"],
        "plan_output_tokens": p["output_tokens"],
        "plan_total_tokens":  p["total_tokens"],
        "plan_calls":         p["calls"],
        # ── execute stage ────────────────────────────────────────────────────
        "exec_input_tokens":  e["input_tokens"],
        "exec_output_tokens": e["output_tokens"],
        "exec_total_tokens":  e["total_tokens"],
        "exec_calls":         e["calls"],
        # ── reflection stage ─────────────────────────────────────────────────
        "refl_input_tokens":  r["input_tokens"],
        "refl_output_tokens": r["output_tokens"],
        "refl_total_tokens":  r["total_tokens"],
        "refl_calls":         r["calls"],
        # ── overall ─────────────────────────────────────────────────────────
        "total_input_tokens":  p["input_tokens"]  + e["input_tokens"]  + r["input_tokens"],
        "total_output_tokens": p["output_tokens"] + e["output_tokens"] + r["output_tokens"],
        "total_tokens":        p["total_tokens"]  + e["total_tokens"]  + r["total_tokens"],
        "total_calls":         total_calls,
        # ── images ──────────────────────────────────────────────────────────
        "gen_images_count":   len(gen_images),
        "gen_images":         ", ".join(sorted(gen_images)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Directory traversal
# ──────────────────────────────────────────────────────────────────────────────

def collect_results(base_dir: Path) -> list[dict]:
    """Walk the two-level structure: base_dir / <style_id> / <exp_dir>."""
    results = []
    for first in sorted(base_dir.iterdir()):
        if not first.is_dir():
            continue

        if (first / "logs").exists():
            # Direct experiment under base_dir (unexpected but handled)
            row = analyze_experiment(first, base_dir)
            if row:
                row["style_id"] = "root"
                results.append(row)
        else:
            # first is a style_id folder
            for exp in sorted(first.iterdir()):
                if not exp.is_dir():
                    continue
                if (exp / "logs").exists():
                    row = analyze_experiment(exp, base_dir)
                    if row:
                        row["style_id"] = first.name
                        results.append(row)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CSV output helpers
# ──────────────────────────────────────────────────────────────────────────────

DETAIL_FIELDS = [
    "style_id", "exp_path",
    "plan_input_tokens",  "plan_output_tokens",  "plan_total_tokens",  "plan_calls",
    "exec_input_tokens",  "exec_output_tokens",  "exec_total_tokens",  "exec_calls",
    "refl_input_tokens",  "refl_output_tokens",  "refl_total_tokens",  "refl_calls",
    "total_input_tokens", "total_output_tokens", "total_tokens", "total_calls",
    "gen_images_count", "gen_images",
]

SUMMARY_STAGES = [
    # (label, input_key, output_key, total_key, calls_key)
    ("Plan",       "plan_input_tokens",  "plan_output_tokens",  "plan_total_tokens",  "plan_calls"),
    ("Execute",    "exec_input_tokens",  "exec_output_tokens",  "exec_total_tokens",  "exec_calls"),
    ("Reflection", "refl_input_tokens",  "refl_output_tokens",  "refl_total_tokens",  "refl_calls"),
    ("Total",      "total_input_tokens", "total_output_tokens", "total_tokens",       "total_calls"),
]

def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def write_detail_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: str, rows: list[dict]) -> None:
    n = len(rows)
    header = ["Stage", "Avg Input Tokens", "Avg Output Tokens",
              "Avg Total Tokens", "Avg API Calls"]
    summary = [header]
    for label, ik, ok, tk, ck in SUMMARY_STAGES:
        summary.append([
            label,
            f"{avg(rows, ik):.2f}",
            f"{avg(rows, ok):.2f}",
            f"{avg(rows, tk):.2f}",
            f"{avg(rows, ck):.2f}",
        ])
    # Extra row for generated images
    summary.append([
        "Generated Images",
        f"{avg(rows, 'gen_images_count'):.2f}",
        "-", "-", "-",
    ])
    # Totals row (sum across all experiments)
    summary.append([])
    summary.append(["=== Totals across all experiments (N={}) ===".format(n)])
    summary.append(["Total Input Tokens",  sum(r["total_input_tokens"]  for r in rows)])
    summary.append(["Total Output Tokens", sum(r["total_output_tokens"] for r in rows)])
    summary.append(["Total Tokens",        sum(r["total_tokens"]        for r in rows)])
    summary.append(["Total API Calls",     sum(r["total_calls"]         for r in rows)])
    summary.append(["Total Generated Images", sum(r["gen_images_count"] for r in rows)])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(summary)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(base_dir_name: str) -> None:
    base_dir = Path(base_dir_name)
    if not base_dir.exists():
        print(f"[ERROR] Directory not found: {base_dir_name}")
        return

    print(f"Analyzing {base_dir_name} ...")
    rows = collect_results(base_dir)

    if not rows:
        print(f"[WARN] No valid experiments found in {base_dir_name}.")
        return

    os.makedirs("stat", exist_ok=True)

    safe_name = base_dir_name.replace("/", "_").replace("\\", "_")
    detail_path  = f"stat/rebuttal_token_stats_{safe_name}.csv"
    summary_path = f"stat/rebuttal_token_summary_{safe_name}.csv"

    write_detail_csv(detail_path, rows)
    write_summary_csv(summary_path, rows)

    print(f"  Experiments analysed : {len(rows)}")
    print(f"  Detail CSV  → {detail_path}")
    print(f"  Summary CSV → {summary_path}")

    # Quick console summary
    total_in  = sum(r["total_input_tokens"]  for r in rows)
    total_out = sum(r["total_output_tokens"] for r in rows)
    total_tok = sum(r["total_tokens"]        for r in rows)
    total_img = sum(r["gen_images_count"]    for r in rows)
    print(f"\n  ── Overall totals ──────────────────────────────────")
    print(f"  Input tokens   : {total_in:,}  (avg {total_in/len(rows):,.1f})")
    print(f"  Output tokens  : {total_out:,}  (avg {total_out/len(rows):,.1f})")
    print(f"  Total tokens   : {total_tok:,}  (avg {total_tok/len(rows):,.1f})")
    print(f"  Total images   : {total_img}  (avg {total_img/len(rows):.2f})")


def main():
    run_analysis("result_for_rebuttal")
    # Uncomment to analyse additional directories:
    # run_analysis("result_for_rebuttal_qwen")
    # run_analysis("result_for_rebuttal_lite")
    # run_analysis("result_for_rebuttal_nolimit")


if __name__ == "__main__":
    main()
