#!/usr/bin/env python3
"""
Extract, from AgenticST run directories, everything the TVCG revision needs from
the execution traces. Run this after re-running the pipeline; it replaces the
lost traces and produces the numbers that several paper claims depend on.

What it produces
----------------
traces/summary.csv          one row per run: stages, reflections, tokens, timing
traces/stage_inputs.csv     one row per stage: which image tags it consumed
traces/contracts.json       the preservation contract of each run, per region
traces/<pair_id>/trace.md   human-readable trace, for the supplementary case study
traces/stats.txt            the aggregate numbers cited in the paper

Paper claims this backs
-----------------------
  Sec. 4.4  "X% of executed stages take the style image as a direct input"
            -> stats.txt: style_image_stage_pct     [R3 comment 1-1]
  Sec. 4.5  "an average of N image generations per style transfer task"
            -> stats.txt: mean_generated_images
  Sec. 6    "converges within two in 97% of cases, average 1.56"
            -> stats.txt: reflection_distribution
  Sec. 3.2  the verbatim preservation contract quoted in the paper
            -> contracts.json                        [R1 comment 2]

IMPORTANT: the aggregate statistics in the paper must be recomputed from ONE run
set. Do not mix numbers taken from the old 800-run logs with numbers from a new
set -- if the new run gives different values, update the paper to the new values.

Usage
-----
  python scripts/extract_traces.py --runs result_exp_compare --out traces
  python scripts/extract_traces.py --runs result_for_stat --out traces_400
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

# Keys under which analysis agents express the preserve/transform split. The agent
# graph is generated per run, so field names vary; we look for any of these.
CONTRACT_KEYS = ("preservation_contract", "similar_regions_transfer_detail",
                 "similar_region_transfer_detail", "non_style_content_to_preserve",
                 "style_image_content_to_remove")


def find_runs(root: Path):
    """A run directory is any directory containing a style_transfer_plan.json."""
    return sorted(p.parent for p in root.rglob("style_transfer_plan.json"))


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def pair_id_of(run_dir: Path) -> str:
    """Recover sty_<s>_cnt_<c> from the run path, matching the output naming."""
    for part in reversed(run_dir.parts):
        m = re.search(r"(sty_[A-Za-z0-9_.-]+?_cnt_[A-Za-z0-9_.-]+?)(?:_t\d{8}_\d{6})?$", part)
        if m:
            return m.group(1)
    return run_dir.name


def collect_tokens(run_dir: Path):
    """Sum token usage across the per-call LLM reports in logs/.

    llm_call_and_report writes input_tokens / output_tokens as DICTS
    ({"text":..,"images_count":..,"images":..,"total":..}), not integers. An
    earlier version of this function tested isinstance(v, (int, float)) and so
    silently recorded zero for every run -- which would have put wrong token
    counts in the paper. Handle both shapes.
    """
    tin = tout = calls = 0

    def total_of(v):
        if isinstance(v, dict):
            return int(v.get("total") or 0)
        return int(v) if isinstance(v, (int, float)) else 0

    for f in (run_dir / "logs").glob("*.json"):
        rep = load_json(f)
        if not isinstance(rep, dict):
            continue
        calls += 1
        usage = rep.get("usage") or {}
        tin += total_of(rep.get("input_tokens", usage.get("input_tokens")))
        tout += total_of(rep.get("output_tokens", usage.get("output_tokens")))
    return tin, tout, calls


def collect_reflections(run_dir: Path):
    """Reflection count and whether the run ended satisfied."""
    n = len(list(run_dir.glob("reflection_*_prompt.txt")))
    satisfied = None
    summary = run_dir / "final_reflection_summary.txt"
    if summary.exists():
        txt = summary.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Final Satisfaction:\s*(\w+)", txt)
        if m:
            satisfied = m.group(1).lower() == "true"
        m = re.search(r"Reflection Count:\s*(\d+)", txt)
        if m:
            n = int(m.group(1))
    return n, satisfied


def collect_contract(run_dir: Path):
    """Pull the preserve/transform declaration out of whichever analysis file has it."""
    found = {}
    for f in run_dir.glob("analysis_*.json"):
        data = load_json(f)
        if not isinstance(data, dict):
            continue
        for key in CONTRACT_KEYS:
            if key in data and data[key]:
                found.setdefault(f.name, {})[key] = data[key]
    for name in ("style_transfer_analysis.json", "comprehensive_style_analysis.json"):
        data = load_json(run_dir / name)
        if isinstance(data, dict):
            for key in CONTRACT_KEYS:
                if key in data and data[key]:
                    found.setdefault(name, {})[key] = data[key]
    return found


def write_trace_md(out_dir: Path, pair_id: str, run_dir: Path, plan, contract, refl_n):
    """Human-readable trace for the supplementary case study."""
    lines = [f"# Execution trace: {pair_id}", "", f"Run directory: `{run_dir}`", ""]

    orch = load_json(run_dir / "system_orchestration.json")
    if orch:
        lines += ["## Agent graph (DyMAG)", ""]
        for a in orch.get("agent_graph", []):
            deps = ", ".join(a.get("dependencies") or []) or "-"
            tags = ", ".join(a.get("required_image_tags") or []) or "-"
            lines.append(f"- **{a.get('agent_name')}** (T={a.get('temperature')}) "
                         f"deps: {deps} | images: {tags}")
        lines += ["", "## Critique specification", "",
                  "```", str(orch.get("result_critique_criteria", "")).strip(), "```", ""]

    if contract:
        lines += ["## Preservation contract (emitted BEFORE generation)", ""]
        for src, fields in contract.items():
            lines.append(f"### from `{src}`")
            for k, v in fields.items():
                lines += [f"**{k}**", "", "```",
                          json.dumps(v, indent=2, ensure_ascii=False) if not isinstance(v, str) else v,
                          "```", ""]

    lines += ["## Plan", ""]
    for i, st in enumerate(plan.get("stages", []), 1):
        lines += [f"### Stage {i}: {st.get('stage_name')}",
                  f"- generates: `{st.get('generated_image_tag')}`",
                  f"- consumes: {st.get('required_image_tags')}",
                  f"- temperature: {st.get('gen_temperature')}",
                  "", "```", str(st.get("text_prompt", "")).strip(), "```", ""]

    lines += [f"## Reflection ({refl_n} round(s))", ""]
    for f in sorted(run_dir.glob("reflection_*_prompt.txt")):
        lines += [f"### `{f.name}`", "", "```",
                  f.read_text(encoding="utf-8", errors="ignore").strip()[:4000], "```", ""]

    d = out_dir / pair_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="root containing run directories")
    ap.add_argument("--out", default="traces")
    args = ap.parse_args()

    root, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(root)
    print(f"found {len(runs)} runs under {root}")
    if not runs:
        print("nothing to do -- is --runs pointing at the result directory?")
        return

    summary, stage_rows, contracts = [], [], {}
    n_stage_total = n_stage_with_style = 0
    refl_counter = Counter()

    for run_dir in runs:
        pid = pair_id_of(run_dir)
        plan = load_json(run_dir / "style_transfer_plan.json") or {}
        stages = plan.get("stages", [])
        tin, tout, calls = collect_tokens(run_dir)
        refl_n, satisfied = collect_reflections(run_dir)
        contract = collect_contract(run_dir)
        if contract:
            contracts[pid] = contract
        refl_counter[refl_n] += 1

        for i, st in enumerate(stages, 1):
            tags = st.get("required_image_tags") or []
            uses_style = any("style" in str(t).lower() for t in tags)
            n_stage_total += 1
            n_stage_with_style += int(uses_style)
            stage_rows.append({"pair_id": pid, "stage_idx": i,
                               "stage_name": st.get("stage_name"),
                               "generated_tag": st.get("generated_image_tag"),
                               "required_tags": "|".join(map(str, tags)),
                               "uses_style_image": int(uses_style),
                               "gen_temperature": st.get("gen_temperature")})

        summary.append({"pair_id": pid, "run_dir": str(run_dir),
                        "n_stages": len(stages),
                        "n_generated_images": len(list(run_dir.glob("*.png"))),
                        "n_reflections": refl_n, "satisfied": satisfied,
                        "input_tokens": tin, "output_tokens": tout,
                        "total_tokens": tin + tout, "n_llm_calls": calls,
                        "has_contract": int(bool(contract))})

        write_trace_md(out, pid, run_dir, plan, contract, refl_n)

    def dump(name, rows):
        if not rows:
            return
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("summary.csv", summary)
    dump("stage_inputs.csv", stage_rows)
    (out / "contracts.json").write_text(
        json.dumps(contracts, indent=2, ensure_ascii=False), encoding="utf-8")

    n = len(summary)
    mean = lambda k: sum(r[k] for r in summary) / n if n else 0
    pct_style = 100.0 * n_stage_with_style / n_stage_total if n_stage_total else 0
    within2 = 100.0 * sum(v for k, v in refl_counter.items() if k <= 2) / n if n else 0

    stats = [
        f"runs                      : {n}",
        f"mean_stages               : {mean('n_stages'):.2f}",
        f"mean_generated_images     : {mean('n_generated_images'):.2f}",
        f"mean_reflections          : {mean('n_reflections'):.2f}",
        f"reflections_within_2_pct  : {within2:.1f}%",
        f"reflection_distribution   : {dict(sorted(refl_counter.items()))}",
        f"mean_input_tokens         : {mean('input_tokens'):.1f}",
        f"mean_output_tokens        : {mean('output_tokens'):.1f}",
        f"mean_total_tokens         : {mean('total_tokens'):.1f}",
        "",
        f"stages_total              : {n_stage_total}",
        f"stages_using_style_image  : {n_stage_with_style}",
        f"style_image_stage_pct     : {pct_style:.1f}%   <- Sec. 4.4, answers R3 comment 1-1",
        "",
        f"runs_with_contract        : {sum(r['has_contract'] for r in summary)}/{n}",
    ]
    (out / "stats.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")
    print("\n".join(stats))
    print(f"\nwrote {out}/summary.csv, stage_inputs.csv, contracts.json, stats.txt, "
          f"and {n} per-run trace.md")


if __name__ == "__main__":
    main()
