#!/usr/bin/env python3
"""
Style-fidelity scoring via an MLLM judge, over the APIYI endpoint (no GPU, no
model deployment).

    python scripts/score_style_fidelity.py --dirs outputs/ours outputs/weak outputs/strong

Why this exists
---------------
Preservation metrics alone cannot support a claim about our method, because they
systematically favour whichever method changes the image least: a pipeline that
does nothing scores perfectly on landmark drift, gaze, ID similarity and edge
distance. Measured on the typical-case set, the direct MLLM baselines do stylize
less and consequently score *better* on preservation than AgenticST. That is not
evidence against the method; it means preservation is only interpretable jointly
with how much style was actually transferred. This script supplies the missing
axis so the two can be plotted against each other.

Design decisions that matter for whether the numbers are trustworthy
--------------------------------------------------------------------
* **Blind.** The judge never sees a method name, only content, style and result.
* **Calibrated.** Every pair also scores two anchors: the *content image itself*
  presented as if it were a result (a floor -- no style was transferred) and the
  *style image itself* (a ceiling). A judge that cannot separate the anchors is
  not measuring anything, and the script says so instead of reporting the scores.
* **Repeated.** MLLM scores are noisy; each item is scored `--reps` times and we
  report mean and standard deviation, not a single draw.
* **Rubric-bound.** The prompt enumerates the same stylistic dimensions the paper
  claims to transfer, so the score is about those, not about generic prettiness.

Outputs
-------
  <out>/style_fidelity.csv      per item, per repetition
  <out>/style_fidelity_summary.csv   per method: mean, sd, n
"""
import argparse
import csv
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import apiyi  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

RUBRIC = """You are an art critic scoring how faithfully a candidate image reproduces
the STYLE of a reference artwork.

Image 1 is the CONTENT source (the scene that was to be restyled).
Image 2 is the STYLE reference.
Image 3 is the CANDIDATE result.

Score ONLY how faithfully image 3 reproduces the style of image 2. Do not reward
or penalise how well it preserves image 1 -- that is measured separately, and a
candidate that is merely a lightly-filtered copy of image 1 must score LOW here.

Judge these dimensions and weigh them equally:
  - colour palette: dominant hues, accents, saturation and value range
  - line: weight, variation, continuity, drawn versus implied contours
  - texture and mark-making: brush or pen behaviour, granularity, edge quality
  - composition treatment: use of negative space, figure-to-ground balance
  - decorative motifs belonging to the style rather than to its subject matter
  - light and atmosphere

Reply with ONLY a JSON object:
{"style_fidelity": <integer 0-100>, "reason": "<one sentence>"}

Anchors: 0 means no stylistic relationship to image 2 at all; 100 means a viewer
would take image 3 for a work by the same hand as image 2."""


def parse_score(text: str):
    try:
        i, j = text.index("{"), text.rindex("}") + 1
        d = json.loads(text[i:j])
        s = float(d.get("style_fidelity"))
        return (s if 0 <= s <= 100 else None), str(d.get("reason", ""))[:200]
    except Exception:
        return None, text[:120]


def score(content: Path, style: Path, cand: Path, model: str):
    msg = [{"role": "user", "content": [
        {"type": "text", "text": RUBRIC},
        {"type": "text", "text": "Image 1 (content source):"},
        {"type": "image_url", "image_url": {"url": apiyi.encode_image(str(content))}},
        {"type": "text", "text": "Image 2 (style reference):"},
        {"type": "image_url", "image_url": {"url": apiyi.encode_image(str(style))}},
        {"type": "text", "text": "Image 3 (candidate result):"},
        {"type": "image_url", "image_url": {"url": apiyi.encode_image(str(cand))}},
    ]}]
    return parse_score(apiyi.chat(msg, model=model, temperature=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="directories of <pair_id>.png outputs, one per method")
    ap.add_argument("--pairs-file", default="compare/pairs.txt")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-anchors", action="store_true",
                    help="skip the calibration anchors (not recommended)")
    args = ap.parse_args()

    pairs = {}
    for line in (ROOT / args.pairs_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            c = [x.strip() for x in line.split(",")]
            pairs[c[0]] = (ROOT / c[1], ROOT / c[2])

    items = []   # (method, pair, candidate_path)
    for d in args.dirs:
        method = Path(d).name
        for png in sorted((ROOT / d).glob("*.png")):
            if png.stem in pairs:
                items.append((method, png.stem, png))
    if not args.no_anchors:
        for pid, (c, s) in pairs.items():
            items.append(("_anchor_content", pid, c))   # floor: nothing transferred
            items.append(("_anchor_style", pid, s))     # ceiling: is the style
    random.shuffle(items)   # judge sees no method-ordered run

    print(f"{len(items)} item(s) x {args.reps} rep(s) = {len(items) * args.reps} calls\n")
    rows = []
    for n, (method, pid, cand) in enumerate(items, 1):
        content, style = pairs[pid]
        for rep in range(1, args.reps + 1):
            try:
                s, why = score(content, style, cand, args.model)
            except Exception as e:
                s, why = None, f"ERROR {type(e).__name__}: {e}"[:160]
            rows.append({"method": method, "pair": pid, "rep": rep,
                         "style_fidelity": "" if s is None else s, "reason": why})
            print(f"[{n}/{len(items)}] {method:18s} {pid:16s} r{rep} -> "
                  f"{'FAIL' if s is None else f'{s:5.1f}'}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "style_fidelity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "pair", "rep", "style_fidelity", "reason"])
        w.writeheader(); w.writerows(rows)

    def vals(m):
        return [float(r["style_fidelity"]) for r in rows
                if r["method"] == m and r["style_fidelity"] != ""]

    methods = sorted({r["method"] for r in rows})
    summary = []
    print(f"\n{'method':20s} {'n':>4s} {'mean':>7s} {'sd':>6s}")
    for m in methods:
        v = vals(m)
        if not v:
            continue
        summary.append({"method": m, "n": len(v), "mean": round(st.mean(v), 2),
                        "sd": round(st.pstdev(v), 2)})
        print(f"{m:20s} {len(v):>4d} {st.mean(v):>7.2f} {st.pstdev(v):>6.2f}")
    with (out / "style_fidelity_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n", "mean", "sd"])
        w.writeheader(); w.writerows(summary)

    # calibration: the judge must separate "no style applied" from "is the style"
    floor, ceil = vals("_anchor_content"), vals("_anchor_style")
    if floor and ceil:
        gap = st.mean(ceil) - st.mean(floor)
        print(f"\ncalibration: content anchor {st.mean(floor):.1f}  "
              f"style anchor {st.mean(ceil):.1f}  gap {gap:.1f}")
        if gap < 30:
            print("!! The judge barely separates an unstyled photograph from the style\n"
                  "   reference itself. Treat every score above as uninformative and fix\n"
                  "   the rubric or the model before using these numbers in the paper.")
        else:
            print("   Gap is wide enough for the scores to carry signal. Report the\n"
                  "   anchors alongside the methods so a reader can see the scale.")
    print(f"\nwrote {out / 'style_fidelity.csv'}")
    print("Plot style fidelity against LMD from compute_preservation_metrics.py: the "
          "claim to support is comparable preservation at higher style fidelity, not "
          "better preservation.")


if __name__ == "__main__":
    main()
