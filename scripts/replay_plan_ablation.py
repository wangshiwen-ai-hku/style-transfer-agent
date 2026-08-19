#!/usr/bin/env python3
"""
Text-only execution ablation: how much of the result comes from the style image
still being visible during execution, rather than from the plan alone?

    python scripts/replay_plan_ablation.py --runs result_rev_compare --out ablation_textonly

Reviewer 3's central objection is that AgenticST "adopts both the style image and
the content image through lossy semantic intermediates before image generation",
i.e. that the style is compressed into language and the continuous signal is lost.
Our architecture does not work that way -- stages declare image inputs and the raw
pixels are passed to the generator -- but saying so is not evidence. This measures
it.

Design
------
The two arms REPLAY THE SAME PLAN. That is the entire point. Re-running the agent
with the style image withheld would also change the plan the agent writes, and the
comparison would then confound "was the reference visible during execution" with
"were these two different plans". So we load a plan produced by a completed run and
execute it twice:

  with_style     every stage receives exactly the image tags the plan declares
  text_only      the style image is removed from every stage's inputs; the text of
                 the plan, which already encodes the style analysis, is unchanged

A stage that declared only the style image would otherwise have no inputs at all;
those stages fall back to the most recent generated image, which is the closest
thing to "execute this instruction without looking at the reference".

Outputs
-------
  <out>/with_style/<pair_id>.png   and per-stage intermediates
  <out>/text_only/<pair_id>.png
  <out>/index.csv                  arm, pair, image, n_stages, n_style_stages

Score with:
  python scripts/compute_preservation_metrics.py --index <out>/index.csv
  python scripts/score_style_fidelity.py --dirs <out>/with_style <out>/text_only
The metric that matters here is style fidelity: if withholding the reference costs
little, the plan really is carrying the style, and Reviewer 3's objection has more
force than we credit. Report whichever way it comes out.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.image_generation import image_generation_tool  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def find_run_for_pair(runs_root: Path):
    """pair_id -> the newest run directory that produced a plan for it."""
    out = {}
    for plan in runs_root.rglob("style_transfer_plan.json"):
        d = plan.parent
        pid = None
        for part in d.parts:
            if part.startswith("sty_") and "_cnt_" in part:
                pid = part
                break
        if pid is None:
            pid = d.parts[-3] if len(d.parts) >= 3 else d.name
        if pid not in out or d.stat().st_mtime > out[pid].stat().st_mtime:
            out[pid] = d
    return out


def initial_images(run_dir: Path):
    """Recover the content/style inputs a run was given."""
    imgs = {}
    cfg = run_dir / "config_limited.yaml"
    order = ["content_image", "style_image"]
    if cfg.exists():
        try:
            import yaml
            tags = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("image_tags")
            if tags:
                order = tags
        except Exception:
            pass
    found = sorted([p for p in run_dir.iterdir()
                    if p.stem in ("image_1", "image_2")], key=lambda p: p.stem)
    for tag, path in zip(order, found):
        imgs[tag] = path
    return imgs


def replay(plan, images, out_dir: Path, drop_style: bool, model: str, ratio: float):
    """Execute the stages of `plan` in order, optionally without the style image."""
    imgs = dict(images)
    out_dir.mkdir(parents=True, exist_ok=True)
    last, n_style = None, 0
    for i, st in enumerate(plan.get("stages", []), 1):
        tags = [t for t in (st.get("required_image_tags") or []) if t in imgs]
        n_style += any("style" in t.lower() for t in tags)
        if drop_style:
            tags = [t for t in tags if "style" not in t.lower()]
            if not tags:
                # The instruction still has to be executed on something; the most
                # recent intermediate is the honest stand-in for "no reference".
                tags = [last] if last and last in imgs else []
        labels = [f"image {k + 1} is `{t}`, " for k, t in enumerate(tags)]
        img = image_generation_tool(
            str(st.get("text_prompt", "")), [str(imgs[t]) for t in tags],
            image_clare_prompt=labels, target_ratio=ratio, model=model)
        tag = st.get("generated_image_tag") or f"stage_{i}"
        path = out_dir / f"{tag}.png"
        img.save(path)
        imgs[tag] = path
        last = tag
    return (imgs[last] if last else None), len(plan.get("stages", [])), n_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="result_rev_compare",
                    help="root of completed runs whose plans will be replayed")
    ap.add_argument("--out", default="ablation_textonly")
    ap.add_argument("--pairs", nargs="*", default=None, help="subset of pair_ids")
    ap.add_argument("--model", default="gemini-2.5-flash-image")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    runs = find_run_for_pair(ROOT / args.runs)
    if args.pairs:
        runs = {k: v for k, v in runs.items() if k in args.pairs}
    if not runs:
        raise SystemExit(f"no runs with a plan under {args.runs}")
    out = ROOT / args.out
    print(f"replaying {len(runs)} plan(s) from {args.runs}\n")

    rows = []
    for pid, run_dir in sorted(runs.items()):
        plan = json.loads((run_dir / "style_transfer_plan.json").read_text(encoding="utf-8"))
        images = initial_images(run_dir)
        if "content_image" not in images:
            print(f"  {pid}: cannot recover inputs, skipping"); continue
        ratio = (lambda s: s[0] / s[1])(Image.open(images["content_image"]).size)

        for arm, drop in (("with_style", False), ("text_only", True)):
            final = out / arm / f"{pid}.png"
            if final.exists() and not args.overwrite:
                print(f"  [{arm}] {pid}: exists, skipping")
                rows.append({"arm": arm, "pair": pid, "repeat": "",
                             "image": str(final.relative_to(ROOT)), "run_dir": ""})
                continue
            print(f"  [{arm}] {pid} ...", end="", flush=True)
            t0 = time.time()
            try:
                img, n, ns = replay(plan, images, out / arm / pid, drop, args.model, ratio)
            except Exception as e:
                print(f" FAILED: {str(e)[:110]}"); continue
            if img is None:
                print(" FAILED: plan produced no image"); continue
            (out / arm).mkdir(parents=True, exist_ok=True)
            shutil.copy(img, final)
            print(f" ok ({time.time() - t0:.0f}s, {n} stages, {ns} used the style image)")
            rows.append({"arm": arm, "pair": pid, "repeat": "",
                         "image": str(final.relative_to(ROOT)),
                         "run_dir": str((out / arm / pid).relative_to(ROOT))})

    idx = out / "index.csv"
    idx.parent.mkdir(parents=True, exist_ok=True)
    with idx.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "pair", "repeat", "image", "run_dir"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {idx.relative_to(ROOT)} ({len(rows)} rows)")
    print("\nnext:")
    print(f"  python scripts/score_style_fidelity.py --dirs {args.out}/with_style {args.out}/text_only")
    print(f"  python scripts/compute_preservation_metrics.py --index {args.out}/index.csv")


if __name__ == "__main__":
    main()
