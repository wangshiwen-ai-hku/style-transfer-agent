#!/usr/bin/env python3
"""
Plan transferability: does an AgenticST plan improve backends that never saw our
agent prompts?

    python scripts/run_plan_transfer.py --backends flux seedream gemini

This is the experiment that separates "task decomposition" from "prompt
engineering", which is the substance of R1.1 and R3.2. A prompt tuned to one
model's idiosyncrasies should not help a different model; a genuine decomposition
of the task should. So for each backend we produce two outputs per pair and let
them be compared directly:

  single_pass   one call, the strong-prompt baseline verbatim, content+style in
  plan_driven   the stages of a plan AgenticST wrote, replayed on this backend

The plan is *reused*, not regenerated. The planner (gemini-2.5-flash) is held
fixed across backends on purpose: if we re-planned per backend we would be
measuring "can the planner adapt to this generator", which is a different and
easier claim than the one under dispute.

Scoring is the pairwise A/B protocol, per backend:

    python scripts/score_style_ab.py --dirs plan_transfer/<backend>/single_pass \
        plan_transfer/<backend>/plan_driven --out plan_transfer/<backend>

Do not score across backends. Two backends differ in base image quality for
reasons that have nothing to do with the plan, and the quantity of interest is
the within-backend delta, not a ranking of generators.

Outputs
-------
  plan_transfer/<backend>/single_pass/<pair_id>.png
  plan_transfer/<backend>/plan_driven/<pair_id>.png
  plan_transfer/<backend>/plan_driven/<pair_id>/   per-stage intermediates
  plan_transfer/index.csv
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
from scripts.replay_plan_ablation import find_run_for_pair, initial_images  # noqa: E402
from scripts.run_baselines_local import PROMPT_STRONG  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BACKENDS = {
    "flux": "flux-kontext-pro",
    "seedream": "seedream-4-0-250828",
    "gemini": "gemini-2.5-flash-image",
}


def single_pass(content: Path, style: Path, model: str, ratio: float) -> Image.Image:
    """The strong-prompt baseline, verbatim, on this backend."""
    return image_generation_tool(
        PROMPT_STRONG, [str(content), str(style)],
        image_clare_prompt=["image 1 is `content_image`, ", "image 2 is `style_image`, "],
        target_ratio=ratio, model=model)


def plan_driven(plan: dict, images: dict, out_dir: Path, model: str, ratio: float):
    """Replay the plan's stages on this backend, honouring declared image inputs."""
    imgs = dict(images)
    out_dir.mkdir(parents=True, exist_ok=True)
    last = None
    for i, st in enumerate(plan.get("stages", []), 1):
        tags = [t for t in (st.get("required_image_tags") or []) if t in imgs]
        if not tags:
            tags = [last] if last and last in imgs else list(images)[:1]
        labels = [f"image {k + 1} is `{t}`, " for k, t in enumerate(tags)]
        img = image_generation_tool(
            str(st.get("text_prompt", "")), [str(imgs[t]) for t in tags],
            image_clare_prompt=labels, target_ratio=ratio, model=model)
        tag = st.get("generated_image_tag") or f"stage_{i}"
        path = out_dir / f"{tag}.png"
        img.save(path)
        imgs[tag] = path
        last = tag
    return (imgs[last] if last else None), len(plan.get("stages", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["flux", "seedream", "gemini"],
                    help=f"any of {sorted(BACKENDS)} or a literal APIYI model id")
    ap.add_argument("--runs", default="result_rev_compare",
                    help="root of completed runs whose plans are replayed")
    ap.add_argument("--pairs-file", default="compare/pairs.txt")
    ap.add_argument("--pairs", nargs="*", default=None, help="subset of pair_ids")
    ap.add_argument("--out", default="plan_transfer")
    ap.add_argument("--arms", nargs="+", default=["single_pass", "plan_driven"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    inputs = {}
    for line in (ROOT / args.pairs_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            c = [x.strip() for x in line.split(",")]
            if len(c) > 2:
                inputs[c[0]] = (ROOT / c[1], ROOT / c[2])

    plans = find_run_for_pair(ROOT / args.runs)
    pids = sorted(p for p in inputs if p in plans)
    if args.pairs:
        pids = [p for p in pids if p in args.pairs]
    missing = sorted(set(inputs) - set(plans))
    if missing:
        print(f"note: no plan found for {len(missing)} pair(s), excluded: {missing}")
    if not pids:
        raise SystemExit(
            f"no pair in {args.pairs_file} has a plan under {args.runs}/. "
            f"Run the agent first, or point --runs at the directory that holds "
            f"style_transfer_plan.json files.")

    out = ROOT / args.out
    rows, failures = [], []
    print(f"{len(pids)} pair(s) x {len(args.backends)} backend(s) x {len(args.arms)} arm(s)\n")

    for bname in args.backends:
        model = BACKENDS.get(bname, bname)
        print(f"=== {bname}  ({model}) ===")
        for pid in pids:
            content, style = inputs[pid]
            ratio = (lambda s: s[0] / s[1])(Image.open(content).size)
            for arm in args.arms:
                final = out / bname / arm / f"{pid}.png"
                if final.exists() and not args.overwrite:
                    print(f"  [{arm:11s}] {pid:16s} exists, skipping")
                    rows.append({"backend": bname, "model": model, "arm": arm,
                                 "pair": pid, "image": str(final.relative_to(ROOT))})
                    continue
                print(f"  [{arm:11s}] {pid:16s} ...", end="", flush=True)
                t0 = time.time()
                try:
                    if arm == "single_pass":
                        img, n = single_pass(content, style, model, ratio), 1
                        final.parent.mkdir(parents=True, exist_ok=True)
                        img.save(final)
                    else:
                        plan = json.loads(
                            (plans[pid] / "style_transfer_plan.json").read_text(encoding="utf-8"))
                        imgs = initial_images(plans[pid])
                        if "content_image" not in imgs:
                            imgs = {"content_image": content, "style_image": style}
                        src, n = plan_driven(plan, imgs, out / bname / arm / pid, model, ratio)
                        if src is None:
                            raise RuntimeError("plan produced no image")
                        final.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(src, final)
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e)[:100]}"
                    print(f" FAILED  {msg}")
                    failures.append((bname, arm, pid, msg))
                    continue
                print(f" ok ({time.time() - t0:.0f}s, {n} call(s))")
                rows.append({"backend": bname, "model": model, "arm": arm,
                             "pair": pid, "image": str(final.relative_to(ROOT))})
        print()

    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["backend", "model", "arm", "pair", "image"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {(out / 'index.csv').relative_to(ROOT)} ({len(rows)} rows)")

    if failures:
        print(f"\n{len(failures)} failure(s) -- rerun this command to retry only these:")
        for b, a, p, m in failures:
            print(f"  {b:10s} {a:11s} {p:16s} {m}")

    # A backend that produced only one arm cannot be compared; say so here rather
    # than letting the scorer report a one-sided ranking.
    print("\nnext:")
    for bname in args.backends:
        have = {a for a in args.arms
                if any(r["backend"] == bname and r["arm"] == a for r in rows)}
        n = {a: sum(r["backend"] == bname and r["arm"] == a for r in rows) for a in args.arms}
        if len(have) < 2 or min(n.values()) == 0:
            print(f"  {bname}: incomplete ({n}) -- not scorable, rerun the missing arm")
            continue
        print(f"  python scripts/score_style_ab.py --dirs {args.out}/{bname}/single_pass "
              f"{args.out}/{bname}/plan_driven --out {args.out}/{bname}")


if __name__ == "__main__":
    main()
