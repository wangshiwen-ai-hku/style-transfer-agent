#!/usr/bin/env python3
"""
Plan transferability on a local, open-weight, generation-only backend.

    /homedata/swwang/conda/envs/svdquant/bin/python scripts/run_flux_local.py \
        --model-path /homedata/HuggingFace/black-forest-labs/FLUX.1-Kontext-dev

Runs FLUX.1-Kontext-dev locally. This is the strongest available evidence against
the reading that AgenticST is prompt engineering (R1.1, R3.2), for three reasons:
the weights are open and the run is reproducible by a reviewer; the model has
never observed our agent prompts; and Kontext accepts a single conditioning image
plus text, so it never sees the style image at all. Whatever the plan contributes
here, it contributes through the text alone.

The arms, and why the middle one exists
---------------------------------------
  generic     one call, a strong hand-written style-transfer instruction naming
              the target style. A floor, not a control.
  flattened   one call, carrying the concatenated text of every stage of our
              plan. Same information as plan_driven, same model, same number of
              inference steps per call -- the only difference is that it is not
              decomposed.
  plan_driven the stages executed in sequence, each conditioned on the previous
              stage's output.

`generic` vs `plan_driven` is not a sufficient experiment: a longer, more specific
prompt beating a shorter one is exactly what a reviewer means by prompt
engineering. `flattened` vs `plan_driven` holds the text content fixed and varies
only whether it is executed as a decomposition, which is the claim actually in
dispute. Report the flattened comparison as the result and the generic one as
context.

Score per backend with the pairwise protocol:
    python scripts/score_style_ab.py --dirs plan_transfer/flux_local/flattened \
        plan_transfer/flux_local/plan_driven --out plan_transfer/flux_local
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

GENERIC = (
    "Restyle this image as an artwork in the target artistic style: match its "
    "colour palette, line quality, texture and mark-making, use of negative "
    "space and overall atmosphere. Preserve the identity of the subject, the "
    "pose and the spatial arrangement. Produce the final stylized image.")

# The `named` arm substitutes the style's name into the same sentence. This is the
# floor that matters. `generic` above never says which style to apply, and Kontext
# never sees the style image, so that arm has no information about the target at
# all -- beating it demonstrates only that naming a style beats not naming one,
# which is not a claim anyone disputes. Keep `generic` as a sanity check that the
# pipeline responds to its prompt, and compare against `named` when arguing that
# the agent's pair-specific analysis contributes something a competent user's
# one-line instruction does not.
NAMED = (
    "Restyle this image in the following artistic style: {name}. Match its "
    "colour palette, line quality, texture and mark-making, use of negative "
    "space and overall atmosphere. Preserve the identity of the subject, the "
    "pose and the spatial arrangement. Produce the final stylized image.")


def load_style_names(path: Path):
    names = {}
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "," in line:
            stem, name = line.split(",", 1)
            names[stem.strip()] = name.strip()
    return names


def pick_device(explicit):
    """Default to the GPU with the most free memory.

    FLUX.1-Kontext-dev is a 12B transformer; in bf16 the transformer alone is
    about 24 GB before the T5 encoder, which does not leave headroom on a 24 GB
    card. Prefer the L40s and fall back to sequential offload if told to use a
    smaller device.
    """
    if explicit is not None:
        return int(explicit)
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return best


def load_pipe(model_path: str, device: int, offload: bool):
    from diffusers import FluxKontextPipeline
    pipe = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    free, total = torch.cuda.mem_get_info(device)
    if offload or free < 32 * 1024**3:
        print(f"  cuda:{device} has {free / 1024**3:.1f} GiB free; enabling CPU offload")
        pipe.enable_model_cpu_offload(gpu_id=device)
    else:
        pipe = pipe.to(f"cuda:{device}")
    return pipe


def gen(pipe, image: Image.Image, prompt: str, steps: int, guidance: float, seed: int):
    g = torch.Generator("cpu").manual_seed(seed)
    # Kontext keeps the conditioning image's aspect ratio; passing it explicitly
    # avoids the pipeline silently resizing to a square and changing composition,
    # which would confound the preservation measurements taken on these outputs.
    w, h = image.size
    w, h = (w // 16) * 16, (h // 16) * 16
    return pipe(image=image.resize((w, h)), prompt=prompt[:4000],
                num_inference_steps=steps, guidance_scale=guidance,
                generator=g, height=h, width=w).images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",
                    default="/homedata/HuggingFace/black-forest-labs/FLUX.1-Kontext-dev")
    ap.add_argument("--runs", default="result_rev_compare")
    ap.add_argument("--pairs-file", default="compare/pairs.txt")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--out", default="plan_transfer/flux_local")
    ap.add_argument("--arms", nargs="+",
                    default=["named", "flattened", "plan_driven"])
    ap.add_argument("--style-names", default="compare/style_names.txt")
    ap.add_argument("--device", default=None, help="cuda index; default = most free")
    ap.add_argument("--offload", action="store_true", help="force CPU offload")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    inputs = {}
    for line in (ROOT / args.pairs_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            c = [x.strip() for x in line.split(",")]
            if len(c) > 2:
                inputs[c[0]] = (ROOT / c[1], ROOT / c[2])

    # Import lazily: replay_plan_ablation pulls in the APIYI stack, which is not
    # installed in the torch environment this script runs in.
    plans = {}
    for plan in (ROOT / args.runs).rglob("style_transfer_plan.json"):
        d = plan.parent
        pid = next((p for p in d.parts if p.startswith("sty_") and "_cnt_" in p), None)
        if pid and (pid not in plans or d.stat().st_mtime > plans[pid].stat().st_mtime):
            plans[pid] = d

    pids = sorted(p for p in inputs if p in plans)
    if args.pairs:
        pids = [p for p in pids if p in args.pairs]
    if not pids:
        raise SystemExit(
            f"no pair in {args.pairs_file} has a style_transfer_plan.json under "
            f"{args.runs}/. Point --runs at the directory holding completed runs.")

    style_names = load_style_names(ROOT / args.style_names)
    if "named" in args.arms:
        missing = sorted({p.split("_cnt_")[0][len("sty_"):] for p in pids} - set(style_names))
        if missing:
            raise SystemExit(
                f"the `named` arm needs a name for every style; missing: {missing}\n"
                f"add them to {args.style_names} (one line per style: <stem>,<short name>).")

    dev = pick_device(args.device)
    print(f"{len(pids)} pair(s), arms={args.arms}, cuda:{dev}")
    print(f"loading {args.model_path} ...", flush=True)
    t0 = time.time()
    pipe = load_pipe(args.model_path, dev, args.offload)
    print(f"loaded in {time.time() - t0:.0f}s\n")

    out = ROOT / args.out
    rows, failures = [], []
    for pid in pids:
        content, _style = inputs[pid]
        plan = json.loads(
            (plans[pid] / "style_transfer_plan.json").read_text(encoding="utf-8"))
        stages = plan.get("stages", [])
        texts = [str(s.get("text_prompt", "")).strip() for s in stages]
        texts = [t for t in texts if t]
        if not texts:
            print(f"  {pid}: plan has no stage prompts, skipping"); continue

        for arm in args.arms:
            final = out / arm / f"{pid}.png"
            if final.exists() and not args.overwrite:
                print(f"  [{arm:11s}] {pid:16s} exists, skipping")
                rows.append({"arm": arm, "pair": pid, "n_calls": "",
                             "image": str(final.relative_to(ROOT))})
                continue
            print(f"  [{arm:11s}] {pid:16s} ...", end="", flush=True)
            t0 = time.time()
            try:
                img = Image.open(content).convert("RGB")
                if arm == "generic":
                    img, n = gen(pipe, img, GENERIC, args.steps, args.guidance, args.seed), 1
                elif arm == "named":
                    sstem = pid.split("_cnt_")[0][len("sty_"):]
                    if sstem not in style_names:
                        raise KeyError(
                            f"no entry for style '{sstem}' in {args.style_names}; add one "
                            f"rather than falling back to the unnamed prompt, which would "
                            f"silently turn this arm back into `generic`")
                    img, n = gen(pipe, img, NAMED.format(name=style_names[sstem]),
                                 args.steps, args.guidance, args.seed), 1
                elif arm == "flattened":
                    joined = " ".join(f"({i}) {t}" for i, t in enumerate(texts, 1))
                    img, n = gen(pipe, img, joined, args.steps, args.guidance, args.seed), 1
                else:
                    inter = out / arm / pid
                    inter.mkdir(parents=True, exist_ok=True)
                    for i, t in enumerate(texts, 1):
                        img = gen(pipe, img, t, args.steps, args.guidance, args.seed)
                        img.save(inter / f"stage_{i}.png")
                    n = len(texts)
                final.parent.mkdir(parents=True, exist_ok=True)
                img.save(final)
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:110]}"
                print(f" FAILED  {msg}")
                failures.append((arm, pid, msg))
                continue
            print(f" ok ({time.time() - t0:.0f}s, {n} call(s))")
            rows.append({"arm": arm, "pair": pid, "n_calls": n,
                         "image": str(final.relative_to(ROOT))})

    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "pair", "n_calls", "image"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {(out / 'index.csv').relative_to(ROOT)} ({len(rows)} rows)")

    if failures:
        print(f"\n{len(failures)} failure(s) -- rerun to retry only these:")
        for a, p, m in failures:
            print(f"  {a:11s} {p:16s} {m}")

    counts = {a: sum(r["arm"] == a for r in rows) for a in args.arms}
    print(f"\nper-arm counts: {counts}")
    if counts.get("flattened") and counts.get("plan_driven"):
        print("\nnext (this is the comparison to report):")
        print(f"  python scripts/score_style_ab.py --dirs {args.out}/flattened "
              f"{args.out}/plan_driven --out {args.out}")
    else:
        print("\nflattened and plan_driven are not both complete -- not scorable yet.")


if __name__ == "__main__":
    main()
