#!/usr/bin/env python3
"""
Idea2Img (ECCV 2024) as a backbone-controlled agentic baseline.

    python scripts/run_idea2img.py --pairs-file compare/pairs.txt

Reviewer 2.1 asks whether our gains come from AgenticST's design or from generic
iterative MLLM generation, and names Idea2Img as the reference point. Idea2Img is
the cleanest available control for that: it is a general-purpose self-refinement
agent wrapped around a fixed generator, it is peer-reviewed, and it ships code.

Crucially we drive it with *our* generation backend (`--t2i apiyi`, already wired
into third_party/idea2img). Both systems then call the same image model, and the
only difference is the orchestration around it -- which is exactly the comparison
under dispute. Running it on its original SDXL backend would confound orchestration
with generator quality and would not answer the reviewer's question.

What this script does that the upstream entry point does not
-----------------------------------------------------------
* One invocation per pair, with --foldername set to the pair id, so results are
  addressable. Upstream keys output directories on the prompt text with spaces
  stripped, which collides across pairs and is unusable as a method directory.
* Pre-creates the output tree. Upstream uses `os.system('mkdir ...')` without -p
  and never checks the exit status, so a missing parent makes the run continue and
  silently write nothing.
* Copies the final selected image to outputs/idea2img/<pair_id>.png, the layout
  every scorer in this repo expects.
* Records the failures rather than aborting, and reruns skip completed pairs.

Note for the table caption: Idea2Img resizes its conditioning image to 1024x1024,
so its outputs are square regardless of the content image's shape. That is the
method's own behaviour and is reported as such, not corrected -- but it must be
stated, or it will be read as the aspect-ratio defect we fixed on our own side.
"""
import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
I2I = ROOT / "third_party" / "idea2img"

# Idea2Img keys its output filename on the text before the first <IMG>, with
# spaces and dots removed. Keeping the text identical across pairs means the name
# is constant and predictable; the pair id lives in --foldername instead.
PROMPT = ("Restyle the first image so that it takes on the artistic style of the "
          "second image, preserving the subject and pose of the first")
STEM = PROMPT.replace(" ", "").replace(".", "")


def load_pairs(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            c = [x.strip() for x in line.split(",")]
            if len(c) > 2:
                out[c[0]] = (ROOT / c[1], ROOT / c[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="compare/pairs.txt")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--out", default="outputs/idea2img")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--num-prompt", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per pair")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not (I2I / "idea2img_pipeline.py").exists():
        raise SystemExit(f"idea2img not found at {I2I}")

    pairs = load_pairs(ROOT / args.pairs_file)
    if args.pairs:
        pairs = {k: v for k, v in pairs.items() if k in args.pairs}
    if not pairs:
        raise SystemExit(f"no pairs matched in {args.pairs_file}")

    final_dir = ROOT / args.out
    final_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    print(f"{len(pairs)} pair(s), {args.max_rounds} round(s), "
          f"{args.num_prompt} prompt(s) per round\n")

    for pid, (content, style) in sorted(pairs.items()):
        final = final_dir / f"{pid}.png"
        if final.exists() and not args.overwrite:
            print(f"  {pid:16s} exists, skipping")
            rows.append({"pair": pid, "image": str(final.relative_to(ROOT)),
                         "seconds": "", "status": "skipped"})
            continue
        for m in (content, style):
            if not m.exists():
                failures.append((pid, f"missing input {m}"))
                print(f"  {pid:16s} FAILED  missing input {m}")
                break
        else:
            # Upstream mkdir calls are unchecked; create the whole tree first.
            for sub in ("tmp", "round1", "iter", "iter_best"):
                (I2I / "output" / pid / sub).mkdir(parents=True, exist_ok=True)
            sample = I2I / f"_sample_{pid}.txt"
            # No separator between the two <IMG> entries: upstream splits on the
            # literal tag and does not strip, so a space before the next tag ends
            # up inside the previous path. Their own sample has a single trailing
            # image, which is why this never showed up there.
            sample.write_text(f"{PROMPT} <IMG>{content}<IMG>{style}\n",
                              encoding="utf-8")

            cmd = [args.python, "-u", "idea2img_pipeline.py",
                   "--testfile", sample.name,
                   "--foldername", pid,
                   "--t2i", "apiyi",
                   "--img2img",
                   "--max_rounds", str(args.max_rounds),
                   "--num_prompt", str(args.num_prompt)]
            print(f"  {pid:16s} ...", end="", flush=True)
            t0 = time.time()
            try:
                p = subprocess.run(cmd, cwd=I2I, capture_output=True, text=True,
                                   timeout=args.timeout)
            except subprocess.TimeoutExpired:
                failures.append((pid, f"timeout after {args.timeout}s"))
                print(f" FAILED  timeout after {args.timeout}s")
                sample.unlink(missing_ok=True)
                continue
            dt = time.time() - t0
            sample.unlink(missing_ok=True)

            produced = I2I / "output" / pid / "iter_best" / f"{STEM}.png"
            if not produced.exists():
                # Fall back to whatever the run did produce, so a naming change
                # upstream degrades to a warning rather than an empty method dir.
                cands = sorted((I2I / "output" / pid / "iter_best").glob("*.png"))
                produced = cands[0] if cands else None
            if produced is None:
                tail = (p.stderr or p.stdout or "")[-300:].replace("\n", " ")
                failures.append((pid, f"no image (rc={p.returncode}): {tail[:160]}"))
                print(f" FAILED  no image produced (rc={p.returncode})")
                (I2I / "output" / pid / "run.log").write_text(
                    (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""),
                    encoding="utf-8")
                continue
            shutil.copy(produced, final)
            print(f" ok ({dt:.0f}s) -> {final.relative_to(ROOT)}")
            rows.append({"pair": pid, "image": str(final.relative_to(ROOT)),
                         "seconds": round(dt, 1), "status": "ok"})

    idx = final_dir / "index.csv"
    with idx.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "image", "seconds", "status"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {idx.relative_to(ROOT)} ({len(rows)} rows)")

    if failures:
        print(f"\n{len(failures)} failure(s) -- rerun this command to retry them:")
        for pid, why in failures:
            print(f"  {pid:16s} {why}")
    ok = sum(1 for r in rows if r["status"] in ("ok", "skipped"))
    if ok < 2:
        print("\nfewer than two outputs; not scorable yet.")
    else:
        print(f"\nnext:\n  python scripts/score_style_ab.py --dirs outputs/ours "
              f"outputs/cot outputs/refine outputs/strong outputs/weak {args.out} "
              f"--out outputs")


if __name__ == "__main__":
    main()
