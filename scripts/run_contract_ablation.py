#!/usr/bin/env python3
"""
Preservation-contract ablation: does declaring what to keep actually keep it?

    python scripts/run_contract_ablation.py --pairs sty_114_cnt_23 sty_112_cnt_23 --repeats 3

Runs each pair under two rule repositories that differ in exactly one thing --
whether style_transfer.md asks the analysis agent to emit a preservation
contract -- and repeats each arm so the comparison is not one sample of a
stochastic generator.

Why the repeats matter
----------------------
We observed that the ink-wash pair the reviewers singled out (closed eyes in the
submitted teaser) now comes out with the expression preserved, and that the
contract for that run declares "maintain ... eye/lip placement". That is
suggestive, not causal: the executor is stochastic and one run proves nothing.
This script produces the paired samples needed to say whether the contract
changes the outcome distribution or merely coincided with a good draw.

Arms
----
  contract    src/general_limited/rules_contract/      (control: contract required)
  nocontract  src/general_limited/rules_nocontract/    (ablation: block removed)

Both repositories contain the same file set and differ only in the contract
block, so skill selection sees an equivalent menu in both arms.

Outputs
-------
  <out>/<arm>/<pair_id>_r<k>.png       final image per arm and repeat
  <out>/<arm>/<pair_id>_r<k>/          full run directory (traces, contract, logs)
  <out>/index.csv                      arm, pair, repeat, image path, run dir

Feed <out> to scripts/compute_preservation_metrics.py to get the numbers.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_compare_batch import (  # noqa: E402
    final_image_of, image_order, is_valid_output, load_pairs, newest_run_under,
    resolve_python, failure_reason)

ROOT = Path(__file__).resolve().parent.parent
ARMS = {
    "contract":   "src/general_limited/rules_contract",
    "nocontract": "src/general_limited/rules_nocontract",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=["sty_114_cnt_23", "sty_112_cnt_23"],
                    help="pair_ids from compare/pairs.txt; default is the two "
                         "portrait cases the reviewers raised")
    ap.add_argument("--pairs-file", default="compare/pairs.txt")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--config", default="config_limited.yaml")
    ap.add_argument("--out", default="ablation_contract")
    ap.add_argument("-g", "--gen-image-model", default="gemini")
    ap.add_argument("-lm", "--llm-provider", default="gemini")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--python", default=None)
    ap.add_argument("--conda-env", default="agenticst")
    args = ap.parse_args()

    py = resolve_python(args.python, args.conda_env)
    # load_pairs takes only the path; content/style stay relative to ROOT, which
    # is also the subprocess cwd, so they resolve without further joining.
    all_pairs = {p["pair_id"]: p for p in load_pairs(ROOT / args.pairs_file)}
    missing = [p for p in args.pairs if p not in all_pairs]
    if missing:
        raise SystemExit(f"unknown pair_id(s): {', '.join(missing)}")
    pairs = [all_pairs[p] for p in args.pairs]
    tags = image_order(ROOT / args.config)

    for arm, rel in ARMS.items():
        if not (ROOT / rel).is_dir():
            raise SystemExit(f"missing rule repository for arm '{arm}': {rel}")

    out = ROOT / args.out
    print(f"python  : {py}")
    print(f"pairs   : {', '.join(args.pairs)}")
    print(f"arms    : {', '.join(ARMS)}   repeats: {args.repeats}")
    print(f"out     : {args.out}\n")

    rows, failures = [], []
    for arm, rel in ARMS.items():
        env = dict(os.environ, AGENTICST_RULES_DIR=str(ROOT / rel))
        arm_dir = out / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for p in pairs:
            for k in range(1, args.repeats + 1):
                tag = f'{p["pair_id"]}_r{k}'
                png = arm_dir / f"{tag}.png"
                if is_valid_output(png):
                    print(f"[{arm}] {tag}: exists, skipping")
                    rows.append({"arm": arm, "pair": p["pair_id"], "repeat": k,
                                 "image": str(png.relative_to(ROOT)), "run_dir": ""})
                    continue

                run_dir = arm_dir / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                ordered = [p["content"] if t == "content_image" else p["style"]
                           for t in tags]
                cmd = [py, "run_agent_limited.py", "--config", args.config,
                       "--images", *ordered, "--result_dir", str(run_dir),
                       "--llm-provider", args.llm_provider,
                       "--gen_image_model", args.gen_image_model]

                print(f"[{arm}] {tag} ...", end="", flush=True)
                t0 = time.time()
                log = run_dir / "batch_stdout.log"
                try:
                    with log.open("wb") as fh:
                        subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                                       stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=True)
                except subprocess.TimeoutExpired:
                    print(" timeout", end="")
                except subprocess.CalledProcessError as e:
                    print(f" exit {e.returncode}", end="")

                inner = newest_run_under(run_dir)
                img = final_image_of(inner) if inner else None
                if img and is_valid_output(img):
                    shutil.copy(img, png)
                    print(f" ok ({time.time() - t0:.0f}s)")
                    rows.append({"arm": arm, "pair": p["pair_id"], "repeat": k,
                                 "image": str(png.relative_to(ROOT)),
                                 "run_dir": str(inner.relative_to(ROOT))})
                else:
                    print(f" FAILED: {failure_reason(log)}")
                    failures.append(tag + f" [{arm}]")

    idx = out / "index.csv"
    with idx.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "pair", "repeat", "image", "run_dir"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} run(s) recorded in {idx.relative_to(ROOT)}")
    if failures:
        print("failed: " + ", ".join(failures) + "\n(re-run to retry only these)")
    print("\nnext:  python scripts/compute_preservation_metrics.py "
          f"--index {idx.relative_to(ROOT)} --pairs-file {args.pairs_file}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
