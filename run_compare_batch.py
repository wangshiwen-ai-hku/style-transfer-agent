#!/usr/bin/env python3
"""
Run AgenticST over the 8 typical cases in compare/pairs.txt.

    python run_compare_batch.py

Everything else has a sensible default. The run is resumable: pairs whose final
image already exists are skipped, so re-running after a network drop costs
nothing. Failures are collected and reported at the end rather than aborting the
batch.

Why this exists instead of run_agent_batch.py
---------------------------------------------
run_agent_batch.py drives run_agent.py with --task_type general, i.e. the
UNCONSTRAINED graph, and passes images as (style, content) to match that entry
point's default image_tags. The runs validated for this revision use
run_agent_limited.py with config_limited.yaml, which is a different graph and
declares image_tags as (content_image, style_image) -- the opposite order.
Mixing the two silently swaps content and style, or silently changes which graph
produced your numbers. This script reads image_tags out of the config and orders
--images to match, so the ordering cannot drift out of sync with the config.

Outputs
-------
  <result_dir>/<pair_id>/...            full run directory, one per pair
  <final_dir>/<pair_id>.png             final image, named for cross-method alignment
  traces/                               stats + per-run trace.md (unless --no-traces)
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


REQUIRED = ("langchain_core", "openai", "langgraph", "yaml", "PIL")


def _works(exe: str) -> bool:
    """True if this interpreter can import everything a run needs."""
    if not exe or not Path(exe).exists():
        return False
    probe = "import " + ", ".join(REQUIRED)
    try:
        return subprocess.run([exe, "-c", probe], capture_output=True,
                              timeout=60).returncode == 0
    except Exception:
        return False


def _conda_envs():
    """Env name -> python path, from `conda env list`.

    conda is often absent from PATH in non-interactive shells, and this machine
    keeps its envs outside the conda root (/homedata/... while conda itself is in
    /home/...), so guessing <conda_root>/envs/<name> is not reliable. Ask conda.
    """
    conda = shutil.which("conda")
    if not conda:
        for c in (Path.home() / "miniconda3/bin/conda", Path.home() / "anaconda3/bin/conda",
                  Path("/opt/conda/bin/conda")):
            if c.exists():
                conda = str(c)
                break
    if not conda:
        return {}
    try:
        out = subprocess.run([conda, "env", "list"], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception:
        return {}
    envs = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace("*", " ").split()
        if len(parts) >= 2:
            envs[parts[0]] = str(Path(parts[-1]) / "bin" / "python")
    return envs


def resolve_python(explicit: str = None, env_name: str = "agenticst") -> str:
    """Pick an interpreter that can actually run the agent.

    Using sys.executable unconditionally is what broke the first batch: the
    launcher was started with a python outside the project environment, and every
    one of the eight subprocesses died on `import langchain_core` in under a
    second. Verify before spending an hour of API calls on it.
    """
    tried = []
    for label, cand in (
        ("--python", explicit),
        ("sys.executable", sys.executable),
        ("$CONDA_PREFIX", (Path(os.environ["CONDA_PREFIX"]) / "bin" / "python").as_posix()
         if os.environ.get("CONDA_PREFIX") else None),
        (f"conda env '{env_name}'", _conda_envs().get(env_name)),
    ):
        if not cand:
            continue
        tried.append(f"{label}: {cand}")
        if _works(cand):
            return cand
        if label == "--python":
            raise SystemExit(f"--python {cand} cannot import {', '.join(REQUIRED)}")

    raise SystemExit(
        "No interpreter found that can import " + ", ".join(REQUIRED) + ".\n"
        "Tried:\n  " + "\n  ".join(tried) + "\n\n"
        f"Activate the environment first:  conda activate {env_name}\n"
        "or point at it explicitly:        --python /path/to/envs/"
        f"{env_name}/bin/python")



def load_pairs(path: Path):
    pairs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 3:
            print(f"  ! malformed line skipped: {raw}", file=sys.stderr)
            continue
        pairs.append({"pair_id": cols[0], "content": cols[1], "style": cols[2],
                      "style_attr": cols[3] if len(cols) > 3 else ""})
    return pairs


def image_order(config_path: Path):
    """Return image_tags from the config so --images can be ordered to match.

    Hard-coding the order here would reintroduce exactly the bug this script
    exists to avoid.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tags = cfg.get("image_tags") or ["content_image", "style_image"]
    if not ({"content_image", "style_image"} <= set(tags)):
        raise SystemExit(f"{config_path} image_tags must contain content_image and "
                         f"style_image; got {tags}")
    return tags


def final_image_of(run_dir: Path):
    """The image produced by the last stage of the plan, not the newest file."""
    import json
    plan = run_dir / "style_transfer_plan.json"
    if plan.exists():
        try:
            stages = json.loads(plan.read_text(encoding="utf-8")).get("stages", [])
            if stages:
                cand = run_dir / f'{stages[-1]["generated_image_tag"]}.png'
                if cand.exists():
                    return cand
        except Exception:
            pass
    pngs = sorted(run_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pngs[0] if pngs else None


def is_valid_output(png: Path) -> bool:
    """A pair counts as done only if its final image is actually loadable.

    Checking existence alone is not enough: a killed generation can leave a
    zero-byte or truncated PNG behind, and that would make the retry pass skip
    exactly the pairs that need retrying.
    """
    try:
        if not png.exists() or png.stat().st_size < 1024:
            return False
        from PIL import Image
        with Image.open(png) as im:
            im.verify()
        return True
    except Exception:
        return False


def failure_reason(log: Path) -> str:
    """Last meaningful error line from a run log, for the summary table."""
    if not log.exists():
        return "no log"
    try:
        lines = [l.strip() for l in
                 log.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    except Exception:
        return "unreadable log"
    for l in reversed(lines):
        if any(k in l for k in ("Error", "error", "Exception", "Traceback",
                                "Failed", "failed", "429", "Timeout")):
            return l[:150]
    return lines[-1][:150] if lines else "empty log"


def newest_run_under(d: Path):
    cands = [p.parent for p in d.rglob("style_transfer_plan.json")]
    if not cands:
        cands = [p for p in d.rglob("*") if p.is_dir() and list(p.glob("*.png"))]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main():
    ap = argparse.ArgumentParser()
    # --pairs-file is an alias. run_contract_ablation.py uses --pairs for a list of
    # pair IDs and --pairs-file for the path; having the same flag mean two different
    # things across two scripts in the same directory is a trap, so accept both here.
    ap.add_argument("--pairs", "--pairs-file", dest="pairs",
                    default="compare/pairs.txt",
                    help="path to a pairs file (pair_id,content,style,...)")
    ap.add_argument("--config", default="config_limited.yaml")
    ap.add_argument("--result-dir", default="result_rev_compare")
    ap.add_argument("--final-dir", default="outputs/ours")
    ap.add_argument("--traces-dir", default="traces")
    ap.add_argument("-lm", "--llm-provider", default="gemini",
                    help="understanding backbone; use 'qwen' for the Qwen3-VL row")
    ap.add_argument("-g", "--gen-image-model", default="gemini",
                    help="generation backend; keep fixed across all methods")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per pair")
    ap.add_argument("--prompt", default=None,
                    help="user-level instruction passed through R_user. The default "
                         "rule set treats expression/gaze as identity-critical; supply "
                         "an instruction here to reproduce the figures in which they "
                         "were deliberately transferred.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--retries", type=int, default=2,
                    help="extra passes over pairs that still have no valid output")
    ap.add_argument("--status", action="store_true",
                    help="report which pairs are missing output and why, run nothing")
    ap.add_argument("--no-traces", action="store_true")
    ap.add_argument("--python", default=None,
                    help="interpreter to run the agent with; auto-detected by default")
    ap.add_argument("--conda-env", default="agenticst",
                    help="conda env to fall back to when the current one lacks deps")
    args = ap.parse_args()

    py = resolve_python(args.python, args.conda_env)

    pairs = load_pairs(ROOT / args.pairs)
    tags = image_order(ROOT / args.config)
    final_dir = ROOT / args.final_dir
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"python       : {py}")
    print(f"pairs        : {len(pairs)}  ({args.pairs})")
    print(f"config       : {args.config}   image_tags={tags}")
    print(f"understanding: {args.llm_provider}   generation: {args.gen_image_model}")
    print(f"results      : {args.result_dir}   finals: {args.final_dir}")
    print(f"user prompt  : {args.prompt or '(default rule set only)'}\n")

    t_batch = time.time()

    def pending(ps):
        """Pairs whose final image is missing or unusable."""
        return [q for q in ps
                if args.overwrite or not is_valid_output(final_dir / f'{q["pair_id"]}.png')]

    already = [q["pair_id"] for q in pairs if q not in pending(pairs)]
    if already:
        print(f"already complete, skipping: {', '.join(already)}\n")

    if args.status:
        todo = pending(pairs)
        print(f"{len(pairs) - len(todo)}/{len(pairs)} complete")
        for q in todo:
            log = ROOT / args.result_dir / q["pair_id"] / "batch_stdout.log"
            print(f"  MISSING {q['pair_id']:16s} {failure_reason(log)}")
        return 0 if not todo else 1

    done, failed, skipped = [], [], already
    reasons = {}

    for attempt in range(1, args.retries + 2):
        todo = pending(pairs)
        if not todo:
            break
        if attempt > 1:
            wait = 30 * (attempt - 1)
            print(f"\n{'-' * 62}\nretry pass {attempt - 1}/{args.retries}: "
                  f"{len(todo)} pair(s) still without output; waiting {wait}s first")
            print("  " + ", ".join(q["pair_id"] for q in todo))
            time.sleep(wait)

        for i, p in enumerate(todo, 1):
            pid = p["pair_id"]
            out_png = final_dir / f"{pid}.png"
            run_dir = ROOT / args.result_dir / pid
            run_dir.mkdir(parents=True, exist_ok=True)

            ordered = [p["content"] if t == "content_image" else p["style"] for t in tags]
            cmd = [py, "run_agent_limited.py",
                   "--config", args.config,
                   "--images", *ordered,
                   "--result_dir", str(run_dir),
                   "--llm-provider", args.llm_provider,
                   "--gen_image_model", args.gen_image_model]

            # flush: with stdout redirected to a log file Python block-buffers,
            # so progress lines sit invisible for minutes and the run looks hung.
            print(f"[pass {attempt}] [{i}/{len(todo)}] {pid}  ({p['style_attr']})",
                  flush=True)
            t0 = time.time()
            log = run_dir / "batch_stdout.log"
            try:
                with log.open("wb") as fh:
                    subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=args.timeout, check=True)
            except subprocess.TimeoutExpired:
                print(f"      TIMEOUT after {args.timeout}s -- salvaging whatever exists")
            except subprocess.CalledProcessError as e:
                print(f"      run_agent_limited.py exited {e.returncode}")

            inner = newest_run_under(run_dir)
            img = final_image_of(inner) if inner else None
            if img and is_valid_output(img):
                shutil.copy(img, out_png)
                print(f"      ok in {time.time() - t0:.0f}s -> "
                      f"{out_png.relative_to(ROOT)} (from {img.name})", flush=True)
                if pid not in done:
                    done.append(pid)
                reasons.pop(pid, None)
            else:
                reasons[pid] = failure_reason(log)
                print(f"      FAILED: {reasons[pid]}", flush=True)

    failed = [q["pair_id"] for q in pending(pairs)]
    print(f"\n{'=' * 62}")
    print(f"done {len(done)}  already-complete {len(skipped)}  failed {len(failed)}  "
          f"in {time.time() - t_batch:.0f}s")
    if failed:
        print(f"\nstill failing after {args.retries} retry pass(es):")
        for pid in failed:
            print(f"  {pid:16s} {reasons.get(pid, '?')}")
            print(f"  {'':16s} log: {args.result_dir}/{pid}/batch_stdout.log")
        print("\nre-run the same command to retry only these.")

    if not args.no_traces and (done or skipped):
        print(f"\nextracting traces -> {args.traces_dir}")
        subprocess.run([py, "scripts/extract_traces.py",
                        "--runs", args.result_dir, "--out", args.traces_dir], cwd=ROOT)
        print(f"\nSend {args.traces_dir}/stats.txt and {args.traces_dir}/contracts.json "
              "for the paper numbers.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
