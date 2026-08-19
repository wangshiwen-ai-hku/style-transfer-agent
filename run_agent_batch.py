#!/usr/bin/env python3
import subprocess
import os
import shutil
import sys
import time
import logging
import random
from pathlib import Path
import argparse
import glob
parser = argparse.ArgumentParser()
parser.add_argument("--task_type", "-t", help="Task type.", default="general")
parser.add_argument("--source", "-s", help="Source directory.", default="data/samples")
# result dir
parser.add_argument("--result_dir", "-r", help="Result directory.", default="result_for_stat")
# result_compare_dir
parser.add_argument("--result_compare_dir", "-rc", help="Result compare directory.", default="result_compare_samples")
# und model
parser.add_argument("--lm_model", "-lm", help="Und model.", default="gpt4o")
# gen model
parser.add_argument("--gen_image_model", "-g", help="Generation model.", default="gemini")
parser.add_argument("--directly", "-d", help="Directly perform style transfer.", action="store_true", default=False)
parser.add_argument("--pair", "-p", help="str of pair", default="")
parser.add_argument("--resize", "-rs", help="Resize the content image.", default=1024)
args = parser.parse_args()

# if not args.pair:
#     pairs = glob.glob('result_compare/*.png')
#     com_pairs = [os.path.basename(p).split('.')[0] for p in pairs]
# else:
#     com_pairs = args.pair.split(',')
# print("total pairs: ", len(com_pairs))
# print("pairs: ", pairs)

STYLE_DIR = Path(args.source) / "style"
CONTENT_DIR = Path(args.source) / "content"
RESULT_EXP_GENERAL = Path(args.result_dir)
RESULT_COMPARE = Path(args.result_compare_dir)


RUN_AGENT = Path("run_agent.py")

os.makedirs(RESULT_EXP_GENERAL, exist_ok=True)
os.makedirs(RESULT_COMPARE, exist_ok=True)

# Logging setup
LOG_LEVEL = os.environ.get("BATCH_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_agent_batch")
logger.info("Initialized run_agent_batch")

def list_images(directory: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in directory.iterdir() if p.suffix.lower() in exts]
    import re

    def sort_key(p: Path):
        stem = p.stem
        m = re.search(r"\d+", stem)
        if m:
            # numeric-first ordering
            return (0, int(m.group()), stem.lower())
        # non-numeric names come after numeric ones, ordered lexicographically
        return (1, stem.lower())

    files.sort(key=sort_key)
    return files

def sanitize_name(name: str) -> str:
    # keep alphanumeric, dash and underscore; replace other chars with underscore
    return ''.join(c if (c.isalnum() or c in ('-', '_')) else '_' for c in name)

def find_last_stage_image(project_dir: Path):
    # Heuristic: look for files with names containing 'final' or 'final_stylized' or 'final_styled' or 'final_styled_portrait' or 'final_stylized_image'
    candidates = []
    for p in project_dir.rglob("*.png"):
        candidates.append(p)
    if candidates:
        # prefer newest
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    # fallback: pick the newest png in the project dir
    all_png = list(project_dir.rglob("*.png"))
    if all_png:
        all_png.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return all_png[0]
    return None

def run_pair(style_path: Path, content_path: Path, timeout: int = 300):
    # use sanitized image base names for directories and output filenames
    sname = sanitize_name(style_path.stem)
    cname = sanitize_name(content_path.stem)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_result_dir = RESULT_EXP_GENERAL / f"sty_{sname}_cnt_{cname}_t{timestamp}"
    run_result_dir.mkdir(parents=True, exist_ok=True)

    # per-run file logger
    fh = logging.FileHandler(run_result_dir / "batch.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    run_logger = logging.getLogger(f"run_agent_batch.run.{sname}.{cname}")
    run_logger.setLevel(logging.DEBUG)
    run_logger.addHandler(fh)

    cmd = [sys.executable, str(RUN_AGENT), "--llm-provider", args.lm_model, "--images", str(style_path), str(content_path), "--result_dir", str(run_result_dir), "--task_type", args.task_type,
    "--gen_image_model", args.gen_image_model]
    if args.directly:
        cmd += ["-d"]
    if args.resize:
        cmd += ["-rs", str(args.resize)]

    start_time = time.time()
    run_logger.info("Starting run for style=%s content=%s", sname, cname)
    run_logger.debug("cmd: %s", " ".join(cmd))

    try:
        # capture stdout/stderr to files inside run_result_dir
        with open(run_result_dir / "stdout.log", "wb") as outf, open(run_result_dir / "stderr.log", "wb") as errf:
            proc = subprocess.Popen(cmd, stdout=outf, stderr=errf)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # attempt to salvage any partially written output
                proc.kill()
                run_logger.warning("Timeout after %ds — attempting to salvage latest image", timeout)
                # give the process a moment to flush files
                time.sleep(1.0)

                # try to find any image produced so far in the run dir
                salvage_img = find_last_stage_image(run_result_dir)
                if not salvage_img:
                    nested_dirs = [d for d in run_result_dir.iterdir() if d.is_dir()]
                    for d in nested_dirs:
                        cand = find_last_stage_image(d)
                        if cand:
                            salvage_img = cand
                            break

                if salvage_img:
                    dest_name = f"sty_{sname}_cnt_{cname}{salvage_img.suffix}"
                    dest_path = RESULT_COMPARE / dest_name
                    try:
                        shutil.copy(salvage_img, dest_path)
                        duration = time.time() - start_time
                        run_logger.info("Saved salvage image after timeout to %s (elapsed %.1fs)", dest_path, duration)
                        return True, str(dest_path)
                    except Exception as e:
                        run_logger.exception("Failed to copy salvage image: %s", e)
                        return False, f"timeout after {timeout}s, failed to save image: {e}"

                run_logger.error("Timeout and no image to salvage")
                return False, f"timeout after {timeout}s, no image found"

        if proc.returncode != 0:
            run_logger.error("Process exited with code %d", proc.returncode)
            return False, f"exit code {proc.returncode}"

        # find last stage image inside run_result_dir
        last_img = find_last_stage_image(run_result_dir)
        if not last_img:
            # maybe run_agent created nested project dir under run_result_dir; search deeper
            nested_dirs = [d for d in run_result_dir.iterdir() if d.is_dir()]
            for d in nested_dirs:
                cand = find_last_stage_image(d)
                if cand:
                    last_img = cand
                    break

        if not last_img:
            run_logger.error("No output image found")
            return False, "no output image found"

        dest_name = f"sty_{sname}_cnt_{cname}{last_img.suffix}"
        dest_path = RESULT_COMPARE / dest_name
        shutil.copy(last_img, dest_path)

        duration = time.time() - start_time
        run_logger.info("Completed successfully in %.1fs, saved to %s", duration, dest_path)
        return True, str(dest_path)

    except Exception as e:
        run_logger.exception("Exception during run: %s", e)
        return False, str(e)
    finally:
        # clean up handlers
        run_logger.removeHandler(fh)
        fh.close()

def main():
    styles = list_images(STYLE_DIR)
    contents = list_images(CONTENT_DIR)

    if not styles:
        logger.error("No style images in %s", STYLE_DIR)
        return
    if not contents:
        logger.error("No content images in %s", CONTENT_DIR)
        return

    failed = []
    success = []

    # Build all combinations and shuffle to avoid running similar images in sequence
    pairs = []
    for c in contents:
        for s in styles:
            pairs.append((c, s))

    random.shuffle(pairs)
    total = len(pairs)
    print("total pairs: ", total)
    for idx, (c, s) in enumerate(pairs, start=1):
        sname = sanitize_name(s.stem)
        cname = sanitize_name(c.stem)

        # Skip if final image for this combination already exists in RESULT_COMPARE
        prefix = f"sty_{sname}_cnt_{cname}"
        print("prefix: ", prefix)
        # if not prefix in com_pairs:
        #     logger.info("Skipping pair %d/%d: %s + %s (not in pairs)", idx, total, c.name, s.name)
        #     continue
        existing = list(RESULT_COMPARE.glob(prefix + ".*"))
        if existing:
            logger.info("Skipping pair %d/%d: %s + %s (already exists: %s)", idx, total, c.name, s.name, existing[0].name)
            continue

        # Also skip if a previous run directory exists in RESULT_EXP_GENERAL for this combo
        # re_prefix = f"sty_{sname}_cnt_{cname}_"
        # prior_runs = [p for p in RESULT_EXP_GENERAL.iterdir() if p.is_dir() and p.name.startswith(re_prefix)]
        # if prior_runs:
        #     logger.info("Skipping pair %d/%d because a run directory exists in result_exp_general: %s", idx, total, prior_runs[0].name)
        #     continue

        logger.info("Running pair %d/%d: %s + %s", idx, total, c.name, s.name)
        ok, info = run_pair(s, c)
        if ok:
            logger.info("  -> success: %s", info)
            success.append((s.name, c.name, info))
        else:
            logger.warning("  -> failed: %s", info)
            failed.append((s.name, c.name, info))

    logger.info("\nBatch finished")
    logger.info("Succeeded: %d, Failed: %d", len(success), len(failed))
    if failed:
        logger.warning("Failures (%d):", len(failed))
        for f in failed:
            logger.warning(" - %s", f)

if __name__ == '__main__':
    main()
