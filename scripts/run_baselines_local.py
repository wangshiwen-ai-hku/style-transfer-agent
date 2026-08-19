#!/usr/bin/env python3
"""
Local (API-only) baselines for the TVCG revision, Sec. 4.4 "Isolating the
Contribution of the Agentic Design".

Every baseline here shares the SAME generation backend, decoding settings and
output aspect ratio as AgenticST, so that differences are attributable to
orchestration rather than to the generator. That control is the entire point of
the experiment -- if you change the backend for one row, the table is void.

Baselines implemented (paper labels in brackets):
  weak    [Direct, minimal prompt]   single call, the prompt used in the original submission
  strong  [Direct, strong prompt]    single call, long human-authored instruction
  cot     [Single-agent CoT]         analyze-then-generate within one session
  refine  [Generic self-refinement]  fixed N-round critique/re-edit loop, no DyMAG,
                                     no per-pair orchestration, no preservation contract

Usage
-----
  python scripts/run_baselines_local.py --pairs compare/pairs.txt \
      --out outputs --methods weak strong cot refine --rounds 3

Outputs
-------
  outputs/<method>/<pair_id>.png       final image, filename == pair_id
  outputs/<method>/<pair_id>.json      prompts, per-round text, token usage, timing

The JSON sidecar is not optional bookkeeping: Sec. 4.1 of the paper promises that
every baseline prompt is reported verbatim, and these files are the source for it.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.image_generation import image_generation_tool  # noqa: E402

# --------------------------------------------------------------------------
# Prompts. These are reproduced verbatim in the supplementary material; edit
# them here and re-export rather than keeping a second copy in the .tex.
# --------------------------------------------------------------------------

PROMPT_WEAK = "Transfer the style of the style image to the content image."

PROMPT_STRONG = """You are an expert artist performing an artistic style transfer.

Image 1 is the CONTENT image. Image 2 is the STYLE reference.

Analyse the style reference along every dimension that carries its identity, and
reproduce all of them in your output:
- colour palette: dominant hues, accents, saturation and value range;
- line: weight, variation, continuity, and whether contours are drawn or implied;
- texture and mark-making: brush or pen behaviour, granularity, edge quality;
- composition: layout, use of negative space, framing, figure-to-ground balance;
- decorative motifs: recurring ornaments, patterns or symbols that belong to the
  style rather than to its subject matter;
- light and atmosphere: light direction, contrast, and the overall mood.

Apply these to the content image in a content-aware way: map facial traits onto
the face, background traits onto the background, and keep the result visually
harmonious rather than uniformly textured.

Preserve from the content image: the identity of the subject, the pose and
spatial arrangement, the number and relative position of objects, and any legible
text. Do not import the subject matter of the style reference into the output.

Produce the final stylized image."""

PROMPT_COT_ANALYZE = """Image 1 is the CONTENT image. Image 2 is the STYLE reference.

Before generating anything, analyse both images and write:
1. the stylistic attributes of the reference that must be reproduced (palette,
   line, texture, composition, motifs, light and mood);
2. the content attributes that must be preserved (subject identity, pose, spatial
   arrangement, objects, text);
3. a short ordered description of how you will apply the style to this particular
   content.

Answer in plain prose. Do not generate an image yet."""

PROMPT_COT_GENERATE = """Using the analysis below, generate the final stylized image.
Image 1 is the CONTENT image, image 2 is the STYLE reference.

--- ANALYSIS ---
{analysis}
--- END ANALYSIS ---

Produce the final stylized image."""

PROMPT_REFINE_CRITIQUE = """Image 1 is the CONTENT image, image 2 is the STYLE reference,
image 3 is the CURRENT RESULT.

Critique the current result: how faithfully does it reproduce the style reference,
and how well does it preserve the content? Then give a single concrete instruction
for improving it in one further editing step. Answer with the critique first, then
a line beginning "INSTRUCTION:" containing only that instruction."""

PROMPT_REFINE_APPLY = """Image 1 is the CURRENT RESULT, image 2 is the STYLE reference.
Apply the following improvement and output the revised image.

{instruction}"""


def load_pairs(path: Path, root: Path):
    """Parse pairs.txt -> list of dicts. Comment lines start with '#'."""
    pairs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 3:
            print(f"  ! skipping malformed line: {raw}", file=sys.stderr)
            continue
        pair_id, content, style = cols[0], cols[1], cols[2]
        pairs.append({
            "pair_id": pair_id,
            "content": str(root / content),
            "style": str(root / style),
            "style_attr": cols[3] if len(cols) > 3 else "",
        })
    return pairs


def target_ratio_of(content_path: str) -> float:
    w, h = Image.open(content_path).size
    return w / h


def call_text_llm(prompt: str, image_paths, model: str):
    """Text/vision LLM call for the CoT and self-refinement baselines.

    Goes through the same APIYI adapter as AgenticST, so the baselines and our
    method share one code path for model configuration and one endpoint.
    """
    from src.utils import apiyi

    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({"type": "image_url", "image_url": {"url": apiyi.encode_image(p)}})
    return apiyi.chat(
        [{"role": "system", "content": "You are an expert artist and art critic."},
         {"role": "user", "content": content}],
        model=model, temperature=0.7)


def run_single_call(pair, prompt, gen_model, out_dir):
    """Baselines 'weak' and 'strong': one generation call, no iteration."""
    img = image_generation_tool(
        prompt,
        [pair["content"], pair["style"]],
        image_clare_prompt=["Image 1 is the content image.", "Image 2 is the style image."],
        target_ratio=target_ratio_of(pair["content"]),
        model=gen_model,
    )
    img.save(out_dir / f"{pair['pair_id']}.png")
    return {"prompt": prompt, "rounds": 1}


def run_cot(pair, gen_model, lm_model, out_dir):
    """Analyze-then-generate inside a single session; still one generation call."""
    analysis = call_text_llm(PROMPT_COT_ANALYZE, [pair["content"], pair["style"]], lm_model)
    prompt = PROMPT_COT_GENERATE.format(analysis=analysis)
    img = image_generation_tool(
        prompt,
        [pair["content"], pair["style"]],
        image_clare_prompt=["Image 1 is the content image.", "Image 2 is the style image."],
        target_ratio=target_ratio_of(pair["content"]),
        model=gen_model,
    )
    img.save(out_dir / f"{pair['pair_id']}.png")
    return {"analysis": analysis, "prompt": prompt, "rounds": 1}


def run_refine(pair, gen_model, lm_model, out_dir, rounds):
    """Generic self-refinement: a fixed critique/re-edit schedule.

    This is the ablation that matters most for the paper's claim, so keep it
    honest: it gets the SAME number of generation calls as AgenticST's mean, and
    the same backend. What it does not get is per-pair orchestration, a
    case-specific critique specification, or a preservation contract.
    """
    ratio = target_ratio_of(pair["content"])
    cur = image_generation_tool(
        PROMPT_STRONG,
        [pair["content"], pair["style"]],
        image_clare_prompt=["Image 1 is the content image.", "Image 2 is the style image."],
        target_ratio=ratio, model=gen_model,
    )
    tmp = out_dir / f"{pair['pair_id']}_r0.png"
    cur.save(tmp)

    trace = [{"round": 0, "prompt": PROMPT_STRONG}]
    for r in range(1, rounds):
        critique = call_text_llm(PROMPT_REFINE_CRITIQUE,
                                 [pair["content"], pair["style"], str(tmp)], lm_model)
        instruction = critique.split("INSTRUCTION:")[-1].strip() if "INSTRUCTION:" in critique else critique.strip()
        apply_prompt = PROMPT_REFINE_APPLY.format(instruction=instruction)
        cur = image_generation_tool(
            apply_prompt, [str(tmp), pair["style"]],
            image_clare_prompt=["Image 1 is the current result.", "Image 2 is the style image."],
            target_ratio=ratio, model=gen_model,
        )
        tmp = out_dir / f"{pair['pair_id']}_r{r}.png"
        cur.save(tmp)
        trace.append({"round": r, "critique": critique, "instruction": instruction})

    cur.save(out_dir / f"{pair['pair_id']}.png")
    return {"trace": trace, "rounds": rounds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="compare/pairs.txt")
    ap.add_argument("--root", default=".", help="root that pair paths are relative to")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--methods", nargs="+", default=["weak", "strong", "cot", "refine"],
                    choices=["weak", "strong", "cot", "refine"])
    ap.add_argument("--gen-model", default="gemini-2.5-flash-image",
                    help="MUST match the generator AgenticST uses, or the comparison is void")
    ap.add_argument("--lm-model", default="gemini-2.5-flash")
    ap.add_argument("--rounds", type=int, default=3,
                    help="generation calls for --methods refine; set to AgenticST's mean")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pairs = load_pairs(Path(args.pairs), root)
    print(f"loaded {len(pairs)} pairs from {args.pairs}")

    for method in args.methods:
        out_dir = Path(args.out) / method
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {method} -> {out_dir} ===")

        for i, pair in enumerate(pairs, 1):
            final = out_dir / f"{pair['pair_id']}.png"
            if final.exists() and not args.overwrite:
                print(f"  [{i}/{len(pairs)}] {pair['pair_id']}: exists, skipping")
                continue

            t0 = time.time()
            try:
                if method == "weak":
                    meta = run_single_call(pair, PROMPT_WEAK, args.gen_model, out_dir)
                elif method == "strong":
                    meta = run_single_call(pair, PROMPT_STRONG, args.gen_model, out_dir)
                elif method == "cot":
                    meta = run_cot(pair, args.gen_model, args.lm_model, out_dir)
                else:
                    meta = run_refine(pair, args.gen_model, args.lm_model, out_dir, args.rounds)
            except Exception as e:
                print(f"  [{i}/{len(pairs)}] {pair['pair_id']}: FAILED -- {e}")
                continue

            meta.update({"pair_id": pair["pair_id"], "method": method,
                         "content": pair["content"], "style": pair["style"],
                         "gen_model": args.gen_model, "lm_model": args.lm_model,
                         "seconds": round(time.time() - t0, 1)})
            (out_dir / f"{pair['pair_id']}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [{i}/{len(pairs)}] {pair['pair_id']}: ok ({meta['seconds']}s)")

    print("\ndone. Outputs are named <pair_id>.png so every method aligns by filename.")


if __name__ == "__main__":
    main()
