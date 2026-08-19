# StyleTransfer Agent (AgenticST)

![Tessar](./tessar.jpg)

Code and evaluation harness for *AgenticST: Rethinking Style Transfer with
Multi-Agent Collaboration*.

The system takes a content image and a style image, has an orchestrator
synthesize a multi-stage plan for that specific pair, and executes the plan
against an image-generation backend. What distinguishes it from prompting a
generator directly is not the number of calls but that the plan states, before
any pixels are produced, which attributes of the content image the stylization
may change and which it must keep.

This repository is the artifact for the paper's claims. It contains the running
system, the sixty-pair evaluation set, every scoring script behind a reported
number, and the raw judgements those numbers were computed from.

---

## What is here

```
src/general/            the agent graph (orchestrator, DyMAG, planner, executor, critic)
src/general_limited/    the released configuration used for every reported number
src/utils/apiyi.py      the OpenAI-compatible client all model calls go through
compare/                the evaluation manifests (the images themselves are not
                        redistributed -- see "The evaluation set" below)
scripts/                every measurement in the paper, one script per table
assets/models/          face_landmarker.task, shipped because the evaluation
                        machine has no CDN access
```

## The evaluation set

`compare/pairs_n60.txt` defines it: ten portraits crossed with six styles, sixty
pairs, each annotated with the stylistic dimension it is meant to stress. The
annotation is what makes the per-dimension analysis possible, and that analysis is
what located the one dimension on which our advantage does not hold.

**The images themselves are not in this repository.** The content images are
photographs of real people that were not collected for public release, and the
style references are third-party artworks whose redistribution licences we have
not cleared. Publishing them would hand a rights question to everyone who clones
this repository, and it is not ours to answer on their behalf.

The manifest is the reproducible part. Each line is

```
pair_id,content_path,style_path,style_attr,note
```

so pointing the paths at your own images — or at the corresponding public
datasets — makes every script in `scripts/` run unchanged. The `style_attr`
column is the only field the analysis depends on beyond the paths; the six values
used in the paper are listed in `compare/style_names.txt`. If you want the exact
set for a direct comparison, contact the authors and we will arrange access to
what we are permitted to share.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in BASE_URL and APIYI_KEY
```

All model calls — reasoning and generation — go through one OpenAI-compatible
endpoint, configured in `.env`. There is no second code path for local versus
cluster runs, which is deliberate: the earlier arrangement made results depend on
where they were produced.

The preservation metrics need `mediapipe` and `insightface` but no GPU. The
FLUX.1-Kontext experiment (`scripts/run_flux_local.py`) needs `diffusers >= 0.35`,
`torch`, and about 32 GB of VRAM; everything else runs on CPU against the API.

## Running the system

One pair:

```bash
python run_agent.py --content path/to/content.jpg --style path/to/style.jpg
```

The evaluation set:

```bash
python run_compare_batch.py --pairs compare/pairs_n60.txt --final-dir outputs/ours
```

Runs are resumable: a pair with a valid output is skipped, so re-running the
command retries only what failed.

## Reproducing the tables

Each script prints its own reliability checks before its results, and the order
matters — the framing audit decides which pairs are scorable, so it runs first.

```bash
# Baselines, same generation backend, orchestration is the only variable
python scripts/run_baselines_local.py --pairs compare/pairs_n60.txt --out outputs

# Which pairs are comparable at all (see "A caveat that shaped the protocol")
python scripts/audit_framing.py --dirs outputs/ours outputs/cot outputs/refine \
    outputs/strong outputs/weak --pairs-file compare/pairs_n60.txt \
    --write-clean compare/pairs_clean.txt

# Style fidelity: forced pairwise comparison
python scripts/score_style_ab.py --dirs outputs/ours outputs/cot outputs/refine \
    outputs/strong outputs/weak --pairs-file compare/pairs_n60.txt --out outputs

# Preservation: landmark drift, eyelid state, gaze, contour distance
python scripts/compute_preservation_metrics.py --dir outputs/ours --method ours \
    --pairs-file compare/pairs_n60.txt

# Whether face-embedding similarity means what it would be used to mean
python scripts/validate_idsim.py --pairs-file compare/pairs_n60.txt \
    --dirs outputs/ours outputs/cot outputs/strong

# Plan transferability on an open-weight, generation-only backend
python scripts/run_flux_local.py --pairs-file compare/pairs_n60.txt
python scripts/score_style_ab.py --dirs plan_transfer/flux_local/named \
    plan_transfer/flux_local/flattened plan_transfer/flux_local/plan_driven \
    --reps 5 --out plan_transfer/flux_local
```

To browse results rather than read CSVs:

```bash
python scripts/compare_dashboard.py --port 8766
```

## Three things worth knowing before you trust a number

**Style fidelity is judged pairwise, and the judge's reliability is measured
rather than assumed.** An earlier protocol asked a multimodal judge for a 0–100
score; re-scoring the same images on two occasions moved one method's mean by 5.3
points, more than the gap it was being used to establish. The current protocol
forces a binary choice, judges every method pair in both presentation orders, and
reports the order-swap agreement (83% over 600 swaps) alongside the ranking. Note
that order invariance and test–retest stability are different quantities: the
second is lower, which is why significance is computed over image pairs and never
over raw comparison counts.

**The self-comparison control is degenerate by design.** Shown two identical
images the judge has no content signal and must still choose, so it falls back on
a slot in over 90% of such trials. That number bounds nothing; the order-swap
rate is the control that does.

**A caveat that shaped the protocol.** The generation endpoint exposes no
aspect-ratio parameter, and the model takes the output's shape from the last
reference image it is shown. Methods differ in how this resolves — AgenticST
reproduces the content image's shape on 30% of pairs against 0–5% for the
single-pass baselines — and where one method's shape departs from the others in a
pair, the judge penalises it for reasons unrelated to style. `audit_framing.py`
finds those pairs so they can be excluded, and the paper reports both the full
set and the restricted one. `AGENTICST_PRESERVE_FRAMING=1` forces the framing by
reordering the reference images; it is **off** for every reported number, because
the model is demonstrably position-sensitive and reordering changes the
conditioning once per stage, which is not neutral between a multi-stage pipeline
and a single call.

## Baselines from other work

`scripts/setup_third_party.sh` clones the upstream repositories at the commits we
used and applies the patches in `patches/`. We ship the patches rather than
copies of the code so that a reader reproduces the baseline rather than trusting
our snapshot of it. Idea2Img needed four fixes to run at all in this environment,
including a message-format conversion for the OpenAI-compatible endpoint and a
token budget large enough for a reasoning model; all four are in the patch, with
comments explaining what failed.

## Repository layout note

`outputs*/`, `result_*/` and `traces/` are generated and are not committed, with
one exception: the raw judgement CSVs behind the reported tables are kept, so that
a reader checking a number does not have to re-run 1500 API calls to see what it
was computed from.
