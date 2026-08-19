#!/usr/bin/env python3
"""
Does face-embedding similarity mean what we would use it to mean?

    python scripts/validate_idsim.py --pairs-file compare/pairs_n60.txt \
        --dirs outputs_n60/ours outputs_n60/cot outputs_n60/strong

Our outputs score ID-Sim 0.33 against the content image while single-pass
baselines score 0.53-0.67. Read naively that says we destroy identity, which is
exactly the charge Reviewers 1 and 3 make. But ArcFace embeddings are trained on
photographs, and every one of these outputs is a painting. A low score could mean
the identity changed, or it could mean the measure has left its training
distribution. Reporting the number without knowing which would be reporting
nothing, and arguing "the metric is unreliable" without evidence would be
self-serving.

So the pilot builds cases whose answer is known in advance and asks whether the
measure recovers it.

  ceiling      an image against itself. Must be 1.0 or the harness is broken.
  appearance   the content image under perturbations that change every pixel's
               value but move no pixel: greyscale, posterisation, hue rotation,
               contrast, blur, and a combination of all of them. Identity and
               geometry are preserved *by construction* -- landmark positions are
               bit-identical -- so any drop here is the measure responding to
               appearance alone.
  floor        content image i against content image j, i != j. Different people,
               both photographs. This is what "identity changed" looks like to
               the measure when nothing else is out of distribution.

The methods' scores are then placed against those anchors. Two readings are
possible and the script does not choose between them:

  * If the appearance-only perturbations sit near the floor, the measure cannot
    separate "restyled" from "different person", and an ID-Sim column would carry
    no information about identity. Drop the column and report this instead.
  * If the appearance-only perturbations stay high while our outputs sit near the
    floor, the low score is about the outputs and not about the measure. Report it
    as a genuine limitation.

Detection failures are counted separately and never silently treated as a low
score; a face the detector cannot find is a different fact from a face it finds
and considers unfamiliar.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.compute_preservation_metrics import _arcface, id_embedding  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _hue_rotate(im: Image.Image, deg: int) -> Image.Image:
    h, s, v = im.convert("HSV").split()
    h = h.point(lambda p: (p + int(deg / 360 * 255)) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


# Every perturbation is a per-pixel recolouring: no resize, crop, warp, or filter
# that displaces content. Blur mixes neighbouring pixels without moving structure
# and is reported separately so a reader who objects can discount it.
#
# The set is graduated on purpose. A mild recolouring is not a fair stand-in for a
# painting, so if only mild conditions were tested, "appearance alone keeps the
# score high" would prove nothing about outputs whose appearance has changed far
# more. The harsh end -- two-level posterisation, duotone, inversion -- changes as
# large a fraction of the pixel values as a stylisation does while leaving every
# pixel where it was. Each condition's mean absolute pixel change is recorded, so
# the result is a curve of similarity against appearance change at fixed geometry,
# and each method can be placed on it.
def _duotone(im, dark=(60, 45, 35), light=(232, 216, 200)):
    g = ImageOps.grayscale(im)
    return ImageOps.colorize(g, black=dark, white=light).convert("RGB")


PERTURBATIONS = {
    "blur r=3": lambda im: im.filter(ImageFilter.GaussianBlur(3)),
    "greyscale": lambda im: ImageOps.grayscale(im).convert("RGB"),
    "hue+120": lambda im: _hue_rotate(im, 120),
    "saturation x3": lambda im: ImageEnhance.Color(im).enhance(3.0),
    "posterize3": lambda im: ImageOps.posterize(im, 3),
    "contrast x2": lambda im: ImageEnhance.Contrast(im).enhance(2.0),
    "combined": lambda im: ImageEnhance.Contrast(
        ImageOps.posterize(_hue_rotate(im, 120), 3)).enhance(2.0),
    "duotone": _duotone,
    "posterize1": lambda im: ImageOps.posterize(im, 1),
    "solarize": lambda im: ImageOps.solarize(im, threshold=110),
    "invert": lambda im: ImageOps.invert(im),
    "duotone+posterize1": lambda im: ImageOps.posterize(_duotone(im), 1),
}


def pixel_change(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute change per channel, 0-255. The x-axis of the curve."""
    x = np.asarray(a.convert("RGB"), dtype=float)
    y = np.asarray(b.convert("RGB"), dtype=float)
    return float(np.abs(x - y).mean())


def cos(a, b):
    return float(np.dot(a, b)) if a is not None and b is not None else None


def summarize(name, vals, ndet, ntot, width=26):
    if not vals:
        return f"  {name:<{width}} {'--':>7}          {ndet}/{ntot} detected"
    v = np.array(vals)
    return (f"  {name:<{width}} {v.mean():>7.3f}  "
            f"[{v.min():.3f}, {v.max():.3f}]   {ndet}/{ntot} detected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="compare/pairs_n60.txt")
    ap.add_argument("--dirs", nargs="*", default=[],
                    help="method output dirs to place against the anchors")
    ap.add_argument("--max-contents", type=int, default=10)
    ap.add_argument("--out", default="outputs_n60/idsim_validation.csv")
    args = ap.parse_args()

    contents, pairs = {}, {}
    for line in (ROOT / args.pairs_file).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            c = [x.strip() for x in s.split(",")]
            if len(c) > 2:
                pairs[c[0]] = ROOT / c[1]
                contents[Path(c[1]).stem] = ROOT / c[1]
    contents = dict(sorted(contents.items())[: args.max_contents])
    if len(contents) < 2:
        raise SystemExit("need at least two content images for the floor condition")

    app = _arcface()
    if app is None:
        raise SystemExit(
            "insightface is unavailable, so this pilot cannot run. That is not a "
            "reason to report the ID-Sim column unvalidated -- install it or drop "
            "the column.")

    print(f"{len(contents)} content image(s): {', '.join(contents)}\n")
    rows = []
    emb = {}
    for stem, p in contents.items():
        emb[stem] = id_embedding(app, p)
        if emb[stem] is None:
            print(f"  ! no face found in content image {stem}; excluded")
    usable = [s for s in contents if emb[s] is not None]

    # ---- ceiling ----
    ceil = [cos(emb[s], emb[s]) for s in usable]
    print(summarize("ceiling (self vs self)", ceil, len(usable), len(usable)))
    for s in usable:
        rows.append({"condition": "ceiling", "detail": "self", "a": s, "b": s,
                     "id_sim": round(cos(emb[s], emb[s]), 4)})

    # ---- appearance-only perturbations ----
    print("\n  appearance-only perturbations, geometry bit-identical,")
    print("  ordered by how much of the pixel content they change:\n")
    print(f"  {'condition':<22} {'dpix':>6}  {'ID-Sim':>7}  {'range':>16}  detected")
    per_cond, per_dpix = {}, {}
    for name, fn in PERTURBATIONS.items():
        vals, det, dp = [], 0, []
        for s in usable:
            im = Image.open(contents[s]).convert("RGB")
            pert = fn(im)
            dp.append(pixel_change(im, pert))
            tmp = ROOT / ".idsim_tmp.png"
            pert.save(tmp)
            e = id_embedding(app, tmp)
            tmp.unlink(missing_ok=True)
            if e is None:
                rows.append({"condition": "appearance", "detail": name, "a": s,
                             "b": name, "id_sim": "", "dpix": round(dp[-1], 1)})
                continue
            det += 1
            v = cos(emb[s], e)
            vals.append(v)
            rows.append({"condition": "appearance", "detail": name, "a": s,
                         "b": name, "id_sim": round(v, 4), "dpix": round(dp[-1], 1)})
        per_cond[name] = vals
        per_dpix[name] = float(np.mean(dp)) if dp else 0.0
        m = f"{np.mean(vals):.3f}" if vals else "--"
        rg = f"[{min(vals):.3f}, {max(vals):.3f}]" if vals else ""
        print(f"  {name:<22} {per_dpix[name]:>6.1f}  {m:>7}  {rg:>16}  "
              f"{det}/{len(usable)}")

    # ---- floor: different people ----
    floor, n = [], 0
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            n += 1
            v = cos(emb[a], emb[b])
            floor.append(v)
            rows.append({"condition": "floor", "detail": "different subject",
                         "a": a, "b": b, "id_sim": round(v, 4)})
    print("\n" + summarize("floor (different people)", floor, n, n))

    # ---- methods ----
    method_means = {}
    if args.dirs:
        print("\n  methods, content image vs output:")
        for d in args.dirs:
            m = Path(d).name
            vals, det, tot = [], 0, 0
            for pid, cpath in pairs.items():
                f = ROOT / d / f"{pid}.png"
                if not f.exists():
                    continue
                tot += 1
                ce = emb.get(Path(cpath).stem)
                oe = id_embedding(app, f)
                if ce is None or oe is None:
                    rows.append({"condition": "method", "detail": m, "a": pid,
                                 "b": "", "id_sim": ""})
                    continue
                det += 1
                v = cos(ce, oe)
                vals.append(v)
                rows.append({"condition": "method", "detail": m, "a": pid,
                             "b": "", "id_sim": round(v, 4)})
            method_means[m] = float(np.mean(vals)) if vals else None
            print(summarize(m, vals, det, tot))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "detail", "a", "b", "id_sim", "dpix"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out.relative_to(ROOT)}")

    # ---- verdict ----
    fl = float(np.mean(floor)) if floor else None
    harsh = max(per_dpix, key=lambda k: per_dpix[k]) if per_dpix else None
    worst = (float(np.mean(per_cond[harsh])) if harsh and per_cond.get(harsh)
             else min((float(np.mean(v)) for v in per_cond.values() if len(v)),
                      default=None))
    print("\n" + "=" * 62)
    if fl is None or worst is None:
        print("not enough detections to judge; do not report the column either way.")
        return
    print(f"floor (different people):            {fl:.3f}")
    print(f"harshest appearance-only condition:  {worst:.3f}"
          f"   ({harsh}, dpix={per_dpix.get(harsh, 0):.1f})")
    margin = worst - fl
    if margin < 0.10:
        print(f"\nAn appearance change that moves no pixel already drives the measure to\n"
              f"within {margin:.3f} of two different people. It cannot separate 'restyled'\n"
              f"from 'different identity', so an ID-Sim column would not be evidence\n"
              f"about identity. Report this negative result instead of the column.")
    else:
        print(f"\nAppearance alone stays {margin:.3f} above the different-person floor, so\n"
              f"the measure is not merely tracking appearance. Where a method's score\n"
              f"approaches the floor, that is about the outputs. Report the column,\n"
              f"with these anchors beside it.")
    for m, v in method_means.items():
        if v is None:
            continue
        where = ("below the floor" if v < fl else
                 "at the floor" if v < fl + 0.05 else
                 "below every appearance-only condition" if v < worst else
                 "within the appearance-only range")
        print(f"  {m:<10} {v:.3f}  ({where})")


if __name__ == "__main__":
    main()
