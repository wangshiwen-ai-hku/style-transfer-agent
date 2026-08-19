#!/usr/bin/env python3
"""
Flag image pairs whose methods disagree on output shape, before any scoring.

    python scripts/audit_framing.py --dirs outputs_n60/ours outputs_n60/cot ... \
        --pairs-file compare/pairs_n60.txt --write-clean compare/pairs_n60_clean.txt

The generation endpoint exposes no aspect-ratio parameter and the model takes the
output's shape from the last reference image it is shown, so outputs normally
follow the style image. That is harmless for a pairwise comparison as long as it
happens to every method equally: within a pair they are handicapped the same way.

It is not harmless when one method deviates. Measured on the earlier eight-pair
set, the two outputs that kept the content image's shape while their four
competitors followed the style reference went 0-8 and 1-7 against methods they
beat elsewhere -- the judge treats a match to the reference artwork's framing as
stylistic fidelity. Two of forty outputs deviated there, so a set six times larger
should be expected to contain a handful, and finding them after scoring means
finding them too late.

This audit runs on the images, before any judging, and writes a pairs file with
the affected pairs removed. Excluding them is the conservative choice: forcing the
framing instead would change the conditioning once per stage, which is not neutral
between a multi-stage pipeline and a single call.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def ratio(p: Path):
    try:
        with Image.open(p) as im:
            return im.size[0] / im.size[1], im.size
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--pairs-file", default="compare/pairs_n60.txt")
    ap.add_argument("--tol", type=float, default=0.06,
                    help="max |ratio - majority| treated as agreement")
    ap.add_argument("--write-clean", default=None,
                    help="write a pairs file with the flagged pairs removed")
    ap.add_argument("--out", default=None, help="per-output CSV of measured ratios")
    args = ap.parse_args()

    lines = (ROOT / args.pairs_file).read_text(encoding="utf-8").splitlines()
    pairs, meta = [], {}
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#"):
            c = [x.strip() for x in s.split(",")]
            if len(c) > 2:
                pairs.append(c[0])
                meta[c[0]] = (ROOT / c[1], ROOT / c[2], c[3] if len(c) > 3 else "")

    methods = [Path(d).name for d in args.dirs]
    dirs = {Path(d).name: ROOT / d for d in args.dirs}

    rows, flagged, missing = [], {}, defaultdict(list)
    for pid in pairs:
        got = {}
        for m in methods:
            f = dirs[m] / f"{pid}.png"
            r, size = ratio(f) if f.exists() else (None, None)
            if r is None:
                missing[pid].append(m)
            else:
                got[m] = (r, size)
            rows.append({"pair": pid, "method": m,
                         "ratio": "" if r is None else round(r, 4),
                         "size": "" if size is None else f"{size[0]}x{size[1]}"})
        if len(got) < 2:
            continue
        maj = Counter(round(v[0], 2) for v in got.values()).most_common(1)[0][0]
        odd = [m for m, v in got.items() if abs(v[0] - maj) > args.tol]
        if odd:
            flagged[pid] = (odd, maj, {m: round(v[0], 2) for m, v in got.items()})

    cr, _ = ratio(meta[pairs[0]][0]) if pairs else (None, None)
    print(f"{len(pairs)} pair(s) x {len(methods)} method(s)\n")
    if missing:
        print(f"missing outputs in {len(missing)} pair(s):")
        for pid, ms in list(missing.items())[:10]:
            print(f"  {pid:18s} {', '.join(ms)}")
        print()

    if flagged:
        print(f"{len(flagged)} pair(s) with disagreeing output shapes -- these are "
              f"excluded:\n")
        print(f"  {'pair':20s} {'content':>8s} {'style':>7s}  {'deviating':<22s} ratios")
        for pid, (odd, maj, allr) in sorted(flagged.items()):
            c, s, _ = meta[pid]
            print(f"  {pid:20s} {ratio(c)[0]:>8.2f} {ratio(s)[0]:>7.2f}  "
                  f"{','.join(odd):<22s} {allr}")
    else:
        print("no pair has disagreeing output shapes; nothing to exclude.")

    by_attr = Counter(meta[p][2] for p in pairs if p not in flagged)
    print(f"\nclean pairs: {len(pairs) - len(flagged)}/{len(pairs)}")
    for a, n in sorted(by_attr.items(), key=lambda x: -x[1]):
        print(f"  {a:34s} {n}")
    thin = [a for a, n in by_attr.items() if n < 5]
    if thin:
        print(f"\n  note: {', '.join(thin)} now has fewer than 5 pairs; a per-dimension\n"
              f"  claim on it will not be supportable.")

    if args.out:
        o = ROOT / args.out
        o.parent.mkdir(parents=True, exist_ok=True)
        with o.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pair", "method", "ratio", "size"])
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {o.relative_to(ROOT)}")

    if args.write_clean:
        keep = [l for l in lines
                if not l.strip() or l.startswith("#")
                or l.split(",")[0].strip() not in flagged]
        hdr = [f"# Auto-generated by scripts/audit_framing.py from {args.pairs_file}.",
               f"# {len(flagged)} pair(s) removed because the methods disagreed on output",
               "# aspect ratio; a deviating output is penalised by the judge for reasons",
               "# unrelated to style. Removed: " + (", ".join(sorted(flagged)) or "none"),
               "#"]
        p = ROOT / args.write_clean
        p.write_text("\n".join(hdr + keep) + "\n", encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)} "
              f"({len(pairs) - len(flagged)} pairs)")


if __name__ == "__main__":
    main()
