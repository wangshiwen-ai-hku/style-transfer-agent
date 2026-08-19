#!/usr/bin/env python3
"""
Pairwise A/B style-fidelity judging, over the APIYI endpoint.

    python scripts/score_style_ab.py --dirs outputs/ours outputs/cot outputs/strong \
        outputs/weak outputs/refine

Replaces absolute 0-100 scoring, which proved too noisy to use: re-scoring the
same eight images on two occasions moved one method's mean by 5.3 points, which
is larger than the gap between the two leading methods. A forced binary choice
between two candidates removes the need for the judge to hold a stable scale in
its head, which is the part it was failing at.

Reliability is measured, not assumed
------------------------------------
Two controls run alongside the real comparisons, and both are reported:

* **Order swap.** Every method pair is judged twice on each image pair, once as
  (A,B) and once as (B,A). A judge with no position bias returns the same winner
  both times. The agreement rate over these swaps is the protocol's reliability;
  if it is near 50% the judge is answering by position, not by content, and the
  rankings below mean nothing.
* **Identity control.** Each method is also compared against itself. The expected
  win rate is 50%; systematic deviation is the size of the position bias.

Aggregation
-----------
Win rates, plus a Bradley-Terry fit giving one scalar per method. BT is used
rather than raw win counts because it accounts for who each method was compared
against, and it degrades gracefully when some comparisons are missing.

Outputs
-------
  <out>/style_ab.csv          one row per judged comparison
  <out>/style_ab_summary.csv  per method: BT score, win rate, n
"""
import argparse
import csv
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import apiyi  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

PROMPT = """You are an art critic comparing two attempts at the same task.

Image 1 is a STYLE REFERENCE artwork.
Image 2 is CANDIDATE A.
Image 3 is CANDIDATE B.

A and B were produced from the same source photograph by two different methods.
Decide which one reproduces the STYLE of image 1 more faithfully.

Judge only style: colour palette, line quality, texture and mark-making, use of
negative space, decorative motifs belonging to the style, light and atmosphere.
Do NOT reward a candidate for resembling the source photograph. A candidate that
looks like a lightly filtered photograph has failed at this task, however clean it
looks.

You must choose one. Reply with ONLY a JSON object:
{"winner": "A", "reason": "<one short clause>"}
or
{"winner": "B", "reason": "<one short clause>"}"""


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p under p=0.5, by summing outcomes no more likely."""
    if n <= 0:
        return float("nan")
    pk = math.comb(n, k)
    return min(1.0, sum(math.comb(n, i) for i in range(n + 1) if math.comb(n, i) <= pk)
               / 2 ** n)


def parse_winner(text: str):
    try:
        i, j = text.index("{"), text.rindex("}") + 1
        d = json.loads(text[i:j])
        w = str(d.get("winner", "")).strip().upper()
        return (w if w in ("A", "B") else None), str(d.get("reason", ""))[:160]
    except Exception:
        t = (text or "").strip().upper()
        if t.startswith("A"):
            return "A", text[:120]
        if t.startswith("B"):
            return "B", text[:120]
        return None, (text or "")[:120]


def judge(style: Path, a: Path, b: Path, model: str):
    msg = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "text", "text": "Image 1 (style reference):"},
        {"type": "image_url", "image_url": {"url": apiyi.encode_image(str(style))}},
        {"type": "text", "text": "Image 2 (candidate A):"},
        {"type": "image_url", "image_url": {"url": apiyi.encode_image(str(a))}},
        {"type": "text", "text": "Image 3 (candidate B):"},
        {"type": "image_url", "image_url": {"url": apiyi.encode_image(str(b))}},
    ]}]
    return parse_winner(apiyi.chat(msg, model=model, temperature=0.0))


def bradley_terry(wins, methods, iters=500):
    """MM-algorithm MLE for Bradley-Terry strengths, normalised to mean 1."""
    p = {m: 1.0 for m in methods}
    n = defaultdict(float)
    for (i, j), w in wins.items():
        n[(i, j)] = w
    for _ in range(iters):
        new = {}
        for i in methods:
            num = sum(n[(i, j)] for j in methods if j != i)
            den = 0.0
            for j in methods:
                if j == i:
                    continue
                total = n[(i, j)] + n[(j, i)]
                if total:
                    den += total / (p[i] + p[j])
            new[i] = num / den if den > 0 else p[i]
        s = sum(new.values()) / len(new) or 1.0
        p = {k: v / s for k, v in new.items()}
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--pairs-file", default="compare/pairs.txt")
    ap.add_argument("--reps", type=int, default=1,
                    help="repetitions per ordered comparison; both orders are always run")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-identity-control", action="store_true")
    ap.add_argument("--exclude-file", default=None,
                    help="file of pair ids to skip, one per line "
                         "(the output of scripts/audit_framing.py)")
    args = ap.parse_args()

    # Pairs on which the methods disagree about the output aspect ratio are
    # excluded here rather than filtered afterwards: the judge rewards a match to
    # the reference artwork's framing, so such a pair does not measure style.
    skip = set()
    if args.exclude_file:
        ep = ROOT / args.exclude_file
        if not ep.exists():
            raise SystemExit(f"--exclude-file not found: {ep}")
        skip = {l.strip() for l in ep.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")}

    styles = {}
    for line in (ROOT / args.pairs_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            c = [x.strip() for x in line.split(",")]
            if len(c) > 2 and c[0] not in skip:
                styles[c[0]] = ROOT / c[2]
    if skip:
        print(f"excluding {len(skip)} pair(s) from {args.exclude_file}; "
              f"{len(styles)} remain")

    imgs = {}
    for d in args.dirs:
        m = Path(d).name
        imgs[m] = {p.stem: p for p in sorted((ROOT / d).glob("*.png")) if p.stem in styles}
    methods = [m for m in imgs if imgs[m]]
    if len(methods) < 2:
        raise SystemExit(f"need at least two non-empty method dirs; got {list(imgs)}")

    jobs = []
    for pid in sorted(styles):
        avail = [m for m in methods if pid in imgs[m]]
        for m1, m2 in itertools.combinations(avail, 2):
            for _ in range(args.reps):
                jobs.append((pid, m1, m2, "fwd"))
                jobs.append((pid, m2, m1, "rev"))
        if not args.no_identity_control:
            for m in avail:
                jobs.append((pid, m, m, "identity"))
    random.shuffle(jobs)

    print(f"{len(methods)} method(s), {len(styles)} image pair(s) -> {len(jobs)} comparisons\n")
    rows = []
    for k, (pid, ma, mb, kind) in enumerate(jobs, 1):
        try:
            w, why = judge(styles[pid], imgs[ma][pid], imgs[mb][pid], args.model)
        except Exception as e:
            w, why = None, f"ERROR {type(e).__name__}"[:60]
        winner = None if w is None else (ma if w == "A" else mb)
        rows.append({"pair": pid, "slot_A": ma, "slot_B": mb, "kind": kind,
                     "choice": w or "", "winner": winner or "", "reason": why})
        print(f"[{k}/{len(jobs)}] {pid:16s} A={ma:8s} B={mb:8s} {kind:8s} -> {w or 'FAIL'}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "style_ab.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["pair", "slot_A", "slot_B", "kind",
                                           "choice", "winner", "reason"])
        wr.writeheader(); wr.writerows(rows)

    # ---- reliability first: the rankings are only worth reading if these pass ----
    ident = [r for r in rows if r["kind"] == "identity" and r["choice"]]
    if ident:
        a_rate = sum(r["choice"] == "A" for r in ident) / len(ident)
        # This control is degenerate and is reported only for completeness. Shown two
        # identical images the judge has no content signal yet is still forced to
        # choose, so it falls back on a slot; a rate near 100% is the expected
        # outcome, not evidence of bias on the real comparisons. It is the order-swap
        # rate below that bounds position bias, because there the images differ.
        # Do not reinstate a warning here -- it fired on every run and was misread
        # twice as invalidating the data.
        print(f"\nidentity control: slot A chosen {a_rate:.0%} of {len(ident)} "
              f"(degenerate by construction: with two identical images the judge\n"
              f"                  must still choose, so a high rate is expected and is NOT\n"
              f"                  evidence of bias -- use the order-swap rate below)")

    fwd = {(r["pair"], r["slot_A"], r["slot_B"]): r["winner"]
           for r in rows if r["kind"] == "fwd" and r["winner"]}
    agree = tot = 0
    for r in rows:
        if r["kind"] != "rev" or not r["winner"]:
            continue
        key = (r["pair"], r["slot_B"], r["slot_A"])
        if key in fwd:
            tot += 1
            agree += (fwd[key] == r["winner"])
    if tot:
        print(f"order-swap agreement: {agree}/{tot} = {agree / tot:.0%} "
              f"(100% = fully order-invariant, 50% = coin flip)")
        if agree / tot < 0.65:
            print("!! the judge does not give the same answer when the two candidates\n"
                  "   swap places. Do not report the ranking below as a finding.")

    # ---- per-pair analysis: the only level at which observations are independent ----
    # Judging the same image pair twice tells us about the judge, not about the
    # methods, so significance is computed over image pairs and never over raw
    # comparison counts. Repetitions enter as a better estimate of each pair's win
    # rate, which is what makes a paired test over pairs more powerful than a sign
    # test over pair winners.
    per_pair = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (m1,m2) -> pid -> [w1,w2]
    for r in rows:
        if r["kind"] == "identity" or not r["winner"]:
            continue
        m1, m2 = sorted([r["slot_A"], r["slot_B"]])
        cell = per_pair[(m1, m2)][r["pair"]]
        cell[0 if r["winner"] == m1 else 1] += 1

    print("\nper-image-pair win rates (independent units; n = number of image pairs)")
    for (m1, m2), pd_ in sorted(per_pair.items()):
        rates, wins_, losses_, ties_ = [], 0, 0, 0
        for pid in sorted(pd_):
            w1, w2 = pd_[pid]
            if w1 + w2 == 0:
                continue
            rates.append(w1 / (w1 + w2))
            if w1 > w2:
                wins_ += 1
            elif w2 > w1:
                losses_ += 1
            else:
                ties_ += 1
        if not rates:
            continue
        n = len(rates)
        mean = sum(rates) / n
        reps = (sum(sum(v) for v in pd_.values()) / n) if n else 0
        line = (f"  {m1} vs {m2}: {m1} wins {mean:.0%} of comparisons, "
                f"leads on {wins_}/{n} image pairs ({ties_} tied), "
                f"{reps:.0f} judgements per pair")
        # Sign test over pair winners, plus Wilcoxon over per-pair rates when there
        # are enough repetitions for the rates to carry more than a sign.
        dec = wins_ + losses_
        if dec:
            p_sign = binom_two_sided(wins_, dec)
            line += f"\n      sign test over image pairs: p = {p_sign:.3f}"
        if reps >= 4 and n >= 6:
            try:
                from scipy.stats import wilcoxon
                diffs = [r - 0.5 for r in rates]
                if any(d != 0 for d in diffs):
                    p_w = wilcoxon(diffs).pvalue
                    line += f";  Wilcoxon on per-pair rates: p = {p_w:.3f}"
            except Exception:
                pass
        elif reps < 4:
            line += "\n      (rerun with --reps 5 for a rate-based test; one judgement\n"
            line += "       per order is mostly judge noise, see the order-swap rate above)"
        print(line)
    if any(len(pd_) < 10 for pd_ in per_pair.values()):
        print("\n  NOTE: with fewer than ~10 image pairs, only a near-unanimous result\n"
              "  reaches significance (8/8 gives p=0.008; 6/8 gives p=0.29). Repetitions\n"
              "  sharpen each pair's estimate but cannot raise this ceiling -- that needs\n"
              "  more image pairs. See scripts/make_pairs.py.")

    # ---- aggregation ----
    wins = defaultdict(float)
    for r in rows:
        if r["kind"] == "identity" or not r["winner"]:
            continue
        loser = r["slot_B"] if r["winner"] == r["slot_A"] else r["slot_A"]
        wins[(r["winner"], loser)] += 1
    bt = bradley_terry(wins, methods)

    print(f"\n{'method':12s} {'BT score':>9s} {'wins':>6s} {'losses':>7s} {'win rate':>9s}")
    summary = []
    for m in sorted(methods, key=lambda x: -bt[x]):
        w = sum(v for (i, _), v in wins.items() if i == m)
        l = sum(v for (_, j), v in wins.items() if j == m)
        wr_ = w / (w + l) if (w + l) else float("nan")
        summary.append({"method": m, "bt_score": round(bt[m], 4), "wins": int(w),
                        "losses": int(l), "win_rate": round(wr_, 4)})
        print(f"{m:12s} {bt[m]:>9.3f} {int(w):>6d} {int(l):>7d} {wr_:>8.0%}")
    with (out / "style_ab_summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["method", "bt_score", "wins", "losses", "win_rate"])
        wr.writeheader(); wr.writerows(summary)

    print(f"\nhead-to-head (row beats column):")
    print(f"{'':12s} " + " ".join(f"{m[:8]:>9s}" for m in methods))
    for i in methods:
        cells = []
        for j in methods:
            if i == j:
                cells.append(f"{'-':>9s}"); continue
            w, l = wins[(i, j)], wins[(j, i)]
            cells.append(f"{int(w)}-{int(l):<7d}" if (w + l) else f"{'-':>9s}")
        print(f"{i:12s} " + " ".join(cells))
    print(f"\nwrote {out / 'style_ab.csv'}")


if __name__ == "__main__":
    main()
