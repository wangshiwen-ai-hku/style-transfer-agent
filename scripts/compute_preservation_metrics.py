#!/usr/bin/env python3
"""
Preservation-aware metrics for the TVCG revision (Sec. 4.3, Table 4).

    # metrics for the contract ablation
    python scripts/compute_preservation_metrics.py --index ablation_contract/index.csv

    # metrics for any directory of <pair_id>.png outputs
    python scripts/compute_preservation_metrics.py --dir outputs/ours --method ours

Measures what LPIPS/CLIP-I/FID cannot: whether the subject in the output is the
same subject, posed and looking the same way, as the subject in the content
image. Reviewer 3 put it exactly right -- those metrics "cannot distinguish a
stylized version of the same person from a stylized version of a similar but
regenerated person".

Metrics
-------
  EAR-Acc   eyelid state agreement (open/closed) with the content image.
            This is the specific artifact the reviewers pointed at.
  LMD       facial landmark drift: mean landmark displacement after similarity
            alignment, normalised by inter-ocular distance. Scale/translation
            invariant, so re-framing is not counted as identity change.
  GAZE      absolute difference in horizontal iris offset, a proxy for gaze.
  ID-Sim    face-embedding cosine similarity. SECONDARY, and only meaningful if
            the validation below passes -- recognition embeddings are trained on
            photographs and go out of distribution on strong stylization.
  EdgeChamfer  symmetric Chamfer distance between content and output edge maps,
            in % of the image diagonal (lower is better). Edge IoU was tried first
            and proved useless: photographs carry dense texture edges while line-art
            outputs carry sparse ones, so IoU sat at 0.02-0.05 for every pair
            regardless of how faithful the structure was.

Every metric is reported per pair, not only as a mean: with a handful of cases a
mean hides exactly the variation that matters.

Dependencies:  bash setup_server.sh --preservation     (no torch required)
  mediapipe + assets/models/face_landmarker.task (landmarks/EAR/gaze),
  insightface + onnxruntime (ID-Sim), scikit-image + scipy (edges).
A missing landmarker is fatal, not a warning: those metrics are the point of this
script, and degrading silently produces a CSV that looks complete but answers
nothing. Pass --no-face to opt out deliberately.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# MediaPipe FaceMesh indices.
L_EYE = [33, 160, 158, 133, 153, 144]     # outer, top x2, inner, bottom x2
R_EYE = [362, 385, 387, 263, 373, 380]
L_IRIS, R_IRIS = 468, 473
EAR_OPEN = 0.18   # below this the eye reads as closed


MODEL = ROOT / "assets/models/face_landmarker.task"


def _mediapipe():
    """Face landmarker, across both mediapipe API generations.

    mediapipe 1.0 removed mp.solutions entirely; only the Tasks API remains, and
    it needs an explicit .task model file. We ship that file in assets/models/
    because the GPU server cannot reach Google's model CDN. The legacy
    solutions API is still tried first so older environments keep working.

    Failures here are fatal rather than warnings: EAR, landmark drift and gaze are
    the metrics this whole script exists for, and an earlier version degraded
    silently, writing a CSV that looked complete but was missing every column that
    mattered.
    """
    try:  # legacy API (mediapipe <= 0.10)
        import mediapipe as mp
        if hasattr(mp, "solutions"):
            fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.3)
            return ("legacy", fm)
    except Exception:
        pass

    try:  # tasks API (mediapipe >= 0.10.9, required on 1.x)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        if not MODEL.exists():
            raise FileNotFoundError(
                f"{MODEL} not found. Download it once on a machine with internet:\n"
                "  curl -L -o assets/models/face_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task\n"
                "then rsync it to the server.")
        # Two detectors at different confidences. Flat line-art faces are a
        # documented weak spot: sty_112_cnt_23 has an unmistakable cartoon face
        # with eyes and a smile that the 0.3 detector misses entirely. Without the
        # permissive retry, "face not detected" would conflate a destroyed face
        # (Kandinsky) with a face the detector simply cannot parse -- two things
        # that mean opposite about preservation.
        dets = []
        for conf in (0.3, 0.1):
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1, min_face_detection_confidence=conf,
                min_face_presence_confidence=conf, min_tracking_confidence=conf)
            dets.append(vision.FaceLandmarker.create_from_options(opts))
        return ("tasks", dets)
    except Exception as e:
        raise SystemExit(
            f"\n!! no usable mediapipe face landmarker: {e}\n"
            "   EAR / landmark drift / gaze cannot be computed without it, and those\n"
            "   are the metrics that answer the reviewers. Install with\n"
            "     bash setup_server.sh --preservation\n"
            "   Re-run with --no-face to compute only the remaining metrics.")


def landmarks(mesh, path: Path):
    """(N,2) landmarks in pixels, or None if no face is detected."""
    if mesh is None:
        return None
    kind, det = mesh
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    if kind == "legacy":
        res = det.process(img)
        if not res.multi_face_landmarks:
            return None
        pts = res.multi_face_landmarks[0].landmark
    else:
        import mediapipe as mp
        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        pts = None
        for d in det:
            res = d.detect(mpimg)
            if res.face_landmarks:
                pts = res.face_landmarks[0]
                break
        if pts is None:
            return None
    return np.array([[p.x * w, p.y * h] for p in pts])



# MediaPipe FaceMesh region groupings, for decomposing drift by facial part.
# The point of the decomposition: "the face changed" and "the eyes changed while
# everything else held" are very different claims, and only the second one is
# compatible with a declared, localised expression transfer.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
EYE_REGION = sorted(set(L_EYE + R_EYE + [L_IRIS, R_IRIS,
                    246, 161, 159, 157, 173, 155, 154, 145, 163, 7,
                    398, 384, 386, 388, 466, 263, 249, 390, 374, 381, 382]))
BROW_REGION = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
NOSE_REGION = [1, 2, 4, 5, 6, 19, 94, 168, 197, 195, 5, 45, 275, 220, 440]
MOUTH_REGION = [61, 291, 0, 17, 13, 14, 78, 308, 82, 87, 312, 317, 40, 270, 39, 269]
REGIONS = {"oval": FACE_OVAL, "eyes": EYE_REGION, "brows": BROW_REGION,
           "nose": NOSE_REGION, "mouth": MOUTH_REGION}


def lmd_by_region(lm_c, lm_o):
    """Per-region landmark drift, aligned on the face oval.

    Alignment uses only the face contour, which a legitimate expression transfer
    should not move. Aligning on all landmarks instead would let a large eye
    displacement partly absorb itself into the fitted transform, understating
    exactly the effect we want to measure.
    """
    T = similarity_align_subset(lm_c, lm_o, FACE_OVAL)
    iod = np.linalg.norm(lm_o[L_IRIS] - lm_o[R_IRIS]) + 1e-8
    out = {}
    for name, idx in REGIONS.items():
        idx = [i for i in idx if i < len(lm_o)]
        out[name] = float(np.linalg.norm(T[idx] - lm_o[idx], axis=1).mean() / iod)
    return out


def similarity_align_subset(src, dst, idx):
    """Fit the similarity transform on `idx` only, then apply it to all points."""
    idx = [i for i in idx if i < len(src) and i < len(dst)]
    a, b = src[idx], dst[idx]
    mu_a, mu_b = a.mean(0), b.mean(0)
    a0, b0 = a - mu_a, b - mu_b
    U, S, Vt = np.linalg.svd(b0.T @ a0 / len(a))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
        S = S.copy(); S[-1] *= -1
    scale = S.sum() / ((a0 ** 2).sum() / len(a) + 1e-12)
    return scale * (R @ src.T).T + (mu_b - scale * (R @ mu_a.T).T)


def ear(lm, idx):
    """Eye aspect ratio: (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
    p = lm[idx]
    a = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    b = 2.0 * np.linalg.norm(p[0] - p[3]) + 1e-8
    return float(a / b)


def eyes_open(lm):
    return (ear(lm, L_EYE) + ear(lm, R_EYE)) / 2.0 >= EAR_OPEN


def similarity_align(src, dst):
    """Umeyama similarity transform of src onto dst (rotation+scale+translation).

    Aligning before measuring drift is what makes the metric mean "the face
    changed" instead of "the crop changed": a stylization that re-frames or
    rescales the subject should not be charged as identity drift.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
        S = S.copy(); S[-1] *= -1
    var = (s0 ** 2).sum() / len(src) + 1e-12
    scale = S.sum() / var
    return (scale * (R @ src.T).T + (mu_d - scale * (R @ mu_s.T).T))


def lmd(lm_c, lm_o):
    """Normalised landmark drift after alignment."""
    aligned = similarity_align(lm_c, lm_o)
    iod = np.linalg.norm(lm_o[L_IRIS] - lm_o[R_IRIS]) + 1e-8
    return float(np.linalg.norm(aligned - lm_o, axis=1).mean() / iod)


def gaze_offset(lm):
    """Horizontal iris position within the eye, in [0,1]; ~0.5 is centred."""
    out = []
    for iris, eye in ((L_IRIS, L_EYE), (R_IRIS, R_EYE)):
        inner, outer = lm[eye[0]], lm[eye[3]]
        span = np.linalg.norm(outer - inner) + 1e-8
        out.append(float(np.linalg.norm(lm[iris] - inner) / span))
    return sum(out) / 2.0


def _arcface():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        print(f"  ! insightface unavailable ({e}); ID-Sim disabled", file=sys.stderr)
        return None


def id_embedding(app, path: Path):
    if app is None:
        return None
    faces = app.get(np.array(Image.open(path).convert("RGB"))[:, :, ::-1])
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    v = f.normed_embedding
    return v / (np.linalg.norm(v) + 1e-8)


def edge_chamfer(a: Path, b: Path, size=512):
    """Symmetric Chamfer distance between edge maps, in % of image diagonal.

    Edge IoU was tried first and is useless here: a photograph has dense texture
    edges (hair, foliage) while a line-art output has sparse clean ones, so the
    two maps barely overlap even when the structure is faithfully preserved --
    measured IoU sat at 0.02-0.05 for every pair, i.e. noise. Chamfer asks the
    better question: for each edge pixel, how far away is the nearest edge in the
    other image. That tolerates density and thickness mismatch while still
    punishing displaced structure. Lower is better.
    """
    try:
        from skimage.feature import canny
        from skimage.color import rgb2gray
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return None

    def edges(p):
        im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        return canny(rgb2gray(np.array(im) / 255.0), sigma=2.0)

    ea, eb = edges(a), edges(b)
    if ea.sum() == 0 or eb.sum() == 0:
        return None
    da, db = distance_transform_edt(~ea), distance_transform_edt(~eb)
    d = 0.5 * (db[ea].mean() + da[eb].mean())
    return float(100.0 * d / (size * np.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", help="index.csv from run_contract_ablation.py")
    ap.add_argument("--dir", help="directory of <pair_id>.png outputs")
    ap.add_argument("--method", default="method", help="label when using --dir")
    ap.add_argument("--pairs-file", default="compare/pairs.txt")
    ap.add_argument("--out", default=None, help="output CSV (default alongside input)")
    ap.add_argument("--no-id", action="store_true", help="skip ID-Sim")
    ap.add_argument("--no-face", action="store_true",
                    help="skip landmark metrics instead of failing when mediapipe is unusable")
    args = ap.parse_args()
    if not (args.index or args.dir):
        raise SystemExit("give --index or --dir")

    contents, styles, style_eyes = {}, {}, {}
    for line in (ROOT / args.pairs_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            c = [x.strip() for x in line.split(",")]
            contents[c[0]] = ROOT / c[1]
            if len(c) > 2:
                styles[c[0]] = ROOT / c[2]
            # Optional key=value annotations in any trailing column. style_eyes_open
            # is a manual annotation of the style reference: the landmarker is trained
            # on photographs and detects nothing in ink-wash or woodblock artwork, so
            # for those references the choice is between a hand annotation a reader can
            # check against the figure and no measurement at all.
            for tok in c[3:]:
                if tok.startswith("style_eyes_open="):
                    style_eyes[c[0]] = int(tok.split("=", 1)[1])

    items = []
    if args.index:
        for r in csv.DictReader((ROOT / args.index).open(encoding="utf-8")):
            items.append((r["arm"], r["pair"], r.get("repeat", ""), ROOT / r["image"]))
        default_out = (ROOT / args.index).with_name("preservation_metrics.csv")
    else:
        for p in sorted(Path(ROOT / args.dir).glob("*.png")):
            items.append((args.method, p.stem, "", p))
        default_out = Path(ROOT / args.dir) / "preservation_metrics.csv"

    mesh = None if args.no_face else _mediapipe()
    app = None if args.no_id else _arcface()

    rows = []
    for arm, pair, rep, img in items:
        cpath = contents.get(pair)
        spath = styles.get(pair)
        if cpath is None or not img.exists():
            print(f"  ! skipping {arm}/{pair}: missing content or output")
            continue
        lm_c, lm_o = landmarks(mesh, cpath), landmarks(mesh, img)
        lm_s = landmarks(mesh, spath) if spath and spath.exists() else None
        row = {"arm": arm, "pair": pair, "repeat": rep, "image": img.name,
               "face_detected": int(lm_o is not None)}
        if lm_c is not None and lm_o is not None:
            row.update(EAR_content_open=int(eyes_open(lm_c)),
                       EAR_output_open=int(eyes_open(lm_o)),
                       EAR_match=int(eyes_open(lm_c) == eyes_open(lm_o)),
                       LMD=round(lmd(lm_c, lm_o), 4),
                       **{f"LMD_{k}": round(v, 4)
                          for k, v in lmd_by_region(lm_c, lm_o).items()},
                       GAZE_delta=round(abs(gaze_offset(lm_c) - gaze_offset(lm_o)), 4))
        # Whose eyelid state does the output follow? Comparing only against the
        # content image cannot tell an uncontrolled regeneration apart from a
        # directed transfer: both look like "it changed". Comparing against the
        # style reference as well distinguishes them, and that distinction is the
        # whole of the disagreement with Reviewer 3.
        so, so_src = None, ""
        if pair in style_eyes:
            so, so_src = bool(style_eyes[pair]), "manual"
        elif lm_s is not None:
            so, so_src = eyes_open(lm_s), "detected"
        if so is not None and lm_o is not None and lm_c is not None:
            row["EAR_style_open"] = int(so)
            row["EAR_style_src"] = so_src
            co, oo = eyes_open(lm_c), eyes_open(lm_o)
            row["EAR_follows"] = ("both_agree" if co == so
                                  else "content" if oo == co
                                  else "style")

        if app is not None:
            ec, eo = id_embedding(app, cpath), id_embedding(app, img)
            row["ID_Sim"] = round(float(ec @ eo), 4) if (ec is not None and eo is not None) else ""
        row["EdgeChamfer"] = (lambda v: round(v, 3) if v is not None else "")(edge_chamfer(cpath, img))
        rows.append(row)
        print(f"  {arm:11s} {pair:16s} r{rep or '-'}  " +
              "  ".join(f"{k}={row[k]}" for k in
                        ("face_detected", "EAR_match", "EAR_follows", "LMD",
                         "LMD_eyes", "LMD_oval", "GAZE_delta", "ID_Sim", "EdgeChamfer")
                        if k in row))

    # mediapipe's FaceLandmarker.__del__ raises a TypeError during interpreter
    # shutdown; closing explicitly keeps that noise out of the run log.
    if mesh and mesh[0] == "tasks":
        for d in mesh[1]:
            try:
                d.close()
            except Exception:
                pass

    if not rows:
        # Be specific about which of the several possible causes it was. The first
        # version of this message blamed missing dependencies, which sent a real
        # debugging session down the wrong path: the dependencies had loaded fine
        # and the actual problem was that the output directory did not exist.
        detail = [f"scanned {len(items)} candidate item(s)"]
        if args.dir:
            d = ROOT / args.dir
            if not d.is_dir():
                detail.append(f"{d} does not exist")
            else:
                pngs = sorted(d.glob("*.png"))
                detail.append(f"{d} holds {len(pngs)} png(s)")
                unmatched = [x.stem for x in pngs if x.stem not in contents][:5]
                if unmatched:
                    detail.append("filenames do not match any pair_id in "
                                  f"{args.pairs_file}; e.g. {unmatched}")
                    detail.append(f"known pair_ids: {sorted(contents)[:5]}")
        raise SystemExit("no rows produced.\n  " + "\n  ".join(detail))
    out = Path(args.out) if args.out else default_out
    keys = sorted({k for r in rows for k in r}, key=lambda k: (
        ["arm", "pair", "repeat", "image", "face_detected"].index(k)
        if k in ["arm", "pair", "repeat", "image", "face_detected"] else 99))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")

    # per-arm summary, and the paired comparison the ablation is for
    arms = sorted({r["arm"] for r in rows})
    print(f"\n{'arm':12s} {'n':>3s} {'face%':>6s} {'EAR match':>16s} {'LMD':>8s} "
          f"{'GAZE':>7s} {'ID-Sim':>7s} {'Chamfer':>8s}")
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        num = lambda k: [r[k] for r in rs if isinstance(r.get(k), (int, float))]
        m = lambda k: f"{np.mean(num(k)):.4f}" if num(k) else "n/a"
        det = 100 * np.mean([r["face_detected"] for r in rs])
        em = num("EAR_match")
        # EAR/LMD/GAZE exist only where a face was found, so their n differs from
        # the row count. Reporting them as if they covered every pair would
        # overstate the coverage.
        print(f"{a:12s} {len(rs):>3d} {det:>5.0f}% "
              f"{(100 * np.mean(em) if em else float('nan')):>7.0f}% (n={len(em)}) "
              f"{m('LMD'):>8s} {m('GAZE_delta'):>7s} {m('ID_Sim'):>7s} {m('EdgeChamfer'):>8s}")

    # Landmark metrics only exist where a photo-trained detector can parse the
    # output. That is a domain limit, not a preservation result: a minimal
    # line-art face can be perfectly faithful and still be unparseable. Comparing
    # per-arm means computed over different subsets would therefore compare
    # different sets of pairs. Restrict to pairs measurable in EVERY arm.
    for a in arms:
        f = [r.get("EAR_follows") for r in rows
             if r["arm"] == a and r.get("EAR_follows")]
        if f:
            from collections import Counter
            print(f"  eyelid state follows, {a}: {dict(Counter(f))}")

    if len(arms) > 1:
        measurable = {p for p in {r["pair"] for r in rows}
                      if all(any(r["pair"] == p and r["arm"] == a
                                 and isinstance(r.get("LMD"), (int, float))
                                 for r in rows) for a in arms)}
        dropped = sorted({r["pair"] for r in rows} - measurable)
        if dropped:
            print(f"\nlandmark metrics unavailable in at least one arm for: "
                  f"{', '.join(dropped)}")
            print("  excluded from the paired comparison below so that every arm is "
                  "scored on the same pairs.")
    else:
        measurable = {r["pair"] for r in rows}

    if set(arms) >= {"contract", "nocontract"}:
        print("\nper-pair paired comparison (contract vs nocontract):")
        for pair in sorted(measurable):
            def agg(a, k):
                v = [r[k] for r in rows
                     if r["arm"] == a and r["pair"] == pair and isinstance(r.get(k), (int, float))]
                return np.mean(v) if v else float("nan")
            print(f"  {pair:16s} EAR-match {agg('contract','EAR_match'):.2f} vs "
                  f"{agg('nocontract','EAR_match'):.2f}   "
                  f"LMD {agg('contract','LMD'):.4f} vs {agg('nocontract','LMD'):.4f}")
        print("\nWith a few repeats these are indicative, not significant. Report the "
              "per-pair numbers and the repeat count; do not claim an effect that a "
              "sign test over this many samples cannot support.")


if __name__ == "__main__":
    main()
