#!/usr/bin/env python3
"""
Before/after viewer for the aspect-ratio re-run.

    python scripts/beforeafter_dashboard.py --port 8767

Fixing the aspect-ratio defect required regenerating four pairs, and the style
ranking inverted: AgenticST went from 8-0 to 0-8 on sty_9_cnt_22 while a
single-agent CoT baseline moved to the top overall. That is either a real
regression in our outputs or an artefact of the judge, and the two cannot be told
apart from aggregate numbers. This page puts the old and new image side by side
for each method, with the judge's verdicts and stated reasons from both rounds, so
the question can be settled by looking.

One control is built in and worth reading first: pairs that were NOT regenerated
are shown too, marked "same image". Any change in their verdicts is pure judge
variance, which calibrates how much of the change on the regenerated pairs needs
explaining at all. On sty_114_cnt_22 the record moved 7-1 to 4-4 on an identical
image.
"""
import argparse
import csv
import html
import mimetypes
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["ours", "cot", "refine", "strong", "weak"]

CSS = """
:root{--bg:#0f1115;--fg:#e6e8eb;--dim:#9aa3ad;--card:#171a21;--line:#242a33;
      --win:#2ea043;--lose:#f85149;--warn:#d29922;--same:#8b949e}
@media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#1f2328;--dim:#656d76;
      --card:#f6f8fa;--line:#d0d7de;--win:#1a7f37;--lose:#cf222e;--warn:#9a6700}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
       padding:12px 20px;z-index:10}
h1{font-size:17px;margin:0 0 4px}
h2{font-size:16px;margin:26px 0 4px}
.sub{color:var(--dim);font-size:12px}
main{padding:16px 20px 60px}
.inputs{display:flex;gap:10px;margin:8px 0 14px}
.inputs .cell{max-width:190px}
.row{display:grid;grid-template-columns:120px 1fr 1fr;gap:10px;align-items:start;
     margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid var(--line)}
.mname{font-weight:600;font-size:13px;padding-top:6px}
.cell{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.cell img{width:100%;height:auto;display:block;border-radius:4px;cursor:zoom-in}
.tag{font-size:11px;color:var(--dim);margin-bottom:6px}
.rec{font-weight:600;font-size:12px}
.up{color:var(--win)} .down{color:var(--lose)} .flat{color:var(--same)}
.reasons{font-size:11px;color:var(--dim);margin:6px 0 0;padding-left:14px}
.reasons li{margin:1px 0}
.note{background:var(--card);border-left:3px solid var(--warn);padding:8px 12px;
      margin:8px 0 16px;font-size:12px;border-radius:0 6px 6px 0}
.badge{display:inline-block;border:1px solid var(--line);border-radius:10px;
       padding:0 8px;font-size:11px;color:var(--dim);margin-left:8px}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.93);display:none;
      align-items:center;justify-content:center;z-index:100;cursor:zoom-out}
#zoom img{max-width:96vw;max-height:96vh}
"""

JS = """
document.addEventListener('click',e=>{
  if(e.target.tagName==='IMG'&&e.target.closest('.cell')){
    const z=document.getElementById('zoom');
    z.querySelector('img').src=e.target.src;z.style.display='flex';
  } else if(e.target.closest('#zoom')){document.getElementById('zoom').style.display='none';}
});
document.addEventListener('keydown',e=>{
  if(e.key==='Escape')document.getElementById('zoom').style.display='none';});
"""


def esc(s):
    return html.escape(str(s), quote=True)


def load_verdicts(path: Path):
    """pair -> method -> [wins, losses], and pair -> method -> [reasons]."""
    wl = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    why = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return wl, why
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("kind") == "identity" or not r.get("winner"):
                continue
            a, b, w = r["slot_A"], r["slot_B"], r["winner"]
            loser = b if w == a else a
            wl[r["pair"]][w][0] += 1
            wl[r["pair"]][loser][1] += 1
            why[r["pair"]][w].append(f"beat {loser}: {r.get('reason','')}")
    return wl, why


def find_input(root: Path, sub: str, stem: str):
    for base in ("compare", "data"):
        for e in (".jpg", ".jpeg", ".png"):
            p = root / base / sub / f"{stem}{e}"
            if p.exists():
                return p
    return None


def cell(root: Path, path: Path, tag: str, rec_html=""):
    if path is None or not path.exists():
        return f"<div class='cell'><div class='tag'>{esc(tag)}</div>(missing)</div>"
    return (f"<div class='cell'><div class='tag'>{esc(tag)} {rec_html}</div>"
            f"<img loading='lazy' src='/f/{esc(path.relative_to(root))}'></div>")


def render(root: Path, old_dir: Path, pairs, wl_old, why_old, wl_new, why_new):
    h = []
    for pid in pairs:
        s, c = pid[len("sty_"):].split("_cnt_")
        regenerated = (old_dir / "ours" / f"{pid}.png").exists()
        h.append(f"<h2>{esc(pid)}"
                 + ("" if regenerated else "<span class='badge'>same image "
                                           "&mdash; any change here is judge variance</span>")
                 + "</h2>")
        h.append("<div class='inputs'>")
        for sub, stem, lab in (("content", c, "content"), ("style", s, "style")):
            h.append(cell(root, find_input(root, sub, stem), lab))
        h.append("</div>")
        for m in METHODS:
            ow, ol = wl_old[pid][m]
            nw, nl = wl_new[pid][m]
            delta = (nw - nl) - (ow - ol)
            klass = "up" if delta > 0 else ("down" if delta < 0 else "flat")
            arrow = "&uarr;" if delta > 0 else ("&darr;" if delta < 0 else "&rarr;")
            h.append("<div class='row'>")
            h.append(f"<div class='mname'>{esc(m)}<br>"
                     f"<span class='rec {klass}'>{ow}-{ol} {arrow} {nw}-{nl}</span></div>")
            oldp = old_dir / m / f"{pid}.png"
            newp = root / "outputs" / m / f"{pid}.png"
            h.append(cell(root, oldp if oldp.exists() else newp,
                          "before" if oldp.exists() else "before (not regenerated)",
                          f"<span class='rec'>{ow}W {ol}L</span>"))
            h.append(cell(root, newp, "after", f"<span class='rec'>{nw}W {nl}L</span>"))
            h.append("</div>")
            rs = [("before", why_old[pid].get(m, [])), ("after", why_new[pid].get(m, []))]
            if any(v for _, v in rs):
                h.append("<ul class='reasons'>")
                for lab, v in rs:
                    for t in v[:3]:
                        h.append(f"<li><b>{lab}</b> &mdash; {esc(t[:170])}</li>")
                h.append("</ul>")
    return "\n".join(h)


def build_page(root: Path, old_dir: Path, pairs):
    wl_old, why_old = load_verdicts(old_dir / "style_ab.csv")
    wl_new, why_new = load_verdicts(root / "outputs" / "style_ab.csv")
    body = ["<header><h1>Aspect-ratio re-run &mdash; before vs after</h1>"
            f"<div class='sub'>{len(pairs)} pair(s). W&ndash;L is that method's record "
            "on this image pair against all others, in that scoring round.</div></header><main>"]
    if not (old_dir / "style_ab.csv").exists():
        body.append("<div class='note'>No pre-fix style_ab.csv found at "
                    f"{esc(old_dir)}/style_ab.csv &mdash; only the &ldquo;after&rdquo; "
                    "records will be shown.</div>")
    body.append(render(root, old_dir, pairs, wl_old, why_old, wl_new, why_new))
    body.append("</main><div id='zoom'><img></div>")
    return ("<!doctype html><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>before vs after</title><style>{CSS}</style>"
            + "".join(body) + f"<script>{JS}</script>")


def make_handler(root: Path, old_dir: Path, pairs):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, data):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8",
                           build_page(root, old_dir, pairs).encode("utf-8"))
                return
            if path.startswith("/f/"):
                try:
                    target = (root / path[3:]).resolve()
                    target.relative_to(root.resolve())
                    data = target.read_bytes()
                except Exception:
                    self._send(404, "text/plain", b"not found")
                    return
                ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self._send(200, ctype, data)
                return
            self._send(404, "text/plain", b"not found")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--old", default="outputs_pre_ratiofix")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--pairs", nargs="*",
                    default=["sty_9_cnt_22", "sty_9_cnt_23",
                             "sty_14_cnt_22", "sty_14_cnt_23",
                             "sty_114_cnt_22"])
    args = ap.parse_args()
    root = Path(args.root).resolve()
    old_dir = root / args.old

    print(f"root: {root}\nold:  {old_dir}  (exists: {old_dir.exists()})")
    for pid in args.pairs:
        n_old = sum((old_dir / m / f"{pid}.png").exists() for m in METHODS)
        n_new = sum((root / "outputs" / m / f"{pid}.png").exists() for m in METHODS)
        print(f"  {pid:16s} before {n_old}/5  after {n_new}/5"
              + ("" if n_old else "   (not regenerated -- shown as a judge-variance control)"))

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(root, old_dir, args.pairs))
    print(f"\nserving on 127.0.0.1:{args.port}\n"
          f"  ssh -N -L {args.port}:localhost:{args.port} swwang@172.18.32.151\n"
          f"  open http://localhost:{args.port}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
