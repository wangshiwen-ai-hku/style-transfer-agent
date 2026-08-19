#!/usr/bin/env python3
"""
Side-by-side viewer for the revision's comparison rounds, over a forwarded port.

    python scripts/compare_dashboard.py --port 8766
    # then, locally:  ssh -N -L 8766:localhost:8766 swwang@172.18.32.151

The existing dashboard.py shows one agent run at a time -- prompts, logs,
reflections. This one answers a different question: across a set of methods or
arms, what do the outputs actually look like on the same pair, and did the judge
agree with what I can see?

Every A/B verdict is shown next to the images it was about, including the judge's
stated reason. A number in a summary table is not checkable; a 62% win rate whose
per-image verdicts you can read is. Several of this revision's conclusions changed
after looking at which pairs drove an aggregate, so the per-pair view is the point
rather than a convenience.

Discovers comparison sets automatically: any directory holding <pair_id>.png files
is an arm, sibling arms form a set, and any style_ab.csv found beside them supplies
the verdicts.
"""
import argparse
import csv
import html
import io
import mimetypes
import re
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
PAIR_RE = re.compile(r"^sty_[A-Za-z0-9]+_cnt_[A-Za-z0-9]+$")

CSS = """
:root{--bg:#0f1115;--fg:#e6e8eb;--dim:#9aa3ad;--card:#171a21;--line:#242a33;
      --win:#2ea043;--lose:#8b949e;--warn:#d29922}
@media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#1f2328;--dim:#656d76;
      --card:#f6f8fa;--line:#d0d7de;--win:#1a7f37;--lose:#8b949e;--warn:#9a6700}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:inherit}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
       padding:12px 20px;z-index:10}
h1{font-size:17px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
.sub{color:var(--dim);font-size:12px}
main{padding:16px 20px 60px}
.grid{display:grid;gap:10px;margin-bottom:6px}
.cell{background:var(--card);border:1px solid var(--line);border-radius:8px;
      padding:8px;min-width:0}
.cell img{width:100%;height:auto;display:block;border-radius:4px;cursor:zoom-in}
.cap{font-size:11px;color:var(--dim);margin-top:6px;word-break:break-word}
.name{font-weight:600;font-size:12px;margin-bottom:6px}
.rec{font-size:11px;color:var(--dim)}
.w{color:var(--win);font-weight:600}
.l{color:var(--lose)}
table{border-collapse:collapse;font-size:13px;margin:6px 0 18px}
th,td{border:1px solid var(--line);padding:5px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:var(--card);font-weight:600}
.reasons{font-size:11px;color:var(--dim);margin:2px 0 18px;padding-left:14px}
.reasons li{margin:2px 0}
.note{background:var(--card);border-left:3px solid var(--warn);padding:8px 12px;
      margin:10px 0;font-size:12px;border-radius:0 6px 6px 0}
.pill{display:inline-block;background:var(--card);border:1px solid var(--line);
      border-radius:12px;padding:1px 9px;font-size:11px;margin-right:6px}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;
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


def find_arms(root: Path):
    """Any directory containing <pair_id>.png files is an arm of some comparison."""
    arms = {}
    skip = {".git", "__pycache__", "node_modules", "logs", ".venv"}
    for d in root.rglob("*"):
        if not d.is_dir() or any(p in skip for p in d.parts):
            continue
        pids = sorted(p.stem for p in d.glob("*.png") if PAIR_RE.match(p.stem))
        if pids:
            arms[d] = pids
    return arms


def group_sets(arms):
    """Sibling arms under one parent are a comparison set."""
    sets = defaultdict(dict)
    for d, pids in arms.items():
        sets[d.parent][d.name] = pids
    return {k: v for k, v in sets.items() if v}


def load_verdicts(parent: Path, root: Path):
    """Collect every style_ab.csv that could describe this set.

    The scorer writes its csv to --out, which is often a sibling of the arms
    rather than the parent itself, so search nearby rather than requiring one
    fixed layout.
    """
    cands = list(parent.glob("style_ab.csv")) + list(parent.glob("*/style_ab.csv"))
    if parent.parent != parent:
        cands += list(parent.parent.glob("*/style_ab.csv"))
    rows = []
    for c in dict.fromkeys(cands):
        try:
            with c.open(encoding="utf-8") as f:
                rows += [r for r in csv.DictReader(f)]
        except Exception:
            continue
    return rows


def records(rows, methods):
    """Per-method W-L, and per-pair verdicts with the judge's reason."""
    wl = defaultdict(lambda: [0, 0])
    per_pair = defaultdict(list)
    ms = set(methods)
    for r in rows:
        if r.get("kind") == "identity" or not r.get("winner"):
            continue
        a, b = r.get("slot_A"), r.get("slot_B")
        if a not in ms or b not in ms:
            continue
        w = r["winner"]
        loser = b if w == a else a
        wl[w][0] += 1
        wl[loser][1] += 1
        per_pair[r["pair"]].append((w, loser, r.get("reason", "")))
    return wl, per_pair


def find_inputs(root: Path, pid: str):
    """Locate the content and style images named by a pair id."""
    m = re.match(r"^sty_(.+)_cnt_(.+)$", pid)
    if not m:
        return []
    s, c = m.group(1), m.group(2)
    out = []
    for sub, stem, label in (("content", c, "content"), ("style", s, "style")):
        for base in ("compare", "data"):
            hit = [p for e in (".jpg", ".jpeg", ".png")
                   for p in [root / base / sub / f"{stem}{e}"] if p.exists()]
            if hit:
                out.append((hit[0], label))
                break
    return out


def render_set(root: Path, parent: Path, arms: dict, rows):
    methods = sorted(arms)
    wl, per_pair = records(rows, methods)
    rel = parent.relative_to(root) if parent != root else Path(".")
    h = [f"<h2>{esc(rel)}</h2>"]
    h.append("<div class='sub'>" + " ".join(
        f"<span class='pill'>{esc(m)} &middot; {len(arms[m])} img</span>" for m in methods)
        + "</div>")

    if wl:
        h.append("<table><tr><th>method</th><th>wins</th><th>losses</th>"
                 "<th>win rate</th></tr>")
        for m in sorted(methods, key=lambda x: -(wl[x][0] / (sum(wl[x]) or 1))):
            w, l = wl[m]
            rate = f"{w / (w + l):.0%}" if w + l else "&mdash;"
            h.append(f"<tr><td>{esc(m)}</td><td>{w}</td><td>{l}</td><td>{rate}</td></tr>")
        h.append("</table>")
        h.append("<div class='note'>Win counts pool both presentation orders of every "
                 "image pair, so they are not independent observations. Judge the "
                 "verdicts below against the images; treat the rates as a summary, "
                 "not as evidence.</div>")

    pids = sorted({p for v in arms.values() for p in v})
    for pid in pids:
        ins = find_inputs(root, pid)
        cols = len(ins) + len(methods)
        h.append(f"<div class='sub' style='margin-top:14px'><b>{esc(pid)}</b></div>")
        h.append(f"<div class='grid' style='grid-template-columns:repeat({cols},1fr)'>")
        for p, label in ins:
            h.append(f"<div class='cell'><div class='name'>{esc(label)}</div>"
                     f"<img loading='lazy' src='/f/{esc(p.relative_to(root))}'></div>")
        for m in methods:
            f = parent / m / f"{pid}.png"
            if not f.exists():
                h.append(f"<div class='cell'><div class='name'>{esc(m)}</div>"
                         "<div class='cap'>missing</div></div>")
                continue
            vs = [v for v in per_pair.get(pid, []) if m in (v[0], v[1])]
            won = sum(1 for v in vs if v[0] == m)
            rec = (f"<span class='w'>{won}W</span> <span class='l'>{len(vs) - won}L</span>"
                   if vs else "")
            h.append(f"<div class='cell'><div class='name'>{esc(m)} "
                     f"<span class='rec'>{rec}</span></div>"
                     f"<img loading='lazy' src='/f/{esc(f.relative_to(root))}'></div>")
        h.append("</div>")
        if per_pair.get(pid):
            h.append("<ul class='reasons'>")
            for w, l, why in per_pair[pid][:12]:
                h.append(f"<li><b>{esc(w)}</b> &gt; {esc(l)} &mdash; {esc(why)}</li>")
            h.append("</ul>")
    return "\n".join(h)


def build_page(root: Path, only=None):
    sets = group_sets(find_arms(root))
    sets = {k: v for k, v in sets.items() if len(v) >= 2}
    if only:
        sets = {k: v for k, v in sets.items() if only in str(k)}
    body = [f"<header><h1>AgenticST &mdash; comparison rounds</h1>"
            f"<div class='sub'>{len(sets)} set(s) under {esc(root)}</div></header><main>"]
    if not sets:
        body.append("<div class='note'>No comparison set found. A set is a directory "
                    "whose subdirectories each contain &lt;pair_id&gt;.png files "
                    "(e.g. outputs/ours, outputs/cot).</div>")
    for parent in sorted(sets):
        body.append(render_set(root, parent, sets[parent], load_verdicts(parent, root)))
    body.append("</main><div id='zoom'><img></div>")
    return (f"<!doctype html><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>AgenticST comparisons</title><style>{CSS}</style>"
            + "".join(body) + f"<script>{JS}</script>")


def make_handler(root: Path, only):
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
                           build_page(root, only).encode("utf-8"))
                return
            if path.startswith("/f/"):
                try:
                    # Resolve and confine to root: the server binds to localhost and
                    # is reached through an ssh tunnel, but a traversal bug would
                    # still expose the whole filesystem to anything on the host.
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
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--filter", default=None,
                    help="only show sets whose path contains this string")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    sets = {k: v for k, v in group_sets(find_arms(root)).items() if len(v) >= 2}
    print(f"root: {root}")
    for p in sorted(sets):
        print(f"  {p.relative_to(root) if p != root else '.'}: {', '.join(sorted(sets[p]))}")
    if not sets:
        print("  (no comparison set found -- nothing will render)")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(root, args.filter))
    print(f"\nserving on 127.0.0.1:{args.port}\n"
          f"from your laptop:\n"
          f"  ssh -N -L {args.port}:localhost:{args.port} swwang@172.18.32.151\n"
          f"  open http://localhost:{args.port}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
