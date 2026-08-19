#!/usr/bin/env python3
"""
Run browser for AgenticST: inspect what the agent actually did.

Serves a small web UI over the result directories: the plan and its stages in
execution order, every intermediate image inline, the preservation contract, the
DyMAG agent graph, reflections, and per-call token/model statistics.

    python scripts/dashboard.py --root . --port 8765

Then from your laptop:

    ssh -N -L 8765:localhost:8765 swwang@172.18.32.151

and open http://localhost:8765

Design notes
------------
* Standard library only. The GPU server has no general internet access, so a CDN
  stylesheet or a pip install at run time would leave you with a broken page.
  All CSS is inline and there are no external requests.
* Binds to 127.0.0.1 by default: reachable through the SSH tunnel above, not from
  the rest of the network. Pass --host 0.0.0.0 only if you know why you want that.
* Stage images are ordered by the plan, not by mtime. Retries and reflection
  appends make timestamps a misleading proxy for execution order.
"""
import argparse
import html
import json
import mimetypes
import os
import re
import time
import urllib.parse
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(".").resolve()

MARKERS = ("style_transfer_plan.json", "system_orchestration.json",
           "direct_generated_image.png")

CONTRACT_KEYS = ("preservation_contract", "similar_regions_transfer_detail",
                 "similar_region_transfer_detail", "non_style_content_to_preserve")


# --------------------------------------------------------------------------- io

def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


_scan_cache = {"key": None, "at": 0.0, "runs": []}
SCAN_TTL = 20.0  # seconds


def find_runs(root: Path, force: bool = False):
    """Locate run directories, with a short TTL cache.

    Three rglob passes over a result tree with thousands of runs takes seconds,
    and the index page would otherwise pay that on every request (including the
    browser's favicon probe).
    """
    now = time.time()
    if (not force and _scan_cache["key"] == str(root)
            and now - _scan_cache["at"] < SCAN_TTL):
        return _scan_cache["runs"]
    seen = {}
    for marker in MARKERS:
        for hit in root.rglob(marker):
            d = hit.parent
            if d not in seen:
                seen[d] = d.stat().st_mtime
    runs = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    _scan_cache.update(key=str(root), at=now, runs=runs)
    return runs


def model_name_of(raw: str) -> str:
    """str(llm) is a whole langchain repr; pull the model id out of it."""
    if not raw:
        return "?"
    m = re.search(r"model_name='([^']+)'|model='([^']+)'", raw)
    if m:
        return m.group(1) or m.group(2)
    return raw.split("\n")[0][:60]


def collect_calls(run: Path):
    """Per-LLM-call stats from logs/*.json. input_tokens/output_tokens are dicts."""
    calls = []
    for f in sorted((run / "logs").glob("*.json")):
        d = load_json(f)
        if not isinstance(d, dict):
            continue

        def tok(key):
            v = d.get(key)
            if isinstance(v, dict):
                return int(v.get("total") or 0), int(v.get("images_count") or 0)
            return (int(v) if isinstance(v, (int, float)) else 0), 0

        tin, imgs_in = tok("input_tokens")
        tout, imgs_out = tok("output_tokens")
        calls.append({
            "name": f.stem.replace("_llm_call_report", ""),
            "timestamp": d.get("timestamp", ""),
            "duration": d.get("duration_seconds", 0),
            "retries": d.get("retry_count", 0),
            "in": tin, "out": tout,
            "total": int(d.get("total_tokens") or (tin + tout)),
            "imgs_in": imgs_in, "imgs_out": imgs_out,
            "model": model_name_of(str(d.get("model", ""))),
        })
    return calls


def collect_contract(run: Path):
    found = []
    for f in sorted(run.glob("analysis_*.json")) + [run / "style_transfer_analysis.json"]:
        d = load_json(f)
        if not isinstance(d, dict):
            continue
        for k in CONTRACT_KEYS:
            if d.get(k):
                found.append((f.name, k, d[k]))
    return found


def run_summary(run: Path):
    plan = load_json(run / "style_transfer_plan.json") or {}
    calls = collect_calls(run)
    refl = sorted(run.glob("reflection_*_prompt.txt"))
    satisfied = None
    summ = run / "final_reflection_summary.txt"
    if summ.exists():
        m = re.search(r"Final Satisfaction:\s*(\w+)",
                      summ.read_text(encoding="utf-8", errors="ignore"))
        if m:
            satisfied = m.group(1).lower() == "true"
    return {
        "path": str(run.relative_to(ROOT)),
        "name": run.name,
        "mtime": datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "stages": len(plan.get("stages", [])),
        "images": len(list(run.glob("*.png"))),
        "reflections": len(refl),
        "satisfied": satisfied,
        "calls": len(calls),
        "tokens": sum(c["total"] for c in calls),
        "seconds": round(sum(c["duration"] for c in calls), 1),
        "has_contract": bool(collect_contract(run)),
    }


# ------------------------------------------------------------------------- html

CSS = """
:root { --bg:#0f1115; --panel:#171a21; --line:#272c36; --fg:#dfe3ea; --dim:#8b93a3;
        --acc:#6aa3ff; --ok:#4ec9a0; --warn:#e0a34a; --bad:#e06a6a; }
@media (prefers-color-scheme: light) {
  :root { --bg:#f7f8fa; --panel:#fff; --line:#e2e5ea; --fg:#1c2028; --dim:#6b7280;
          --acc:#2563eb; --ok:#0f8a63; --warn:#b45309; --bad:#b42318; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.55 -apple-system,
       BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }
a { color:var(--acc); text-decoration:none; } a:hover { text-decoration:underline; }
header { padding:14px 22px; border-bottom:1px solid var(--line); position:sticky; top:0;
         background:var(--bg); z-index:5; display:flex; gap:16px; align-items:baseline; }
h1 { font-size:16px; margin:0; font-weight:650; }
main { padding:22px; max-width:1500px; margin:0 auto; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
        padding:16px 18px; margin-bottom:18px; }
.card > h2 { font-size:13px; margin:0 0 12px; text-transform:uppercase;
             letter-spacing:.06em; color:var(--dim); font-weight:600; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
        vertical-align:top; }
th { color:var(--dim); font-weight:600; white-space:nowrap; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.wrap { overflow-x:auto; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:14px; }
figure { margin:0; }
figure img { width:100%; border:1px solid var(--line); border-radius:8px;
             background:#fff; display:block; }
figcaption { font-size:12px; color:var(--dim); margin-top:6px; word-break:break-all; }
pre { background:var(--bg); border:1px solid var(--line); border-radius:8px;
      padding:11px 13px; overflow-x:auto; font-size:12.5px; line-height:1.5;
      font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap;
      word-break:break-word; margin:0; }
details { border:1px solid var(--line); border-radius:8px; margin-bottom:9px;
          background:var(--bg); }
summary { cursor:pointer; padding:9px 13px; font-size:13px; font-weight:550;
          user-select:none; }
details[open] summary { border-bottom:1px solid var(--line); }
details > div { padding:11px 13px; }
.stage { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:18px;
         padding:15px 0; border-bottom:1px solid var(--line); }
.stage:last-child { border-bottom:none; }
@media (max-width:900px) { .stage { grid-template-columns:1fr; } }
.pill { display:inline-block; padding:1.5px 8px; border-radius:20px; font-size:11.5px;
        border:1px solid var(--line); color:var(--dim); margin-right:5px; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .bad { color:var(--bad); }
.dim { color:var(--dim); }
.kv { display:flex; gap:26px; flex-wrap:wrap; font-size:13px; }
.kv div span { color:var(--dim); margin-right:6px; }
.empty { color:var(--dim); font-style:italic; }
"""


def page(title, body):
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<header><h1><a href="/">AgenticST runs</a></h1>
<span class="dim">{html.escape(str(ROOT))}</span></header>
<main>{body}</main></body></html>"""


def img_tag(path: Path, caption: str):
    src = "/file?p=" + urllib.parse.quote(str(path.relative_to(ROOT)))
    return (f'<figure><a href="{src}" target="_blank"><img src="{src}" loading="lazy">'
            f'</a><figcaption>{html.escape(caption)}</figcaption></figure>')


def render_index(root: Path, limit: int = 200, offset: int = 0):
    all_runs = find_runs(root)
    runs = all_runs[offset:offset + limit]
    if not runs:
        return page("AgenticST runs",
                    '<div class="card"><p class="empty">No run directories found under '
                    f'{html.escape(str(root))}. A run directory is one containing '
                    'style_transfer_plan.json, system_orchestration.json or '
                    'direct_generated_image.png.</p></div>')
    rows = []
    for d, _ in runs:
        s = run_summary(d)
        sat = ('<span class="ok">yes</span>' if s["satisfied"]
               else '<span class="warn">no</span>' if s["satisfied"] is False
               else '<span class="dim">-</span>')
        contract = ('<span class="ok">yes</span>' if s["has_contract"]
                    else '<span class="bad">missing</span>')
        rows.append(
            f'<tr><td><a href="/run?p={urllib.parse.quote(s["path"])}">'
            f'{html.escape(s["name"])}</a><div class="dim" style="font-size:11.5px">'
            f'{html.escape(s["path"])}</div></td>'
            f'<td class="dim">{s["mtime"]}</td>'
            f'<td class="num">{s["stages"]}</td><td class="num">{s["images"]}</td>'
            f'<td class="num">{s["reflections"]}</td><td>{sat}</td><td>{contract}</td>'
            f'<td class="num">{s["calls"]}</td><td class="num">{s["tokens"]:,}</td>'
            f'<td class="num">{s["seconds"]}s</td></tr>')
    total = len(all_runs)
    nav = []
    if offset:
        nav.append(f'<a href="/?offset={max(0, offset - limit)}&limit={limit}">newer</a>')
    if offset + limit < total:
        nav.append(f'<a href="/?offset={offset + limit}&limit={limit}">older</a>')
    navhtml = ' &middot; '.join(nav)
    return page("AgenticST runs", f"""<div class="card">
<h2>{total} run(s) &middot; showing {offset + 1}-{min(offset + limit, total)} {navhtml}</h2>
<input id="q" placeholder="filter by name..." oninput="filt()"
 style="width:100%;padding:8px 11px;margin-bottom:12px;border-radius:8px;
 border:1px solid var(--line);background:var(--bg);color:var(--fg);font-size:13px">
<script>function filt(){{var v=document.getElementById('q').value.toLowerCase();
document.querySelectorAll('tbody tr').forEach(function(r){{
r.style.display=r.textContent.toLowerCase().includes(v)?'':'none';}});}}</script>
<div class="wrap"><table><tbody>
<tr><th>run</th><th>modified</th><th class="num">stages</th><th class="num">images</th>
<th class="num">refl</th><th>satisfied</th><th>contract</th><th class="num">calls</th>
<th class="num">tokens</th><th class="num">llm time</th></tr>
{''.join(rows)}</tbody></table></div></div>""")


def render_run(run: Path):
    rel = run.relative_to(ROOT)
    plan = load_json(run / "style_transfer_plan.json") or {}
    orch = load_json(run / "system_orchestration.json") or {}
    stages = plan.get("stages", [])
    calls = collect_calls(run)
    parts = []

    # inputs -------------------------------------------------------------
    inputs = [p for p in sorted(run.glob("*.jpg")) + sorted(run.glob("*.jpeg"))]
    named = {"content_image", "style_image", "image_1", "image_2"}
    inputs += [p for p in sorted(run.glob("*.png")) if p.stem in named]
    if inputs:
        parts.append('<div class="card"><h2>Inputs</h2><div class="grid">'
                     + "".join(img_tag(p, p.name) for p in inputs) + "</div></div>")

    # preservation contract ----------------------------------------------
    contract = collect_contract(run)
    if contract:
        blocks = []
        for fname, key, val in contract:
            txt = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, indent=2)
            blocks.append(f'<details open><summary>{html.escape(fname)} &middot; '
                          f'<code>{html.escape(key)}</code></summary>'
                          f'<div><pre>{html.escape(txt)}</pre></div></details>')
        parts.append('<div class="card"><h2>Preservation contract '
                     '<span class="dim">(emitted before generation)</span></h2>'
                     + "".join(blocks) + "</div>")
    else:
        parts.append('<div class="card"><h2>Preservation contract</h2>'
                     '<p class="empty">Not present in this run. If you expect one, the '
                     'analysis agent may have dropped it under the response-length cap '
                     '(raise limited_response_length in the config).</p></div>')

    # agent graph ---------------------------------------------------------
    if orch:
        rows = "".join(
            f'<tr><td>{html.escape(str(a.get("agent_name","")))}</td>'
            f'<td class="num">{a.get("temperature","")}</td>'
            f'<td class="dim">{html.escape(", ".join(a.get("dependencies") or []) or "-")}</td>'
            f'<td class="dim">{html.escape(", ".join(a.get("required_image_tags") or []) or "-")}</td>'
            f'</tr>' for a in orch.get("agent_graph", []))
        crit = str(orch.get("result_critique_criteria", "")).strip()
        parts.append(
            f'<div class="card"><h2>DyMAG &middot; {html.escape(str(orch.get("task_type","")))}</h2>'
            f'<div class="wrap"><table><tr><th>agent</th><th class="num">temp</th>'
            f'<th>depends on</th><th>images</th></tr>{rows}</table></div>'
            + (f'<details style="margin-top:12px"><summary>Critique specification</summary>'
               f'<div><pre>{html.escape(crit)}</pre></div></details>' if crit else "")
            + "</div>")

    # plan stages, in execution order -------------------------------------
    if stages:
        items = []
        for i, st in enumerate(stages, 1):
            tag = st.get("generated_image_tag", "")
            img = run / f"{tag}.png"
            thumb = img_tag(img, tag) if img.exists() else \
                f'<p class="empty">no image for tag “{html.escape(tag)}”</p>'
            reqs = "".join(f'<span class="pill">{html.escape(t)}</span>'
                           for t in (st.get("required_image_tags") or []))
            items.append(
                f'<div class="stage"><div><b>Stage {i}. '
                f'{html.escape(str(st.get("stage_name","")))}</b>'
                f'<div style="margin:7px 0">{reqs}'
                f'<span class="pill">T={st.get("gen_temperature")}</span></div>'
                f'<pre>{html.escape(str(st.get("text_prompt","")))}</pre></div>'
                f'<div>{thumb}</div></div>')
        parts.append(f'<div class="card"><h2>Plan &middot; {len(stages)} stage(s)</h2>'
                     + "".join(items) + "</div>")

    # any other generated images -------------------------------------------
    known = {f'{s.get("generated_image_tag","")}.png' for s in stages} | {p.name for p in inputs}
    extra = [p for p in sorted(run.glob("*.png")) if p.name not in known]
    if extra:
        parts.append('<div class="card"><h2>Other images</h2><div class="grid">'
                     + "".join(img_tag(p, p.name) for p in extra) + "</div></div>")

    # reflections ----------------------------------------------------------
    refl = sorted(run.glob("reflection_*_prompt.txt"))
    summ = run / "final_reflection_summary.txt"
    if refl or summ.exists():
        blocks = []
        if summ.exists():
            blocks.append('<details open><summary>final_reflection_summary.txt</summary>'
                          f'<div><pre>{html.escape(summ.read_text(encoding="utf-8", errors="ignore"))}'
                          "</pre></div></details>")
        for f in refl:
            blocks.append(f'<details><summary>{html.escape(f.name)}</summary><div><pre>'
                          f'{html.escape(f.read_text(encoding="utf-8", errors="ignore")[:20000])}'
                          "</pre></div></details>")
        parts.append(f'<div class="card"><h2>Reflection &middot; {len(refl)} round(s)</h2>'
                     + "".join(blocks) + "</div>")

    # token / model accounting ---------------------------------------------
    if calls:
        def retry_cell(n):
            return "-" if not n else f'<span class="warn">{n}</span>'

        rows = "".join(
            f'<tr><td>{html.escape(c["name"])}</td><td class="dim">{html.escape(c["model"])}</td>'
            f'<td class="num">{c["in"]:,}</td><td class="num">{c["out"]:,}</td>'
            f'<td class="num">{c["total"]:,}</td>'
            f'<td class="num">{c["imgs_in"]}/{c["imgs_out"]}</td>'
            f'<td class="num">{c["duration"]}s</td>'
            f'<td class="num">{retry_cell(c["retries"])}</td>'
            f'</tr>' for c in calls)
        ti, to = sum(c["in"] for c in calls), sum(c["out"] for c in calls)
        rt = sum(c["retries"] for c in calls)
        parts.append(
            f'<div class="card"><h2>Model calls &middot; {len(calls)}</h2>'
            f'<div class="kv" style="margin-bottom:12px">'
            f'<div><span>input</span>{ti:,}</div><div><span>output</span>{to:,}</div>'
            f'<div><span>total</span>{ti + to:,}</div>'
            f'<div><span>llm time</span>{round(sum(c["duration"] for c in calls),1)}s</div>'
            f'<div><span>retries</span>{rt}</div></div>'
            f'<div class="wrap"><table><tr><th>call</th><th>model</th><th class="num">in</th>'
            f'<th class="num">out</th><th class="num">total</th><th class="num">imgs i/o</th>'
            f'<th class="num">time</th><th class="num">retries</th></tr>{rows}</table></div>'
            '<p class="dim" style="font-size:12px;margin:10px 0 0">Token counts are the '
            'pipeline\'s own estimates from src/utils/llm_helper.py, not provider-reported '
            'usage. Treat them as consistent across runs, not as billing truth.</p></div>')

    # raw files -------------------------------------------------------------
    files = sorted(p for p in run.iterdir() if p.is_file())
    links = "".join(
        f'<tr><td><a href="/file?p={urllib.parse.quote(str(p.relative_to(ROOT)))}" '
        f'target="_blank">{html.escape(p.name)}</a></td>'
        f'<td class="num dim">{p.stat().st_size:,} B</td></tr>' for p in files)
    parts.append('<div class="card"><h2>Files</h2><div class="wrap"><table>'
                 f'{links}</table></div></div>')

    return page(run.name, f'<p class="dim" style="margin-top:0">{html.escape(str(rel))}</p>'
                + "".join(parts))


# ------------------------------------------------------------------------ server

class Handler(BaseHTTPRequestHandler):
    server_version = "AgenticSTDashboard/1.0"

    def log_message(self, fmt, *a):
        pass  # keep the console usable for the run you are watching

    def _send(self, code, body: bytes, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve(self, rel: str) -> Path:
        """Resolve a request path inside ROOT, refusing anything that escapes it."""
        p = (ROOT / urllib.parse.unquote(rel)).resolve()
        if not str(p).startswith(str(ROOT)):
            raise PermissionError("path escapes the served root")
        return p

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/":
                limit = int(q.get("limit", ["200"])[0])
                offset = int(q.get("offset", ["0"])[0])
                self._send(200, render_index(ROOT, limit, offset).encode())
            elif u.path == "/favicon.ico":
                self._send(204, b"")
            elif u.path == "/run":
                run = self._resolve(q.get("p", [""])[0])
                if not run.is_dir():
                    self._send(404, b"no such run")
                    return
                self._send(200, render_run(run).encode())
            elif u.path == "/file":
                p = self._resolve(q.get("p", [""])[0])
                if not p.is_file():
                    self._send(404, b"no such file")
                    return
                ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                if ctype.startswith("text/") or p.suffix in (".json", ".txt", ".md", ".yaml"):
                    ctype = "text/plain; charset=utf-8"
                self._send(200, p.read_bytes(), ctype)
            elif u.path == "/api/runs":
                data = [run_summary(d) for d, _ in find_runs(ROOT)]
                self._send(200, json.dumps(data, indent=2).encode(),
                           "application/json; charset=utf-8")
            else:
                self._send(404, b"not found")
        except PermissionError as e:
            self._send(403, str(e).encode())
        except Exception as e:  # never take the server down over one bad page
            self._send(500, f"<pre>{html.escape(repr(e))}</pre>".encode())


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="directory to scan for runs")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="default is loopback only; reach it through an SSH tunnel")
    args = ap.parse_args()

    ROOT = Path(args.root).resolve()
    n = len(find_runs(ROOT))
    print(f"serving {ROOT}  ({n} run(s) found)")
    print(f"  http://{args.host}:{args.port}")
    if args.host == "127.0.0.1":
        print(f"  from your laptop:  ssh -N -L {args.port}:localhost:{args.port} "
              f"{os.environ.get('USER','user')}@<server>")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
