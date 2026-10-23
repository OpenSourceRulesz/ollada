#!/usr/bin/env python3
"""Build the ollada shelf site from the catalog.

Reads ollada/catalog/*.yaml (the same files the runtime loads) and emits a
static site into site/ — index (hero + shelf cards) plus one page per model.
No numbers are typed here: everything comes from the recipes, so the site
cannot drift from the verified data. Deploy by pointing GitHub Pages at the
generated output (or copy site/ into docs/ on main).

Usage: python3 scripts/build_site.py [--out site]
"""
from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "ollada" / "catalog"
ASSETS = ROOT / "docs" / "assets"

CSS = """
:root { --ink:#111; --mut:#555; --line:#e5e2dc; --card:#faf9f7; --accent:#b5541d; }
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       color:var(--ink); background:#fff; line-height:1.55; }
a { color:var(--ink); }
.wrap { max-width:880px; margin:0 auto; padding:0 20px; }
header { border-bottom:1px solid var(--line); }
.nav { display:flex; align-items:center; gap:12px; padding:14px 0; }
.nav img { width:40px; height:40px; display:block; }
.nav .brand { font-weight:700; font-size:20px; text-decoration:none; }
.nav nav { margin-left:auto; display:flex; gap:18px; }
.nav nav a { text-decoration:none; color:var(--mut); font-size:15px; }
.nav nav a:hover { color:var(--ink); }
.hero { text-align:center; padding:48px 0 30px; }
.hero img { width:110px; }
.hero h1 { font-size:40px; margin:14px 0 8px; letter-spacing:-0.5px; }
.hero p { color:var(--mut); font-size:19px; max-width:620px; margin:0 auto; }
.term { background:#111; color:#f2f2f2; border-radius:12px; padding:18px 22px;
        font-family:ui-monospace,'Cascadia Mono',Consolas,monospace; font-size:15px;
        max-width:560px; margin:28px auto; text-align:left; overflow-x:auto; }
.term .c { color:#8a8a8a; }
.term div { white-space:nowrap; }
h2.section { font-size:24px; margin:42px 0 6px; }
.sub { color:var(--mut); margin:0 0 18px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:16px; }
.card { border:1px solid var(--line); border-radius:14px; padding:18px; background:var(--card);
        text-decoration:none; display:block; }
.card:hover { border-color:#c9c4ba; }
.card h3 { margin:0 0 4px; font-size:19px; font-family:ui-monospace,Consolas,monospace; }
.card p { color:var(--mut); font-size:14.5px; margin:0 0 12px; }
.badges { display:flex; flex-wrap:wrap; gap:6px; }
.badge { font-size:12px; border:1px solid var(--line); border-radius:999px; padding:2px 9px;
         background:#fff; color:var(--mut); }
.badge.ver { color:var(--accent); border-color:#e8cdbb; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:18px 0; }
.stat { border:1px solid var(--line); border-radius:12px; padding:12px 14px; }
.stat .k { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); }
.stat .v { font-size:19px; font-weight:650; margin-top:2px; }
table { border-collapse:collapse; width:100%; margin:10px 0 20px; font-size:15px; }
td { border-top:1px solid var(--line); padding:7px 10px 7px 0; vertical-align:top; }
td.k { color:var(--mut); width:210px; }
code { font-family:ui-monospace,'Cascadia Mono',Consolas,monospace; font-size:.92em;
       background:#f3f1ec; padding:1px 6px; border-radius:6px; }
.notes { background:var(--card); border:1px solid var(--line); border-radius:12px;
         padding:16px 18px; font-size:14.5px; color:#333; }
footer { border-top:1px solid var(--line); margin-top:54px; padding:22px 0 34px;
         color:var(--mut); font-size:14px; }
.modelhead { padding:36px 0 6px; }
.modelhead h1 { font-family:ui-monospace,Consolas,monospace; font-size:34px; margin:0 0 6px; }
.modelhead p { color:var(--mut); font-size:17px; margin:0; max-width:640px; }
"""

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{root}style.css">
<link rel="icon" href="{root}assets/logo-badge.png">
</head>
<body>
<header><div class="wrap nav">
  <a href="{root}index.html"><img src="{root}assets/logo-badge.png" alt="Ollada — home"></a>
  <nav>
    <a href="{root}index.html#shelf">Library</a>
    <a href="https://github.com/OpenSourceRulesz/ollada">GitHub</a>
    <a href="https://pypi.org/project/ollada/">PyPI</a>
  </nav>
</div></header>
<main class="wrap">
{body}
</main>
<footer><div class="wrap">
  Ollada — the shelf for open-source diffusion language models. Local-first, $0 project:
  no telemetry, no hosted registry — models come from Hugging Face, recipes are verified here.
  Every number on this site was measured by a real-load run on the hardware named.
</div></footer>
</body>
</html>
"""


def esc(x) -> str:
    return html.escape(str(x))


def slug(name: str) -> str:
    return name.replace(":", "-")


def gen_defaults(m: dict) -> list[tuple[str, str]]:
    g = m.get("generate", {})
    rows = []
    labels = [("steps", "Denoising steps"), ("denoising_steps", "Denoising steps"),
              ("gen_length", "Canvas length (tokens)"), ("block_length", "Block length"),
              ("remasking", "Remasking"), ("temperature", "Temperature"),
              ("confidence_threshold", "Confidence threshold"), ("threshold", "Threshold")]
    for key, label in labels:
        if key in g and g[key] is not None:
            rows.append((label, str(g[key])))
    rows.append(("Default quantization", str(m.get("quant_default", ""))))
    rows.append(("Weights size (bf16)", f"{m.get('size_gb_bf16')} GB"))
    rows.append(("License", str(m.get("license", ""))))
    rows.append(("Source", str(m.get("source", ""))))
    return rows


def card(m: dict) -> str:
    v = m.get("verified", {})
    return f"""
<a class="card" href="models/{slug(m['name'])}.html">
  <h3>{esc(m['name'])}</h3>
  <p>{esc(m.get('description', ''))}</p>
  <div class="badges">
    <span class="badge">{esc(m.get('task', ''))}</span>
    <span class="badge">{esc(m.get('license', ''))}</span>
    <span class="badge">{esc(m.get('size_gb_bf16'))} GB</span>
    <span class="badge ver">verified · {esc(f"{v['vram_gb']} GB VRAM" if v.get('vram_gb') else "CPU (no GPU)")} · {esc(v.get('tokens_per_sec'))} tok/s</span>
  </div>
</a>"""


def index_body(models: list[dict]) -> str:
    cards = "\n".join(card(m) for m in models)
    return f"""
<div class="hero">
  <h1>Diffusion models, verified and ready to run</h1>
  <p><strong>Ollada</strong> is the shelf for open <strong>diffusion</strong> models —
     curated names, verified recipes, one command, running locally on your GPU — or, for the SDAR models, your CPU.
     The same idea <strong>Ollama</strong> proved for autoregressive models,
     rebuilt for diffusion.</p>
  <div class="term">
    <div><span class="c">$</span> pip install ollada</div>
    <div><span class="c">$</span> ollada pull diffcoder:8b</div>
    <div><span class="c">$</span> ollada run diffcoder:8b</div>
    <div><span class="c">$</span> ollada serve <span class="c"># local API on http://localhost:11435</span></div>
  </div>
</div>

<h2 class="section" id="shelf">Library</h2>
<p class="sub">A model earns its short name only after a real-load verification run —
measured VRAM, speed, and quality on named hardware. No vendor benchmarks.
<a href="library.html">Compare all models in the full catalog →</a></p>
<div class="grid">{cards}
</div>

<h2 class="section">Why a shelf?</h2>
<p class="sub">Diffusion models on Hugging Face ship with incompatible generation code,
undocumented defaults, and no promise that the recipe on the card actually runs.
Ollada wraps each model family in a tested adapter, bakes the verified defaults
(steps, block size, quantization) into a recipe, and serves it behind an
Ollama-shaped API — so tools that already speak Ollama can point at port 11435.</p>
"""


def model_body(m: dict) -> str:
    v = m.get("verified", {})
    stats = [
        ("Verified on", v.get("hardware", "")),
        ("Peak VRAM", f"{v['vram_gb']} GB" if v.get("vram_gb") else "— (CPU run)"),
        ("Speed", f"{v.get('tokens_per_sec')} canvas tok/s"),
    ]
    if v.get("load_seconds"):
        stats.append(("Load time", f"{v['load_seconds']} s"))
    stat_html = "\n".join(
        f'<div class="stat"><div class="k">{esc(k)}</div><div class="v">{esc(val)}</div></div>'
        for k, val in stats)
    rows = "\n".join(f'<tr><td class="k">{esc(k)}</td><td>{esc(val)}</td></tr>'
                     for k, val in gen_defaults(m))
    src = esc(m.get("source", ""))
    return f"""
<div class="modelhead">
  <h1>{esc(m['name'])}</h1>
  <p>{esc(m.get('description', ''))}</p>
</div>
<div class="term" style="margin:20px 0;">
  <div><span class="c">$</span> ollada pull {esc(m['name'])}</div>
  <div><span class="c">$</span> ollada run {esc(m['name'])}</div>
</div>
<div class="stats">{stat_html}
</div>
<p class="sub">Verified {esc(v.get('date', ''))} through Ollada's own code. Load time
includes the first weight download where noted in the verification notes —
cached loads are faster. {esc("This model's verification is a CPU run (see the notes); a GPU verification has not been run for it." if "CPU" in v.get("hardware", "") else "Desktop verification (RTX 5060 Ti) is the pre-launch re-check; T4 numbers are the conservative floor.")}</p>
<h2 class="section">Recipe defaults</h2>
<table>{rows}
</table>
<p>Weights: <a href="https://huggingface.co/{src}">{src}</a> (pulled from Hugging Face —
Ollada hosts no registry).</p>
<h2 class="section">Verification notes</h2>
<div class="notes">{esc(v.get('notes', ''))}</div>
"""


def library_body(models: list[dict]) -> str:
    rows = []
    for m in models:
        v = m.get("verified", {})
        g = m.get("generate", {})
        rows.append(f"""<tr>
  <td><a href="models/{slug(m['name'])}.html"><code>{esc(m['name'])}</code></a></td>
  <td>{esc(m.get('task', ''))}</td>
  <td>{esc(m.get('size_gb_bf16'))} GB</td>
  <td>{esc(f"{v['vram_gb']} GB" if v.get('vram_gb') else "— (CPU)")}</td>
  <td>{esc(v.get('tokens_per_sec'))}</td>
  <td>{esc(m.get('quant_default', ''))}</td>
  <td>{esc(v.get('date', ''))}</td>
  <td>{esc(m.get('license', ''))}</td>
</tr>""")
    pulls = "\n".join(
        f'    <div><span class="c">$</span> ollada pull {esc(m["name"])}</div>'
        for m in models)
    return f"""
<div class="modelhead">
  <h1 style="font-family:inherit;">Model catalog</h1>
  <p>Every model on the shelf, with the numbers Ollada measured itself.
     One page per model carries the full recipe and verification notes.</p>
</div>
<h2 class="section">The shelf</h2>
<table>
<tr><td class="k">Model</td><td class="k">Task</td><td class="k">Weights (bf16)</td>
    <td class="k">Peak VRAM, measured</td><td class="k">Speed, canvas tok/s</td>
    <td class="k">Default quant</td><td class="k">Verified</td><td class="k">License</td></tr>
{"".join(rows)}
</table>
<p class="sub">VRAM and speed were measured through Ollada's own code on the hardware
named on each model's page (GPU shelf: Colab Tesla T4, 2026-10-21; sdar:1.7b: a
laptop CPU, 2026-10-23 — single-run samples, treated as measured floors, not
laws). Desktop verification on the RTX 5060 Ti is the pre-launch re-check.
Weights come from Hugging Face — Ollada hosts no registry.</p>
<h2 class="section">Pull them</h2>
<div class="term" style="margin:20px 0;">
{pulls}
</div>
"""


LIBRARY_MD = """# Ollada library

The shelf: every model Ollada has verified through its own code. Numbers are
measured on the hardware named — no vendor benchmarks.

| Model | Task | Weights (bf16) | Peak VRAM, measured | Speed (canvas tok/s) | Default quant | Verified | License |
|---|---|---|---|---|---|---|---|
{rows}

GPU numbers are from a Colab Tesla T4 (2026-10-21) and are the conservative
floor; desktop verification (RTX 5060 Ti) is the pre-launch re-check.
`sdar:1.7b` was verified on a laptop CPU (2026-10-23) — no GPU was used, so
it has no VRAM figure. Weights come from Hugging Face — Ollada hosts no
registry.

Full recipes and verification notes live on each model's site page
(https://opensourcerulesz.github.io/ollada/models/).
"""


def library_md(models: list[dict]) -> str:
    rows = []
    for m in models:
        v = m.get("verified", {})
        peak = f"{v['vram_gb']} GB" if v.get("vram_gb") else "— (CPU)"
        rows.append(
            f"| `{m['name']}` | {m.get('task', '')} | {m.get('size_gb_bf16')} GB "
            f"| {peak} | {v.get('tokens_per_sec')} "
            f"| {m.get('quant_default', '')} | {v.get('date', '')} | {m.get('license', '')} |")
    return LIBRARY_MD.format(rows="\n".join(rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "site"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)

    models = []
    for path in sorted(CATALOG.glob("*.yaml")):
        with open(path) as f:
            models.append(yaml.safe_load(f))
    models.sort(key=lambda m: m["name"])

    (out / "style.css").write_text(CSS.strip() + "\n")
    shutil.copy(ASSETS / "logo-badge.png", out / "assets" / "logo-badge.png")

    (out / "index.html").write_text(PAGE.format(
        title="Ollada — the shelf for open-source diffusion language models",
        root="", body=index_body(models)))
    (out / "library.html").write_text(PAGE.format(
        title="Model catalog · Ollada", root="", body=library_body(models)))
    for m in models:
        (out / "models" / f"{slug(m['name'])}.html").write_text(PAGE.format(
            title=f"{m['name']} · Ollada", root="../", body=model_body(m)))
    (ROOT / "docs" / "LIBRARY.md").write_text(library_md(models))
    print(f"built {out}: index + library + {len(models)} model pages "
          f"({', '.join(m['name'] for m in models)}); wrote docs/LIBRARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
