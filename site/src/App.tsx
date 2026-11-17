import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  Tabs, Headline, IconCopy, CMDS, CMD_LABELS, CMD, PROMPT, WORDS, DIFF, wrongTok, fresh, type Sim,
  SHELF, FAMILIES, status, type M, SERVE, CURL, PS, JS, PY, VENV, VENV_PS, detectOs,
} from "./shared";
import "./palette.css";

const reduced = () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

function Squares() {
  return <span className="sq" aria-hidden="true"><i /><i /><i /><i /><i /><i /></span>;
}
function Progress({ lit }: { lit: number }) {
  return <span className="sq prog" aria-hidden="true">{[0, 1, 2, 3, 4, 5].map((i) => <i key={i} className={i < lit ? "on" : ""} />)}</span>;
}
function Alert() {
  return <svg className="must-i" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9" /><path d="M12 7.5v5.5" /><path d="M12 16.5v.01" /></svg>;
}
function CopyIcon() {
  return <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></svg>;
}
function Cmd({ cmd }: { cmd: string }) {
  return (
    <div className="pill">
      <code><span className="p" aria-hidden="true">$ </span>{cmd}</code>
      <IconCopy text={cmd} />
    </div>
  );
}
function Code({ text, label }: { text: string; label: string }) {
  const [done, setDone] = useState(false);
  const copy = () => navigator.clipboard?.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1600); }, () => {});
  return (
    <div className="code">
      <pre>{text}</pre>
      <button className={`icopy${done ? " done" : ""}`} onClick={copy} aria-label={done ? "Copied" : label} title="Copy">
        {done ? <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5" /></svg> : <CopyIcon />}
      </button>
    </div>
  );
}

/* ---------- the machine picker, shared by the shelf and the table ---------- */
const HW = [
  { id: "cpu", label: "No GPU" }, { id: "6", label: "6 GB" }, { id: "8", label: "8 GB" },
  { id: "12", label: "12 GB" }, { id: "16", label: "16 GB" }, { id: "24", label: "24 GB" },
];
function Picker({ hw, setHw, label }: { hw: string; setHw: (v: string) => void; label: string }) {
  return (
    <div className="pick" role="group" aria-label={label}>
      {HW.map((o) => <button key={o.id} aria-pressed={hw === o.id} onClick={() => setHw(o.id)}>{o.label}</button>)}
    </div>
  );
}
function verdict(hw: string) {
  const n: Record<string, number> = { ok: 0, warn: 0, bad: 0, na: 0 };
  SHELF.filter((m) => !m.listed).forEach((m) => { n[status(m, hw)[0]]++; });
  const words: [string, string, string][] = hw === "cpu"
    ? [["ok", "runs", "run"], ["warn", "runs slowly", "run slowly"], ["bad", "needs a GPU", "need a GPU"]]
    : [["ok", "fits", "fit"], ["warn", "is tight", "are tight"], ["bad", "won’t fit", "won’t fit"], ["na", "has only been tested on CPU", "have only been tested on CPU"]];
  const parts = words.filter(([k]) => n[k] > 0);
  const lead = hw === "cpu" ? "With no GPU" : `With a${hw === "8" ? "n" : ""} ${hw} GB card`;
  return <>{lead}: {parts.map(([k, one, many], i) => <span key={k}><strong>{n[k]}</strong> {n[k] === 1 ? one : many}{i < parts.length - 1 ? ", " : "."}</span>)}</>;
}

/* ---------- the shelf: say which GPU you have, then every model is a bar
   measured against it. Tap a model to open its commands in place. ---------- */
const GB = 18; /* the bars' scale with no card, or a small one */
const art = (hw: string) => (hw === "8" ? "an" : "a");
function Chevron() {
  return <svg className="fd-ch" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg>;
}
function fitNote(m: M, hw: string, tone: string): ReactNode {
  if (m.listed) return null;
  if (hw === "cpu") {
    if (tone === "ok") return "Measured on a laptop CPU. No GPU needed.";
    if (tone === "warn") return "Runs on a laptop CPU, slower than on a GPU.";
    return "Needs a GPU. With no GPU, sdar:1.7b, sdar:4b and nemotron:3b run on CPU.";
  }
  if (tone === "ok") return `Fits ${art(hw)} ${hw} GB card with at least 1 GB to spare.`;
  if (tone === "warn") return `Tight on ${art(hw)} ${hw} GB card: under 1 GB to spare.`;
  if (tone === "bad") return <>Too big for {art(hw)} {hw} GB card. Ollada stops before loading it instead of crashing (<code>--force</code> overrides).</>;
  return "Only measured on a laptop CPU so far, not on a GPU.";
}
function Finder({ hw, setHw }: { hw: string; setHw: (v: string) => void }) {
  const [open, setOpen] = useState<string | null>(null);
  const [grown, setGrown] = useState(reduced);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (grown) return;
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") { setGrown(true); return; }
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { setGrown(true); io.disconnect(); } }, { threshold: 0.15 });
    io.observe(el);
    return () => io.disconnect();
  }, [grown]);
  const gpu = hw !== "cpu";
  const scale = gpu ? Math.max(GB, +hw + 2) : GB;
  const pct = (v: number) => `${(v / scale) * 100}%`;
  const rows = [...SHELF].sort((a, b) => Number(!!a.listed) - Number(!!b.listed) || a.peak - b.peak);
  return (
    <div className="win finder" ref={ref}>
      <div className="fd-ask">
        <h2 id="gpu-h">What GPU do you have?</h2>
        <p>Pick its memory. Each bar is a model, measured on real hardware.</p>
        <div className="gpus" role="group" aria-labelledby="gpu-h">
          {HW.map((o) => <button key={o.id} aria-pressed={hw === o.id} onClick={() => setHw(o.id)}>{o.label}</button>)}
        </div>
        <details className="unsure">
          <summary>Not sure how much memory it has?</summary>
          <p>On Windows, open Task Manager, go to Performance, pick your GPU and read “Dedicated GPU memory”. On an NVIDIA card, <code>nvidia-smi</code> shows it too.</p>
        </details>
      </div>
      <div className="fd-sum">
        <p aria-live="polite">{verdict(hw)}</p>
        <p className="fd-tip">Tap a model for its commands.</p>
      </div>
      <div className="fd-grid fd-scale" aria-hidden="true">
        <span className="fd-track">{gpu && <span className={`fd-cardlabel${+hw / scale > 0.6 ? " end" : ""}`} style={{ left: pct(+hw) }}>your {hw} GB card</span>}</span>
      </div>
      <ul className="fd-list" aria-label="Models, smallest first">
        {rows.map((m, i) => {
          const [tone, text] = m.listed ? ["listed", "Not on shelf"] : status(m, hw);
          /* solid = runs; hatched = too big for this card; faded = not for this machine; dashed = not on the shelf */
          const look = tone === "listed" ? "listed" : tone === "bad" ? (gpu ? "over" : "faded") : tone === "na" ? "faded" : "solid";
          const isOpen = open === m.n;
          const id = `m-${m.n.replace(/[:.]/g, "-")}`;
          const note = fitNote(m, hw, tone);
          return (
            <li key={m.n} className={`fd-row fam-${m.fam} ${look}${isOpen ? " open" : ""}`}>
              <button className="fd-grid fd-hit" aria-expanded={isOpen} aria-controls={id} onClick={() => setOpen(isOpen ? null : m.n)}>
                <span className="fd-hd">
                  <span className="fd-nm"><code>{m.n}</code>{m.isNew && <span className="tag">New</span>}<small>{m.use}</small></span>
                  <span className={`fd-st st-${tone}`}><i aria-hidden="true" />{text}</span>
                </span>
                <span className="fd-track" aria-hidden="true">
                  <span className="fd-bar" style={{ width: grown ? pct(m.peak) : "0%", transitionDelay: grown ? `${i * 40}ms` : "0ms" }} />
                  {gpu && <span className="fd-line" style={{ left: pct(+hw) }} />}
                </span>
                <span className="fd-gb">{m.peak} GB</span>
                <Chevron />
              </button>
              <div className="fd-panel" id={id} role="region" aria-label={m.n} inert={!isOpen}>
                <div className="fd-in">
                  <div className="fd-body">
                    <div className="fd-info">
                      {note && <p className="fd-note"><i className={`st-${tone}`} aria-hidden="true" /><span>{note}</span></p>}
                      <dl className="fd-meta">
                        <div><dt>Good for</dt><dd>{m.use}</dd></div>
                        <div><dt>Speed</dt><dd>{m.spd === "—" ? "Not recorded" : m.spd}</dd></div>
                        <div><dt>Memory</dt><dd>{m.mem}</dd></div>
                        <div><dt>Measured on</dt><dd>{m.on}</dd></div>
                      </dl>
                    </div>
                    {m.listed
                      ? <p className="fd-off">Listed for context only. It isn’t on the shelf and isn’t verified, so there’s no pull command.</p>
                      : <div className="fd-cmds"><p>Download it, then run it:</p><Cmd cmd={`ollada pull ${m.n}`} /><Cmd cmd={`ollada run ${m.n}`} /></div>}
                  </div>
                  <details className="fd-how"><summary>How we measured this</summary><p>{m.date}</p></details>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
      <p className="fd-foot">Bar length is peak memory as measured; for nemotron:3b it’s the weights size.</p>
    </div>
  );
}

/* ---------- the same shelf, in detail ---------- */
function PullButton({ m }: { m: M }) {
  const cmd = `ollada pull ${m.n}`;
  const [done, setDone] = useState(false);
  const copy = () => navigator.clipboard?.writeText(cmd).then(() => { setDone(true); setTimeout(() => setDone(false), 1600); }, () => {});
  return <button className={`pull${done ? " done" : ""}`} onClick={copy} aria-label={done ? "Copied" : `Copy: ${cmd}`} title={cmd}>{done ? "Copied" : "Copy pull"}<CopyIcon /></button>;
}
function Detail({ hw, setHw }: { hw: string; setHw: (v: string) => void }) {
  return (
    <details className="tbl-d">
      <summary>Every model in one table</summary>
      <Picker hw={hw} setHw={setHw} label="Your machine" />
      <div className="tbl" role="table" aria-label="Models on the shelf">
        <div className="tr th" role="row">
          <span role="columnheader">Model</span><span role="columnheader">Good for</span><span role="columnheader">Peak memory</span><span role="columnheader">Speed</span><span role="columnheader">On your machine</span><span role="columnheader"><span className="sr">Pull</span></span>
        </div>
        {FAMILIES.map((f) => (
          <div role="rowgroup" key={f.id}>
            <div className="fam" role="row"><span role="cell"><i className={`dot fam-${f.id}`} aria-hidden="true" />{f.name}</span></div>
            {SHELF.filter((m) => m.fam === f.id).map((m) => {
              const [tone, text] = m.listed ? ["na", "Not verified"] : status(m, hw);
              return (
                <div className={`tr${m.listed ? " listed" : ""}`} role="row" key={m.n}>
                  <span role="cell" className="c-name"><code>{m.n}</code>{m.isNew && <span className="tag">New</span>}{m.listed && <span className="tag off">Not on shelf</span>}<small>{m.on}</small></span>
                  <span role="cell" className="c-use">{m.use}</span>
                  <span role="cell" className="c-mem">{m.mem}</span>
                  <span role="cell" className="c-spd">{m.spd}</span>
                  <span role="cell" className={`c-st st-${tone}`}><i aria-hidden="true" />{text}</span>
                  <span role="cell" className="c-pull">{m.listed ? null : <PullButton m={m} />}</span>
                  <details className="c-notes"><summary>Run notes</summary><p>{m.date}</p></details>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </details>
  );
}

/* ---------- how it writes: the live site's demo, played once when it comes into view ---------- */
function Demo() {
  const [steps, setSteps] = useState(12);
  const [sim, setSim] = useState<Sim>(fresh);
  const [running, setRunning] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const box = useRef<HTMLDivElement>(null);
  const started = useRef(false);
  const go = (n = steps) => {
    started.current = true;
    clearInterval(timer.current);
    let s = fresh();
    setSim(s); setRunning(true);
    timer.current = window.setInterval(() => {
      const step = s.step + 1;
      const idx = s.masked.map((mk, i) => (mk ? i : -1)).filter((i) => i >= 0);
      const k = Math.ceil(idx.length / (n - step + 1));
      const pick = idx.map((i) => [i, DIFF[i] + Math.random() * 0.2]).sort((a, b) => b[1] - a[1]).slice(0, k).map((p) => p[0]);
      const next: Sim = { masked: [...s.masked], shown: [...s.shown], bad: [...s.bad], step };
      pick.forEach((i) => {
        const wrong = Math.random() < (1 - DIFF[i]) * Math.min(1, (k - 1) / 6);
        next.masked[i] = false; next.bad[i] = wrong; next.shown[i] = wrong ? wrongTok(WORDS[i]) : WORDS[i];
      });
      idx.filter((i) => !pick.includes(i)).forEach((i) => {
        next.shown[i] = Math.random() < 0.6 ? (Math.random() < DIFF[i] * (step / n) ? WORDS[i] : wrongTok(WORDS[i])) : "";
      });
      s = next;
      setSim(next);
      if (next.masked.every((m) => !m)) { clearInterval(timer.current); setRunning(false); }
    }, Math.max(90, 3600 / n));
  };
  useEffect(() => () => clearInterval(timer.current), []);
  const goRef = useRef(go);
  useEffect(() => { goRef.current = go; });
  useEffect(() => {
    const el = box.current;
    if (!el || reduced() || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((es) => {
      if (!es.some((e) => e.isIntersecting) || started.current) return;
      io.disconnect(); goRef.current();
    }, { threshold: 0.6 });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  const left = sim.masked.filter(Boolean).length;
  const wrong = sim.bad.filter(Boolean).length;
  const finished = sim.step > 0 && left === 0;
  const lit = finished ? 6 : Math.floor((sim.step / steps) * 6);
  const hint = finished
    ? `Done in ${steps} steps${wrong ? `, with ${wrong} word${wrong > 1 ? "s" : ""} wrong. Try more steps.` : "."}`
    : steps <= 4 ? "Few steps: many words lock in at once and mistakes slip through."
    : steps >= 14 ? "Many steps: slower, but each word is chosen with more of the answer in view."
    : "A middle setting. Try the extremes.";
  return (
    <section id="demo" className="sec">
      <div className="wrap">
        <div className="sec-head">
          <Progress lit={lit} />
          <h2>How a diffusion model writes</h2>
          <p className="sub">A diffusion model starts with a fully masked answer and fills it in over several passes, locking in the words it is most sure of first. The trade-off is real: too few steps and quality drops. That’s why every recipe on the shelf ships with a tested step count.</p>
        </div>
        <div className="win demo-win" tabIndex={0} aria-label="Demo: arrow keys change the steps, Enter runs it" onKeyDown={(e) => {
          if (e.target !== e.currentTarget) return;
          if (e.key === "ArrowLeft" && !running) { e.preventDefault(); setSteps((v) => Math.max(2, v - 1)); setSim(fresh()); }
          else if (e.key === "ArrowRight" && !running) { e.preventDefault(); setSteps((v) => Math.min(WORDS.length, v + 1)); setSim(fresh()); }
          else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); }
        }}>
          <div className="win-bar" aria-hidden="true"><span className="dots"><i /><i /><i /></span><span className="win-title">{CMD}</span></div>
          <div className="demo">
          <div className="term" ref={box}>
            <p className="prompt"><span aria-hidden="true">$ </span>{CMD}<br /><span aria-hidden="true">&gt;&gt;&gt; </span>{PROMPT}</p>
            <p className="ans" role="img" aria-label={WORDS.join(" ")}>
              {WORDS.map((w, i) => {
                const masked = sim.masked[i] && !sim.shown[i];
                const guess = sim.masked[i] && !!sim.shown[i];
                const bad = !sim.masked[i] && sim.bad[i] && finished;
                return <span key={i} aria-hidden="true" className={`tk${masked ? " m" : guess ? " g" : bad ? " bad" : ""}`}>{masked ? w : sim.shown[i] || w}</span>;
              })}
            </p>
          </div>
          <div className="dial">
            <label htmlFor="steps">Denoising steps: <strong>{steps}</strong></label>
            <input id="steps" type="range" min={2} max={WORDS.length} value={steps} disabled={running} onChange={(e) => { setSteps(+e.target.value); setSim(fresh()); }} />
            <button className="run" onClick={() => go()}>{running ? "Restart" : finished ? "Run again" : "Run"}</button>
            <p className="stat">step {sim.step}/{steps} · {WORDS.length - left} of {WORDS.length} words committed{finished ? ` · ${wrong} wrong` : ""}</p>
            <p className="hint" aria-live="polite">{hint}</p>
            <p className="keys" aria-hidden="true"><kbd>←</kbd><kbd>→</kbd> steps <kbd>↵</kbd> run</p>
          </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------- get started ---------- */
function GetStarted() {
  const [os, setOs] = useState<"mac" | "win" | "linux">(detectOs);
  const venv = os === "win" ? VENV_PS : VENV;
  return (
    <section id="install" className="sec">
      <div className="sec-head">
        <Squares />
        <h2>Get started</h2>
        <p className="sub">Three steps, then the quick commands just work.</p>
      </div>
      <div className="slab">
        <div className="osrow">
          <span>Your system</span>
          <Tabs tabs={[{ id: "mac", label: "Mac" }, { id: "win", label: "Windows" }, { id: "linux", label: "Linux" }]} value={os} onChange={(id) => setOs(id as "mac" | "win" | "linux")} label="Operating system" />
        </div>
        <ol className="steps">
          <li className="s1">
            <h3><span className="num" aria-hidden="true">1</span>Install Python</h3>
            <p>You need Python 3.10 or newer. It only installs Ollada: the CLI, the server and the worker setup run from one bundled binary.{os === "mac" && " The Python that ships with a Mac is 3.9, which is too old."}</p>
            {os === "linux" ? (
              <p>Most Linux distributions include Python. Check with <code>python3 --version</code>; if it’s older than 3.10, install a newer one with your package manager, or get it from <a href="https://www.python.org/downloads/" target="_blank" rel="noopener">python.org</a>.</p>
            ) : (
              <p><a className="btn" href={os === "win" ? "https://www.python.org/downloads/windows/" : "https://www.python.org/downloads/macos/"} target="_blank" rel="noopener">Download Python for {os === "win" ? "Windows" : "Mac"}</a></p>
            )}
            {os === "win" && <p className="must"><Alert /><span><strong>Don’t skip:</strong> during install, tick ‘Add python.exe to PATH’.</span></p>}
            {os === "mac" && <p className="must"><Alert /><span><strong>Don’t skip:</strong> after installing, run the Install Certificates step for the version you installed, for example <code>open "/Applications/Python 3.12/Install Certificates.command"</code>. Without it, the SDAR load fails with an SSL certificate error.</span></p>}
          </li>
          <li className="s2">
            <h3><span className="num" aria-hidden="true">2</span>Open a terminal</h3>
            {os === "mac" && <p>Terminal: Applications → Utilities, or Spotlight → Terminal.</p>}
            {os === "win" && <p>PowerShell or Terminal.</p>}
            {os === "linux" && <p>Your terminal.</p>}
          </li>
          <li className="s3">
            <h3><span className="num" aria-hidden="true">3</span>Run the commands</h3>
            <p>First, make a virtual environment. Some systems refuse to install packages outside one. <strong>{os === "win" ? "Windows PowerShell" : "Mac / Linux"}:</strong></p>
            <Code text={venv} label="Copy the virtual environment commands" />
            <p><strong>Then run these one at a time, in order.</strong> They use sdar:1.7b because it’s the smallest model on the shelf and runs without a GPU.</p>
            <ol className="cmds">
              {CMDS.cpu.map((c, i) => <li key={c}><span className="lbl">{CMD_LABELS[i]}</span><Cmd cmd={c} /></li>)}
            </ol>
            <p className="fine">Check first: <code>pip show ollada</code> should say 0.3.8, and <code>ollada list</code> should show seven models including sdar:1.7b. At the <code>&gt;&gt;&gt;</code> prompt, ask: What is the capital of France? Answer with exactly one word. Expect Paris. Ctrl-C quits. NVIDIA GPU? Same steps — no extra needed; the first run picks CPU or CUDA torch for you.</p>
          </li>
        </ol>
        <details className="more-d">
          <summary>Windows PATH, PowerShell, GPU, what to expect, if it fails</summary>
          <dl>
            <dt>Windows PATH</dt>
            <dd>If <code>py</code> is not found, or you get the Microsoft Store stub (“Python was not found”), that is not real Python. Install from <a href="https://www.python.org/downloads/windows/" target="_blank" rel="noopener">python.org/downloads/windows/</a> and tick ‘Add python.exe to PATH’.</dd>
            <dt>PowerShell execution policy</dt>
            <dd>If PowerShell refuses to run the activate script, run <code>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass</code> first, then activate again.</dd>
            <dt>NVIDIA GPU</dt>
            <dd>No extra install — plain <code>pip install ollada</code>. The worker env pulls CUDA torch on an NVIDIA machine and CPU torch otherwise. DiffCoder is a tight fit on a 16 GB card — close other GPU apps first.</dd>
            <dt>What to expect</dt>
            <dd>The first run is slow twice: the model download (about 4 GB for sdar:1.7b) and a one-time worker-environment build (torch plus transformers, a few hundred MB). Later runs skip both. On CPU, short answers take seconds, long ones up to a minute.</dd>
            <dt>If it fails</dt>
            <dd><code>command not found: pip</code> or <code>python3</code> — Python is not installed or not on PATH. See step 1. <code>externally managed environment</code> — you are not inside a virtual environment. See step 3. <code>requires Python &gt;= 3.10</code> — your Python is too old. See step 1. SSL <code>CERTIFICATE_VERIFY_FAILED</code> on a Mac — run the Install Certificates step in step 1. The worker-environment build fails on the first model run — torch only ships wheels for the Pythons it supports, so a brand-new Python release can break it; use the previous Python release. Anything else — copy the whole error; that is what is needed to help.</dd>
          </dl>
        </details>
      </div>
    </section>
  );
}

/* ---------- local API ---------- */
function Api() {
  const [tab, setTab] = useState("curl");
  const bodies: Record<string, string> = { curl: CURL, ps: PS, js: JS, py: PY };
  return (
    <section id="api" className="sec">
      <div className="sec-head">
        <Squares />
        <h2>Local API</h2>
        <p className="sub">Start the server, then point any tool that speaks HTTP at it.</p>
      </div>
      <div className="api">
        <div>
          <h3>Start the server</h3>
          <Cmd cmd={SERVE} />
          <h3 className="gap">Endpoints</h3>
          <ul className="eps">
            <li><code>POST /api/chat</code><span>Chat, Ollama-shaped</span></li>
            <li><code>POST /api/generate</code><span>A single prompt</span></li>
            <li><code>POST /v1/chat/completions</code><span>Chat, OpenAI-shaped</span></li>
            <li><code>POST /v1/completions</code><span>Prompt-only, for editors</span></li>
          </ul>
        </div>
        <div>
          <div className="row">
            <h3>Call it</h3>
            <Tabs tabs={[{ id: "curl", label: "curl" }, { id: "ps", label: "PowerShell" }, { id: "js", label: "Node.js" }, { id: "py", label: "Python" }]} value={tab} onChange={setTab} label="Language" />
          </div>
          {tab === "ps" && <p className="fine top">On Windows PowerShell, <code>curl.exe</code> mangles the quoted JSON — use this instead:</p>}
          <Code text={bodies[tab]} label="Copy this example" />
          <div className="resp">
            <p className="resp-h">What comes back</p>
            <pre>{"{\n  \"model\": \"sdar:1.7b\",\n  \"message\": { \"role\": \"assistant\", \"content\": "}<span className="ok">"Paris"</span>{" },\n  \"done\": true,\n  \"eval_count\": <tokens>,\n  \"eval_duration_s\": <seconds>,\n  \"nfe\": <model calls>\n}"}</pre>
          </div>
          <p className="fine">The answer arrives whole, in one JSON object — no streaming. If the model won’t fit your machine you’ll get HTTP 409 instead of a crash (pass <code>{'options: {"force": true}'}</code> to override, the same as the CLI’s <code>--force</code>).</p>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [hw, setHw] = useState("8");
  return (
    <div className="site">
      <header className="wrap bar">
        <a className="brand" href="#top" aria-label="Ollada home">
          <svg width="34" height="32" viewBox="12 24 106 100" fill="#6C6FD8" aria-hidden="true"><circle cx="26" cy="34" r="10" /><circle cx="22" cy="76" r="10" /><circle cx="38" cy="114" r="10" /><circle cx="72" cy="74" r="10" /><circle cx="108" cy="52" r="10" /><circle cx="108" cy="96" r="10" /></svg>
          Ollada
        </a>
        <nav aria-label="Primary">
          <a href="#shelf">Shelf</a>
          <a href="#demo">How it writes</a>
          <a href="#api">Local API</a>
          <a href="https://github.com/OpenSourceRulesz/ollada">GitHub</a>
          <a className="cta" href="#install">Install</a>
        </nav>
      </header>
      <main id="top">
        <section className="hero">
          <div className="wrap hero-in">
            <Headline />
            <p className="lede">Open-source diffusion language models on your own machine, with every number measured on real hardware.</p>
            <div className="hero-act">
              <a className="btn" href="#install">Get started</a>
              <Cmd cmd="pip install ollada" />
            </div>
            <p className="then">Needs Python 3.10 or newer.</p>
          </div>
          <div className="wrap stage" id="shelf">
            <Finder hw={hw} setHw={setHw} />
            <p className="fine fd-fine">Seven models in four families, each run through Ollada’s own code on the hardware listed. Fit is a simplified estimate (fits with 1 GB to spare, tight under that); <code>ollada recommend</code> uses the recorded measurements. The 4-bit variants were measured on an A100 (Colab) with desktop verification still pending; nemotron:8b6bit is listed for context only.</p>
            <Detail hw={hw} setHw={setHw} />
          </div>
        </section>
        <Demo />
        <div className="wrap">
          <GetStarted />
          <Api />
        </div>
      </main>
      <div className="closing">
        <div className="wrap">
          <section className="sec rules" aria-labelledby="rules-h">
            <div className="sec-head"><Squares /><h2 id="rules-h">Ground rules</h2></div>
            <div className="three">
              <div><h3>Free and local</h3><p>Runs entirely on your computer. Ollada doesn't track your usage or send your prompts back to us. No paid services. MIT licensed.</p></div>
              <div><h3>Honest about hardware</h3><p>Dream, DiffCoder and nemotron:8b4bit need a GPU. sdar:1.7b, sdar:4b and nemotron:3b run on a laptop CPU — slower, but no GPU required.</p></div>
              <div><h3>Our numbers only</h3><p>Vendor benchmarks are never shown as Ollada measurements.</p></div>
            </div>
          </section>
        </div>
        <section className="wrap final" aria-labelledby="final-h">
          <svg className="final-mark" width="64" height="60" viewBox="12 24 106 100" fill="#6C6FD8" aria-hidden="true"><circle cx="26" cy="34" r="10" /><circle cx="22" cy="76" r="10" /><circle cx="38" cy="114" r="10" /><circle cx="72" cy="74" r="10" /><circle cx="108" cy="52" r="10" /><circle cx="108" cy="96" r="10" /></svg>
          <h2 id="final-h">Start with one command.</h2>
          <div className="hero-act"><Cmd cmd="pip install ollada" /><a className="btn" href="#install">Step-by-step setup</a></div>
        </section>
        <footer className="foot">
          <div className="wrap">
            <p>Version 0.3.8 — seven models verified on named hardware. 0.3.8 adds the verified nemotron:8b4bit recipe; 0.3.7 shipped the Rust front-end: pip install now bundles a single binary, with the same Python workers underneath. Named the way Ollama was: Ollama ← LLaMA, Ollada ← LLaDA.</p>
            <p className="links">
              <a href="https://github.com/OpenSourceRulesz/ollada">Source</a>
              <a href="https://github.com/OpenSourceRulesz/ollada/blob/main/docs/FINDINGS.md">Findings</a>
              <a href="https://github.com/OpenSourceRulesz/ollada/blob/main/docs/LIBRARY.md">Library</a>
              <a href="https://pypi.org/project/ollada/">PyPI</a>
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}
