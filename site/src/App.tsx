import { useEffect, useRef, useState } from "react";
import "./site.css";

const rnd = (n: number) => Math.floor(Math.random() * n);

/* ---------- headline: letters resolve in random order ---------- */
const TITLE = "Pull a diffusion model. Run it.";
const ABC = "abcdefghijklmnopqrstuvwxyz";
type CS = { s: "m" | "g" | "c"; ch: string };

function Headline() {
  const [cs, setCs] = useState<CS[]>(() => [...TITLE].map((ch) => ({ s: "c", ch })));
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [run, setRun] = useState(0);
  const [playing, setPlaying] = useState(!reduce);
  useEffect(() => {
    if (reduce) return;
    setPlaying(true);
    let left = [...TITLE].map((_, i) => i).filter((i) => TITLE[i] !== " ");
    const STEPS = 16;
    let step = 0;
    setCs([...TITLE].map((ch) => ({ s: ch === " " ? "c" : "m", ch: "" })));
    const id = setInterval(() => {
      step++;
      const take = step >= STEPS ? left.length : Math.ceil(left.length / (STEPS - step + 1));
      const done = new Set<number>();
      for (let j = 0; j < take; j++) done.add(left.splice(rnd(left.length), 1)[0]);
      setCs((prev) =>
        prev.map((c, i) => {
          if (TITLE[i] === " " || done.has(i)) return { s: "c", ch: TITLE[i] };
          if (left.includes(i))
            return Math.random() < 0.45
              ? { s: "g", ch: /[a-z]/i.test(TITLE[i]) ? ABC[rnd(26)] : TITLE[i] }
              : { s: "m", ch: "" };
          return c;
        })
      );
      if (!left.length) { clearInterval(id); setPlaying(false); }
    }, 110);
    return () => clearInterval(id);
  }, [run]);
  let k = 0;
  return (
    <>
      <h1 aria-label={TITLE}>
        {TITLE.split(" ").map((w, wi, arr) => (
          <span key={wi}>
            <span className="w" aria-hidden="true">
              {[...w].map((f) => {
                const c = cs[k++] ?? { s: "c", ch: f };
                return (
                  <span key={k} className={`c ${c.s === "c" ? "" : c.s}`}>
                    <i>{f}</i>
                    <b>{c.ch}</b>
                  </span>
                );
              })}
            </span>
            {wi < arr.length - 1 && " "}
            {(k += 1) && null}
          </span>
        ))}
      </h1>
      {!reduce && (
        <button className={`rp${playing ? " off" : ""}`} onClick={() => setRun((r) => r + 1)} aria-label="Replay headline animation" title="Replay" tabIndex={playing ? -1 : 0}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v5h5" /></svg>
        </button>
      )}
    </>
  );
}

/* ---------- install ---------- */
const CMDS = {
  cpu: ['pip install "ollada[cpu]"', "ollada pull sdar:1.7b", "ollada run sdar:1.7b"],
  gpu: ['pip install "ollada[gpu]"', "ollada pull diffcoder:8b", "ollada run diffcoder:8b"],
};
function Install() {
  const [t, setT] = useState<"cpu" | "gpu">("cpu");
  const [label, setLabel] = useState("Copy");
  const copy = () => {
    navigator.clipboard?.writeText(CMDS[t].join("\n")).then(
      () => setLabel("Copied"),
      () => setLabel("Select and copy")
    );
    setTimeout(() => setLabel("Copy"), 1600);
  };
  return (
    <div className="cmdbox">
      <div className="tabs" role="tablist">
        {(["cpu", "gpu"] as const).map((x) => (
          <button key={x} role="tab" aria-selected={t === x} onClick={() => setT(x)}>
            {x === "cpu" ? "Laptop (CPU)" : "GPU"}
          </button>
        ))}
      </div>
      <div className="code" role="tabpanel" tabIndex={0}>
        {CMDS[t].map((l) => (
          <span key={l}>
            <span className="p">$ </span>
            {l + "\n"}
          </span>
        ))}
        <button className="copy" onClick={copy}>{label}</button>
      </div>
    </div>
  );
}

/* ---------- denoising demo (illustrative simulation, not a real model) ---------- */
const CMD = "ollada run sdar:1.7b";
const PROMPT = "What is a diffusion language model?";
const ANSWER = "Instead of writing left to right, it starts with a blank canvas and refines the whole answer at once.";
const WORDS = ANSWER.split(" ");
const key = (w: string) => w.toLowerCase().replace(/[^a-z]/g, "");
const EASY = new Set("of to it a and the at with".split(" "));
const HARD = new Set("left right writing blank canvas refines whole answer starts instead once".split(" "));
const DIFF = WORDS.map((w, i) => (EASY.has(key(w)) ? 0.95 : HARD.has(key(w)) ? 0.45 : 0.7) + ((i * 7) % 5) * 0.01);
const WRONG: Record<string, string[]> = { left: ["up", "back"], right: ["down", "front"], writing: ["reading", "walking"], blank: ["full", "dark"], canvas: ["page", "table"], refines: ["repeats", "erases"], whole: ["next", "last"], answer: ["word", "number"], starts: ["ends", "waits"], instead: ["because", "maybe"], once: ["twice", "later"] };
const POOL = ["the", "a", "soft", "quiet", "very", "and", "of", "then"];
const wrongTok = (w: string) => {
  const c = WRONG[key(w)] ?? POOL;
  let x = c[rnd(c.length)];
  while (x === key(w)) x = POOL[rnd(POOL.length)];
  if (/^[A-Z]/.test(w)) x = x[0].toUpperCase() + x.slice(1);
  return x + (w.match(/[.,]+$/)?.[0] ?? "");
};

type Sim = { masked: boolean[]; shown: string[]; bad: boolean[]; step: number };
const fresh = (): Sim => ({ masked: WORDS.map(() => true), shown: WORDS.map(() => ""), bad: WORDS.map(() => false), step: 0 });

function Demo() {
  const [steps, setSteps] = useState(8);
  const [sim, setSim] = useState<Sim>(fresh);
  const [running, setRunning] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const go = () => {
    clearInterval(timer.current);
    let s = fresh();
    setSim(s);
    setRunning(true);
    timer.current = window.setInterval(() => {
      const step = s.step + 1;
      const idx = s.masked.map((m, i) => (m ? i : -1)).filter((i) => i >= 0);
      const k = Math.ceil(idx.length / (steps - step + 1));
      const pick = idx.map((i) => [i, DIFF[i] + Math.random() * 0.2] as const).sort((a, b) => b[1] - a[1]).slice(0, k).map((p) => p[0]);
      const next: Sim = { masked: [...s.masked], shown: [...s.shown], bad: [...s.bad], step };
      pick.forEach((i) => {
        const wrong = Math.random() < (1 - DIFF[i]) * Math.min(1, (k - 1) / 6);
        next.masked[i] = false; next.bad[i] = wrong; next.shown[i] = wrong ? wrongTok(WORDS[i]) : WORDS[i];
      });
      idx.filter((i) => !pick.includes(i)).forEach((i) => {
        next.shown[i] = Math.random() < 0.6 ? (Math.random() < DIFF[i] * (step / steps) ? WORDS[i] : wrongTok(WORDS[i])) : "";
      });
      s = next; setSim(next);
      if (next.masked.every((m) => !m)) { clearInterval(timer.current); setRunning(false); }
    }, Math.max(90, 3600 / steps));
  };
  useEffect(() => () => clearInterval(timer.current), []);
  const left = sim.masked.filter(Boolean).length;
  const wrong = sim.bad.filter(Boolean).length;
  const done = sim.step > 0 && left === 0;
  const hint = steps <= 4 ? "Few steps: many words lock in at once and mistakes slip through." : steps >= 14 ? "Many steps: slower, but each word is chosen with more of the answer in view." : "A middle setting. Try the extremes.";
  const caption = `Done in ${steps} steps${wrong ? `, with ${wrong} word${wrong > 1 ? "s" : ""} wrong. Try more steps.` : "."}`;
  return (
    <div className="demo">
      <div className="term">
        <div className="bar" aria-hidden="true"><i /><i /><i /></div>
        <div className="body">
          <div><span className="pr">$ </span>{CMD}</div>
          <div className="q"><span className="pr">&gt;&gt;&gt; </span>{PROMPT}</div>
          <div className="ans" role="img" aria-label={ANSWER}>
            {WORDS.map((w, i) => {
              const cls = sim.masked[i] ? (sim.shown[i] ? "g" : "m") : sim.bad[i] && done ? "bad" : "";
              return <span key={i} className={`tk ${cls}`} aria-hidden="true">{sim.masked[i] && !sim.shown[i] ? w : sim.shown[i] || w}</span>;
            })}
          </div>
        </div>
      </div>
      <div className="ctl">
        <label htmlFor="steps">Denoising steps: {steps}</label>
        <input id="steps" type="range" min={2} max={WORDS.length} value={steps} disabled={running} onChange={(e) => setSteps(+e.target.value)} />
        <button className="btn" onClick={go}>{running ? "Restart" : done ? "Run again" : "Run"}</button>
        <p className="stat">step {sim.step}/{steps}<br />{WORDS.length - left} of {WORDS.length} words committed{done ? ` · ${wrong} wrong` : ""}</p>
        <p className="hint">{done ? caption : hint}</p>
      </div>
    </div>
  );
}

/* ---------- shelf ---------- */
type M = { n: string; use: string; on: string; date: string; mem: string; peak: number; spd: string; kind: "gpu" | "both" | "cpu" };
const SHELF: M[] = [
  { n: "diffcoder:8b", use: "Coding", on: "Desktop RTX 5060 Ti (15.9 GB), Windows 11", date: "2026-10-25 · also T4 (Colab) 2026-10-21: 15.15 GB · tight fit, ~1.2 GB spare on a display-driving card", mem: "13.65 GB VRAM", peak: 13.65, spd: "~4.1 tok/s", kind: "gpu" },
  { n: "dream:7b", use: "General flagship", on: "Desktop RTX 5060 Ti (15.9 GB), Windows 11", date: "2026-10-25 · also T4 (Colab) 2026-10-21: 9.80 GB · noisy on the fixed set, see below", mem: "9.13 GB VRAM", peak: 9.13, spd: "~6.7 tok/s", kind: "gpu" },
  { n: "sdar:4b", use: "General, strongest per size", on: "Desktop RTX 5060 Ti (15.9 GB), Windows 11", date: "2026-10-25 · also T4 (Colab, High-RAM) 2026-10-21: 8.87 GB · also laptop CPU 2026-10-23: 8.6 GB host RAM, 63.5s / 2.4s / 16.2s, 4-step floor", mem: "8.34 GB VRAM", peak: 8.34, spd: "~12 tok/s", kind: "both" },
  { n: "nemotron:3b", use: "NVIDIA tri-mode", on: "Intel Core Ultra 7 365 laptop, 32 GB, Windows 11", date: "2026-10-26 · CPU only · fixed set: Paris 17.9s, train 90 km/h ~60s, fib 19.0s · mini code eval 11/12", mem: "7.7 GB weights", peak: 7.66, spd: "—", kind: "cpu" },
  { n: "sdar:1.7b", use: "The CPU pick", on: "Intel Core Ultra 7 365 laptop, 32 GB, Windows 11", date: "2026-10-23 · CPU only", mem: "3.9 GB RAM", peak: 3.9, spd: "~6.1 tok/s", kind: "cpu" },
];
function status(m: M, hw: string): [string, string] {
  if (hw === "cpu") return m.kind === "gpu" ? ["bad", "Needs a GPU"] : m.kind === "both" ? ["warn", "Runs, slowly"] : ["ok", "Runs on CPU"];
  if (m.kind === "cpu") return ["na", "Not verified on GPU"];
  const room = +hw - m.peak;
  return room >= 1 ? ["ok", "Fits"] : room >= 0 ? ["warn", "Tight"] : ["bad", "Won’t fit"];
}
function Shelf() {
  const [hw, setHw] = useState("cpu");
  return (
    <section id="shelf">
      <h2>The shelf</h2>
      <p className="sub">Five models, each run through Ollada’s own code on the hardware listed. Pick your machine to see what fits before you download anything.</p>
      <div className="fit">
        <label htmlFor="hw">Your machine</label>
        <select id="hw" value={hw} onChange={(e) => setHw(e.target.value)}>
          <option value="cpu">Laptop, no GPU</option>
          {[6, 8, 12, 16, 24].map((g) => <option key={g} value={g}>GPU with {g} GB</option>)}
        </select>
      </div>
      <p className="sub">No GPU? nemotron:3b and sdar:1.7b run on CPU.</p>
      <div className="scroll">
        <table>
          <thead><tr><th>Model</th><th>Good for</th><th>Verified on</th><th>Peak memory</th><th>Speed</th><th>On your machine</th></tr></thead>
          <tbody>
            {SHELF.map((m) => {
              const [c, t] = status(m, hw);
              return (
                <tr key={m.n}>
                  <td className="n">{m.n}</td><td>{m.use}</td>
                  <td>{m.on}<small>{m.date}</small></td>
                  <td className="num">{m.mem}</td><td className="num">{m.spd}</td>
                  <td><span className={`st ${c}`}>{t}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="fine">Peak memory is the latest measurement per model: desktop RTX 5060 Ti (2026-10-25) for the GPU models, laptop CPU (2026-10-23) for sdar:1.7b and (2026-10-26) for nemotron:3b. Speed is canvas tokens per second at the recipe defaults, measured by the Ollada project on T4 (2026-10-21) for the GPU models and on the laptop CPU for sdar:1.7b; nemotron:3b's speed is given as fixed-set times, not tok/s. The fit column is a simplified estimate (fits with 1 GB to spare, tight under that); <code>ollada recommend</code> uses the recorded measurements.</p>
    </section>
  );
}

/* ---------- mobile nav (burger at ≤640px, plain row on desktop) ---------- */
function NavMenu() {
  const [open, setOpen] = useState(false);
  const close = () => setOpen(false);
  return (
    <div className="navwrap">
      <button className="burger" aria-label="Menu" aria-expanded={open} onClick={() => setOpen(!open)}>☰</button>
      <nav aria-label="Primary" className={open ? "open" : ""}>
        <a href="#install" onClick={() => { close(); const el = document.getElementById("install"); if (el instanceof HTMLDetailsElement) el.open = true; }}>Install</a>
        <a href="#demo" onClick={close}>Demo</a>
        <a href="#shelf" onClick={close}>Shelf</a>
        <a href="#api" onClick={close}>Local API</a>
        <a href="https://github.com/OpenSourceRulesz/ollada" onClick={close}>GitHub</a>
        <a href="https://pypi.org/project/ollada/" onClick={close}>PyPI</a>
      </nav>
    </div>
  );
}

/* ---------- use it from code ---------- */
const SERVE = `pip install "ollada[server]"
ollada serve`;
const CURL = `curl http://localhost:11435/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{"model":"sdar:1.7b","messages":[{"role":"user","content":"What is the capital of France? Answer with exactly one word."}]}'`;
const PS = `Invoke-RestMethod -Uri http://localhost:11435/api/chat -Method Post -ContentType "application/json" -Body '{"model":"sdar:1.7b","messages":[{"role":"user","content":"What is the capital of France? Answer with exactly one word."}]}'`;
const JS = `const res = await fetch("http://localhost:11435/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "sdar:1.7b",
    messages: [{ role: "user", content: "What is the capital of France? Answer with exactly one word." }],
  }),
});
const data = await res.json();
console.log(data.message.content); // Paris`;
const PY = `import json, urllib.request

req = urllib.request.Request(
    "http://localhost:11435/api/chat",
    data=json.dumps({
        "model": "sdar:1.7b",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer with exactly one word."}],
    }).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    print(json.load(resp)["message"]["content"])  # Paris`;
function Api() {
  const [tab, setTab] = useState<"curl" | "ps" | "js" | "py">("curl");
  const [copyLabel, setCopyLabel] = useState("Copy");
  const labels = { curl: "curl", ps: "PowerShell", js: "JavaScript (Node)", py: "Python" };
  const bodies = { curl: CURL, ps: PS, js: JS, py: PY };
  const copy = () => {
    navigator.clipboard?.writeText(bodies[tab]).then(
      () => setCopyLabel("Copied"),
      () => setCopyLabel("Select and copy")
    );
    setTimeout(() => setCopyLabel("Copy"), 1600);
  };
  return (
    <section id="api">
      <h2>Local API</h2>
      <p className="sub">Start the server, then point any tool that speaks HTTP at it. The chat endpoint is Ollama-shaped; there is also <code>POST /api/generate</code> for a single prompt and OpenAI-shaped <code>POST /v1/chat/completions</code>.</p>
      <pre className="code">{SERVE}</pre>
      <div className="cmdbox">
        <div className="tabs" role="tablist">
          {(["curl", "ps", "js", "py"] as const).map((x) => (
            <button key={x} role="tab" aria-selected={tab === x} onClick={() => setTab(x)}>
              {labels[x]}
            </button>
          ))}
        </div>
        <div role="tabpanel" tabIndex={0}>
          {tab === "ps" && <p className="fine">On Windows PowerShell, <code>curl.exe</code> mangles the quoted JSON — use this instead:</p>}
          <div className="copyrow"><button className="copy" onClick={copy}>{copyLabel}</button></div>
          <pre className="code">{bodies[tab]}</pre>
        </div>
      </div>
      <p className="fine">The answer arrives whole, in one JSON object — no streaming. If the model won’t fit your machine you’ll get HTTP 409 instead of a crash (pass <code>{'options: {"force": true}'}</code> to override, the same as the CLI’s <code>--force</code>).</p>
    </section>
  );
}

/* ---------- how to install (guided path; the hero box is the quick path) ---------- */
const VENV = `python3 -m venv ~/.ollada-venv
source ~/.ollada-venv/bin/activate`;
const FIRSTRUN = `pip install "ollada[cpu]"
ollada pull sdar:1.7b
ollada run sdar:1.7b`;
function InstallGuide() {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const open = () => {
      const el = ref.current;
      if (!el) return;
      el.open = true;
      el.scrollIntoView();
    };
    if (location.hash === "#install") open();
    const onHash = () => { if (location.hash === "#install") open(); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return (
    <details id="install" ref={ref}>
      <summary>New to Python? Step-by-step setup</summary>
      <p className="sub">Three things before the three commands: Python, a terminal, and a virtual environment. Five minutes once, then the quick commands above just work.</p>
      <dl>
        <dt>1. Check Python</dt>
        <dd>You need Python 3.10 or newer. Check with <code>python3 --version</code>. The Python that ships with a Mac is 3.9 — too old. Get a current one from python.org/downloads. On Windows, tick ‘Add python.exe to PATH’ during install.</dd>
        <dt>2. Open a terminal</dt>
        <dd>Mac: Terminal (Applications → Utilities, or Spotlight → Terminal). Windows: PowerShell or Terminal. Linux: your terminal.</dd>
        <dt>3. Make a virtual environment</dt>
        <dd>Modern Python refuses to install packages globally, so ollada lives in its own folder. One command to make it, one to switch into it:
          <pre className="code">{VENV}</pre>
          <span className="fine">Windows PowerShell instead: <code>~\.ollada-venv\Scripts\Activate.ps1</code>. If PowerShell refuses to run it, run <code>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass</code> first, then activate again.</span>
        </dd>
        <dt>4. Install and run</dt>
        <dd>Laptop without a GPU — the try-first path:
          <pre className="code">{FIRSTRUN}</pre>
          NVIDIA GPU? Same steps with <code>ollada[gpu]</code> instead. At the <code>&gt;&gt;&gt;</code> prompt, ask: What is the capital of France? Answer with exactly one word. Expect Paris. Ctrl-C quits.
        </dd>
      </dl>
      <p className="fine">What to expect: the first run is slow twice — the model download (about 4 GB for sdar:1.7b) and a one-time worker-environment build (torch plus transformers, a few hundred MB). Later runs skip both. On CPU, short answers take seconds, long ones up to a minute.</p>
      <dl>
        <dt>If it fails</dt>
        <dd><code>command not found: pip</code> or <code>python3</code> — Python isn’t installed or isn’t on PATH. See step 1. <code>externally managed environment</code> — you’re not inside a virtual environment. See step 3. <code>requires Python &gt;= 3.10</code> — your Python is too old. See step 1. Anything else — copy the whole error; that’s what’s needed to help.</dd>
      </dl>
    </details>
  );
}

export default function App() {
  return (
    <div className="wrap">
      <header>
        <a className="brand" href="#top" aria-label="Ollada home">
          <svg width="52" height="49" viewBox="12 24 106 100" fill="#6C6FD8" aria-hidden="true">
            {[[26,34],[22,76],[38,114],[72,74],[108,52],[108,96]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="10"/>)}
          </svg>
          Ollada
        </a>
        <NavMenu />
      </header>
      <main id="top">
        <div className="hero">
          <Headline />
          <p className="lede">Diffusion language models are scattered across Hugging Face with mismatched code and undocumented settings. <strong>Ollada puts them on one shelf: short names, one command, and a real run on named hardware before anything is listed.</strong></p>
          <Install />
          <p className="note">Needs Python 3.10 or newer.</p>
          <InstallGuide />
        </div>
        <section id="demo">
          <h2>How a diffusion model writes</h2>
          <p className="sub">Press Run. A diffusion model starts with a fully masked answer and fills it in over several passes, locking in the words it is most sure of first. This is a simplified simulation, not a real model, and its 19-word answer caps the slider at 19 steps. The trade-off is real, though: too few steps and quality drops. Real runs use more, and Dream’s recipe uses 128 steps because 32 steps garbled the code.</p>
          <Demo />
        </section>
        <Shelf />
        <Api />
        <section>
          <h2>Desktop verification — RTX 5060 Ti, 2026-10-25</h2>
          <p className="sub">All three GPU models fit on the desktop RTX 5060 Ti. But DiffCoder is a tight fit — close other GPU apps first — and Dream’s output varies between runs, so the same prompt can occasionally fail. Full notes live in the project’s <a href="https://github.com/OpenSourceRulesz/ollada/blob/main/docs/FINDINGS.md">Findings</a>.</p>
          <details>
            <summary>Per-model findings</summary>
            <dl>
            <dt>DiffCoder:8b — 13.65 GB, fits but tight (about 1.2 GB spare on a display-driving card)</dt>
            <dd>Fixed set steady; variants wobble: the 60/45 train problem came out 40 km/h (a 45/60 = 3/2 slip), a docstring Fibonacci was cut off at the 256-token canvas, and ‘one word’ without ‘exactly’ went verbose.</dd>
            <dt>Dream:7b — 9.13 GB, fits</dt>
            <dd>The morning fixed set passed clean (90 km/h, ‘Paris’, clean Fibonacci, about 30 s each), but wider runs on the same set failed twice on the train problem and once on Fibonacci. At temperature 0.2 that reads as sampling noise, not a deterministic break — but Dream is the variable one of the three.</dd>
            <dt>SDAR:4b — 8.34 GB, fits, peak stable across 15 observations</dt>
            <dd>Fixed set and variants all correct, and the server path works: POST /api/chat answered ‘Paris’ with done=True.</dd>
            <dt>Install findings</dt>
            <dd>A fresh [gpu] install lands transformers 5.x, which breaks Dream (use 4.57.6), and torch from plain PyPI was CUDA-blind on this card (use the PyTorch CUDA index). Both are encoded in the per-family worker environments.</dd>
          </dl>
          </details>
        </section>
        <section>
          <h2>Ground rules</h2>
          <dl>
            <dt>Free and local</dt><dd>No telemetry, no phone-home, no paid services. MIT licensed.</dd>
            <dt>Honest about hardware</dt><dd>Dream and DiffCoder need a GPU. The SDAR models also run on a CPU: seconds for short answers on the 1.7B, minutes for long ones on the 4B.</dd>
            <dt>Our numbers only</dt><dd>Vendor benchmarks are never shown as Ollada measurements.</dd>
          </dl>
        </section>
      </main>
      <footer>
        <p>Version 0.3.0 — five models verified on named hardware. Named the way Ollama was: Ollama ← LLaMA, Ollada ← LLaDA.</p>
        <p><a href="https://github.com/OpenSourceRulesz/ollada">Source</a> · <a href="https://github.com/OpenSourceRulesz/ollada/blob/main/docs/FINDINGS.md">Findings</a> · <a href="https://github.com/OpenSourceRulesz/ollada/blob/main/docs/LIBRARY.md">Library</a> · <a href="https://pypi.org/project/ollada/">PyPI</a></p>
      </footer>
    </div>
  );
}
