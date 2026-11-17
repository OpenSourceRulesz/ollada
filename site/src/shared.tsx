/* Shared data and helpers, carried over unchanged from the live site so the numbers,
   fit rules, commands and headline animation are identical. */

import { useEffect, useState, type KeyboardEvent } from "react";

export type TabDef = { id: string; label: string };
export function Tabs({ tabs, value, onChange, label }: { tabs: TabDef[]; value: string; onChange: (id: string) => void; label: string }) {
  const key = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    let next: number | null = null;
    if (e.key === "ArrowRight") next = (i + 1) % tabs.length;
    if (e.key === "ArrowLeft") next = (i - 1 + tabs.length) % tabs.length;
    if (e.key === "Home") next = 0;
    if (e.key === "End") next = tabs.length - 1;
    if (next === null) return;
    e.preventDefault();
    onChange(tabs[next].id);
    e.currentTarget.parentElement?.querySelectorAll("button")[next]?.focus();
  };
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((t, i) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} tabIndex={value === t.id ? 0 : -1} onClick={() => onChange(t.id)} onKeyDown={(e) => key(e, i)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export const rnd = (n: number) => Math.floor(Math.random() * n);

/* ---------- headline: letters resolve in random order ---------- */
export const TITLE = "Pull a diffusion model. Run it.";
export const ABC = "abcdefghijklmnopqrstuvwxyz";
export type CS = { s: "m" | "g" | "c"; ch: string };

export function Headline() {
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


export const CMDS = {
  cpu: ["pip install ollada", "ollada pull sdar:1.7b", "ollada run sdar:1.7b"],
  gpu: ["pip install ollada", "ollada pull sdar:4b", "ollada run sdar:4b"],
};

export const CMD_LABELS = ["Install Ollada", "Download a model (one time)", "Start chatting"];

export function IconCopy({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setDone(true);
      setTimeout(() => setDone(false), 1600);
    }, () => {});
  };
  return (
    <button className={`icopy${done ? " done" : ""}`} onClick={copy} aria-label={done ? "Copied" : `Copy: ${text}`} title="Copy">
      {done ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5" /></svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></svg>
      )}
    </button>
  );
}


export const CMD = "ollada run sdar:1.7b";
export const PROMPT = "What is a diffusion language model?";
export const ANSWER = "Instead of writing left to right, it starts with a blank canvas and refines the whole answer at once.";
export const WORDS = ANSWER.split(" ");
export const key = (w: string) => w.toLowerCase().replace(/[^a-z]/g, "");
export const EASY = new Set("of to it a and the at with".split(" "));
export const HARD = new Set("left right writing blank canvas refines whole answer starts instead once".split(" "));
export const DIFF = WORDS.map((w, i) => (EASY.has(key(w)) ? 0.95 : HARD.has(key(w)) ? 0.45 : 0.7) + ((i * 7) % 5) * 0.01);
export const WRONG: Record<string, string[]> = { left: ["up", "back"], right: ["down", "front"], writing: ["reading", "walking"], blank: ["full", "dark"], canvas: ["page", "table"], refines: ["repeats", "erases"], whole: ["next", "last"], answer: ["word", "number"], starts: ["ends", "waits"], instead: ["because", "maybe"], once: ["twice", "later"] };
export const POOL = ["the", "a", "soft", "quiet", "very", "and", "of", "then"];
export const wrongTok = (w: string) => {
  const c = WRONG[key(w)] ?? POOL;
  let x = c[rnd(c.length)];
  while (x === key(w)) x = POOL[rnd(POOL.length)];
  if (/^[A-Z]/.test(w)) x = x[0].toUpperCase() + x.slice(1);
  return x + (w.match(/[.,]+$/)?.[0] ?? "");
};

export type Sim = { masked: boolean[]; shown: string[]; bad: boolean[]; step: number };
export const fresh = (): Sim => ({ masked: WORDS.map(() => true), shown: WORDS.map(() => ""), bad: WORDS.map(() => false), step: 0 });

export type M = { n: string; use: string; on: string; date: string; mem: string; peak: number; spd: string; kind: "gpu" | "both" | "cpu"; fam: string; isNew?: boolean; listed?: boolean };
export const SHELF: M[] = [
  { n: "diffcoder:8b", use: "Coding", on: "Desktop RTX 5060 Ti (15.9 GB), Windows 11", date: "2026-10-25 · also T4 (Colab) 2026-10-21: 15.15 GB · tight fit, ~1.2 GB spare on a display-driving card", mem: "13.65 GB VRAM", peak: 13.65, spd: "~4.1 tok/s", kind: "gpu", fam: "diffcoder" },
  { n: "diffcoder:8b4bit", use: "Coding, fits 8 GB cards", on: "NVIDIA A100-SXM4-80GB (Colab)", date: "2026-11-06 · fixed set 3/3 — train 90 km/h 35.4s, Paris 18.7s, fib 26.2s · desktop 5060 Ti verification still pending", mem: "6.59 GB VRAM", peak: 6.59, spd: "—", kind: "gpu", fam: "diffcoder" },
  { n: "dream:7b", use: "General flagship", on: "Desktop RTX 5060 Ti (15.9 GB), Windows 11", date: "2026-10-25 · also T4 (Colab) 2026-10-21: 9.80 GB · noisy on the fixed set, see below", mem: "9.13 GB VRAM", peak: 9.13, spd: "~6.7 tok/s", kind: "gpu", fam: "dream" },
  { n: "sdar:4b", use: "General, strongest per size", on: "Desktop RTX 5060 Ti (15.9 GB), Windows 11", date: "2026-10-25 · also T4 (Colab, High-RAM) 2026-10-21: 8.87 GB · also laptop CPU 2026-10-23: 8.6 GB host RAM, 63.5s / 2.4s / 16.2s, 4-step floor", mem: "8.34 GB VRAM", peak: 8.34, spd: "~12 tok/s", kind: "both", fam: "sdar" },
  { n: "sdar:1.7b", use: "The CPU pick", on: "Intel Core Ultra 7 365 laptop, 32 GB, Windows 11", date: "2026-10-23 · CPU only", mem: "3.9 GB RAM", peak: 3.9, spd: "~6.1 tok/s", kind: "cpu", fam: "sdar" },
  { n: "nemotron:3b", use: "NVIDIA tri-mode", on: "Desktop RTX 5060 Ti (15.9 GB), Windows 11 · also laptop CPU", date: "2026-11-01 · desktop GPU: fixed set 3/3 (Paris, train 90 km/h, fib correct; no VRAM peak recorded) · laptop CPU 2026-10-26: Paris 17.9s, train 90 km/h ~60s, fib 19.0s · mini code eval 11/12", mem: "7.7 GB weights", peak: 7.66, spd: "—", kind: "both", fam: "nemotron" },
  { n: "nemotron:8b4bit", use: "Chat, fits 8 GB cards", on: "NVIDIA A100-SXM4-80GB (Colab)", date: "2026-11-16 · fixed set 3/3 — train 90 km/h 32.8s, Paris 21.4s, fib 22.1s · desktop 5060 Ti verification still pending", mem: "6.18 GB VRAM", peak: 6.18, spd: "—", kind: "gpu", fam: "nemotron", isNew: true },
  { n: "nemotron:8b6bit", use: "6-bit (HQQ) test, not on the shelf", on: "Not verified for the shelf", date: "2026-11-16 test run on an A100-SXM4-40GB (Colab): fixed set 3/3 (train 90 km/h 56.3s, Paris 33.7s, fib 37.3s), but peak 16.41 GB, so it doesn’t fit a 16 GB card, and it ran slower than the 4-bit variant", mem: "16.41 GB VRAM", peak: 16.41, spd: "—", kind: "gpu", fam: "nemotron", listed: true },
];
export const FAMILIES: { id: string; name: string }[] = [
  { id: "diffcoder", name: "DiffCoder" },
  { id: "dream", name: "Dream" },
  { id: "sdar", name: "SDAR" },
  { id: "nemotron", name: "Nemotron" },
];
/** Smallest measured peak among a family's shelved (verified) variants, with its unit as shown in the table. */
export function smallestPeak(rows: M[]): string {
  const verified = rows.filter((m) => !m.listed);
  if (verified.length === 0) return "";
  return verified.reduce((a, b) => (b.peak < a.peak ? b : a)).mem;
}
export function status(m: M, hw: string): [string, string] {
  if (hw === "cpu") return m.kind === "gpu" ? ["bad", "Needs a GPU"] : m.kind === "both" ? ["warn", "Runs, slowly"] : ["ok", "Runs on CPU"];
  if (m.kind === "cpu") return ["na", "Not verified on GPU"];
  const room = +hw - m.peak;
  return room >= 1 ? ["ok", "Fits"] : room >= 0 ? ["warn", "Tight"] : ["bad", "Won’t fit"];
}

export const SERVE = `ollada serve`;
export const CURL = `curl http://localhost:11435/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "sdar:1.7b",
    "messages": [
      {"role": "user", "content": "What is the capital of France? Answer with exactly one word."}
    ]
  }'`;
export const PS = `Invoke-RestMethod -Uri http://localhost:11435/api/chat -Method Post -ContentType "application/json" -Body '{"model":"sdar:1.7b","messages":[{"role":"user","content":"What is the capital of France? Answer with exactly one word."}]}'`;
export const JS = `const res = await fetch("http://localhost:11435/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "sdar:1.7b",
    messages: [{ role: "user", content: "What is the capital of France? Answer with exactly one word." }],
  }),
});
const data = await res.json();
console.log(data.message.content); // Paris`;
export const PY = `import json, urllib.request

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

export const VENV = `python3 -m venv ~/.ollada-venv
source ~/.ollada-venv/bin/activate`;
export const VENV_PS = `py -m venv $HOME\\.ollada-venv
& $HOME\\.ollada-venv\\Scripts\\Activate.ps1`;

export function detectOs(): "mac" | "win" | "linux" {
  const ua = (typeof navigator !== "undefined" ? navigator.userAgent : "").toLowerCase();
  if (ua.includes("windows")) return "win";
  if (ua.includes("linux") && !ua.includes("android")) return "linux";
  return "mac";
}
