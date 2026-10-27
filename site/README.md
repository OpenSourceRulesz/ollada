# Ollada site source

The live homepage (https://opensourcerulesz.github.io/ollada) is built from
here — a React + Vite single-page app — and deployed as ONE file.

## Build

```sh
npx -y pnpm@10 install
npx -y pnpm@10 run build:single
```

Use pnpm@10: npm peer resolution drifts from the lockfile. The
inline-build script's printed size runs ~69 bytes smaller than the on-disk
file; the on-disk size is the truth.

Output: `dist/index.single.html`.

## Deploy

From a detached worktree synced to `origin/gh-pages` (never the stale local
branch): replace only `index.html` with `dist/index.single.html`, commit,
push `origin HEAD:gh-pages`. Wait ~60s (Pages serves stale cached copies),
then curl-verify the live page is byte-identical to the built file
(use `curl -sL`: the slash-less URL 301-redirects).

## What is live

`gh-pages` `index.html` is the real deploy output. `library.html` and
`models/*.html` are frozen legacy pages from the old Python generator
(`scripts/build_site.py`, retired) — kept for URL stability, do not
regenerate them.
