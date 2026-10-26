"""ollada CLI — list / show / pull / run / serve / ps / rm / recommend / workers.

The base install is dependency-light (pyyaml + huggingface-hub). Models run
in per-family worker venvs built on first load, so a base install can run
models without the [gpu] extra; serving needs `pip install "ollada[server]"`.
Commands that need a missing extra say so instead of crashing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time

from ollada import __version__, fit, measurements
from ollada.recipes import RecipeError, get_recipe, load_catalog
from ollada.workers import WorkerEnvError


def _parse_length_command(text: str) -> int | None:
    """Parse the REPL's /length command.

    Returns the new canvas length for '/length <n>', None when the
    text is not the command, and raises ValueError for a malformed
    one — the canvas is fixed before generation starts, so a bad
    value must fail here, not inside the model.
    """
    stripped = text.strip()
    if stripped != "/length" and not stripped.startswith("/length "):
        return None
    rest = stripped[len("/length") :].strip()
    try:
        value = int(rest)
    except ValueError:
        raise ValueError("usage: /length <positive number of tokens>") from None
    if value < 1:
        raise ValueError("length must be a positive number of tokens")
    return value


class _Thinking:
    """A live "working… Ns" line on stderr while a blocking call runs.

    Generation can take seconds to minutes (especially on CPU); a
    silent terminal reads as a hang. The indicator ticks on a daemon
    thread, only when stderr is a real terminal — piped output stays
    clean, and the answer itself always goes to stdout untouched.
    `elapsed` remains readable after the block exits, for the timing
    footer.
    """

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stderr
        self._stop = threading.Event()
        self._thread = None
        self._started = None

    @property
    def elapsed(self) -> float:
        if self._started is None:
            return 0.0
        return time.monotonic() - self._started

    def __enter__(self):
        self._started = time.monotonic()
        try:
            tty = self._stream.isatty()
        except Exception:
            tty = False
        if tty:
            self._stream.write("\rworking… 0s ")
            self._stream.flush()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self):
        while not self._stop.wait(0.25):
            try:
                self._stream.write(f"\rworking… {self.elapsed:.0f}s ")
                self._stream.flush()
            except Exception:
                return

    def extinguish(self) -> None:
        """Stop the indicator immediately (watch mode is taking over
        the terminal). Idempotent; __exit__ stays safe afterwards."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
            try:
                self._stream.write("\r" + " " * 24 + "\r")
                self._stream.flush()
            except Exception:
                pass

    def __exit__(self, *exc):
        self.extinguish()
        return False


class _CanvasWatch:
    """Watch a diffusion canvas converge, live in the terminal.

    The renderer for watch-capable adapters' per-step frames (see
    adapters/sdar.py): committed tokens print plain, the model's
    current guess at a still-masked position prints dim, and an
    untouched position prints as ░. Frames redraw in place on stderr
    (the answer itself is printed clean on stdout afterwards, exactly
    as without watch mode), throttled so a fast GPU cadence drops
    frames instead of ever slowing the model. finish() erases the
    canvas so the final text lands on a clean terminal.

    Redraw contract, learned from a smeared Dell run: a frame always
    occupies exactly self._lines terminal rows. _draw clears rows a
    shorter frame leaves behind, and the wrap width is measured on the
    watch's own stream — shutil's width comes from stdout, which is
    wrong whenever stdout is piped while stderr stays a terminal.
    """

    _MIN_FRAME_SECONDS = 0.12
    _DIM = "\033[2m"
    _MASK = "\033[90m"
    _RESET = "\033[0m"

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stderr
        self._lines = 0
        self._last_draw = 0.0
        self._started = False
        # Set by the caller: invoked once, just before the first
        # frame draws (used to extinguish the working spinner).
        self.on_first_frame = None

    def _width(self) -> int:
        """Wrap width, measured on the watch's own stream.

        shutil.get_terminal_size() measures stdout, but the watch
        writes to stderr — when stdout is piped (to capture the
        answer) its width comes back wrong, wrapped rows miscount,
        and every redraw lands in the wrong place. A stream with a
        tty fd gives the true width; anything else falls back.
        """
        try:
            fd = self._stream.fileno()
        except Exception:
            fd = None
        if fd is not None:
            try:
                return max(40, os.get_terminal_size(fd).columns - 1)
            except Exception:
                pass
        return max(40, shutil.get_terminal_size().columns - 1)

    def __call__(self, pieces, block_i, num_blocks, step_i, steps):
        now = time.monotonic()
        if self._started and now - self._last_draw < self._MIN_FRAME_SECONDS:
            return
        if not self._started:
            self._started = True
            if self.on_first_frame is not None:
                self.on_first_frame()
        self._last_draw = now
        width = self._width()
        lines = self._compose(pieces, width)
        lines.append(
            f"block {block_i + 1}/{num_blocks} · step {min(step_i + 1, steps)}/{steps}"
        )
        self._draw(lines)

    def _compose(self, pieces, width):
        # Flatten to styled chunks, then wrap by display width,
        # breaking between chunks where possible. A chunk carries
        # (style_prefix, text); unstyled chunks use "".
        chunks = []
        for text, state in pieces:
            if state == "done":
                chunks.append(("", text))
            elif state == "tent":
                chunks.append((self._DIM, text))
            else:
                chunks.append((self._MASK, "░"))
        lines: list[list[tuple[str, str]]] = [[]]
        col = 0
        for style, text in chunks:
            for k, seg in enumerate(text.split("\n")):
                if k > 0:
                    lines.append([])
                    col = 0
                if not seg:
                    continue
                if col > 0 and col + len(seg) > width:
                    lines.append([])
                    col = 0
                if len(seg) > width and col == 0:
                    seg = seg[:width]
                lines[-1].append((style, seg))
                col += len(seg)
        rendered = []
        for line in lines:
            parts = []
            for style, seg in line:
                parts.append(f"{style}{seg}{self._RESET}" if style else seg)
            rendered.append("".join(parts))
        return rendered

    def _draw(self, lines):
        out = []
        prev = self._lines
        if prev:
            out.append(f"\033[{prev}A")
        for i, line in enumerate(lines):
            out.append("\r\033[2K" + line)
            if i < len(lines) - 1:
                out.append("\n")
        # A shorter frame leaves the previous frame's bottom rows on
        # screen: clear them, then move back up to the new last row.
        # Without this, a shrinking canvas (committed text replacing
        # a taller masked canvas, or a short answer after a long one)
        # leaves residue lines behind — the smearing the Dell run
        # showed, where only status lines survived visibly.
        extra = prev - len(lines)
        for _ in range(max(0, extra)):
            out.append("\n\r\033[2K")
        if extra > 0:
            out.append(f"\033[{extra}A")
        try:
            self._stream.write("".join(out))
            self._stream.flush()
        except Exception:
            pass
        self._lines = len(lines)

    def finish(self):
        """Erase the canvas; the caller prints the final answer next.
        Resets for the next chat turn (the REPL reuses one watch)."""
        if self._lines:
            out = []
            if self._lines > 1:
                out.append(f"\033[{self._lines - 1}A")
            for i in range(self._lines):
                out.append("\r\033[2K")
                if i < self._lines - 1:
                    out.append("\n")
            if self._lines > 1:
                out.append(f"\033[{self._lines - 1}A")
            out.append("\r")
            try:
                self._stream.write("".join(out))
                self._stream.flush()
            except Exception:
                pass
        self._lines = 0
        self._started = False
        self._last_draw = 0.0


def _cmd_list(_args) -> int:
    recipes = load_catalog()
    if not recipes:
        print("the shelf is empty")
        return 0
    print(f"{'NAME':<16} {'TASK':<8} {'SIZE(bf16)':<11} {'QUANT':<6} VERIFIED")
    for r in recipes:
        verified = r.verified.get("hardware") or "no"
        print(f"{r.name:<16} {r.task:<8} {r.size_gb_bf16:<11} {r.quant_default:<6} {verified}")
    return 0


def _cmd_show(args) -> int:
    r = get_recipe(args.name)
    print(f"name:        {r.name}")
    print(f"family:      {r.family}")
    print(f"source:      {r.source}")
    print(f"license:     {r.license}")
    print(f"task:        {r.task}")
    print(f"description: {r.description}")
    print(f"size (bf16): {r.size_gb_bf16} GB")
    print(f"quant:       {r.quant_default}   dtype: {r.dtype}")
    print(f"env:         transformers{r.env.get('transformers') or ' (pin set at verification)'}")
    print(f"generate:    {r.generate}")
    if r.is_verified:
        print(f"verified:    {r.verified}")
    else:
        print("verified:    no — defaults are provisional until a real-load run (PLAN.md §7)")
    record = measurements.load_record(r.name)
    if record:
        n = record["observations"]
        print(
            f"measured here: peak {record['peak_vram_gb']:.2f} GB on "
            f"{record.get('hardware') or 'this machine'} "
            f"({n} observation{'s' if n != 1 else ''}, updated {record['updated']})"
        )
    report = fit.assess(r)
    print(f"fit here:    {report.summary()}")
    for note in report.notes:
        print(f"             note: {note}")
    return 0


def _cmd_pull(args) -> int:
    r = get_recipe(args.name)
    report = fit.assess(r)
    if report.verdict in (fit.TIGHT, fit.WONT_FIT):
        print(f"fit warning: {report.summary()}", file=sys.stderr)
        for note in report.notes:
            print(f"  note: {note}", file=sys.stderr)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        # huggingface-hub is a base dependency, so this fires only on a
        # broken install — not on a missing [gpu] extra (pull needs no torch).
        print('pulling needs huggingface-hub: pip install huggingface-hub', file=sys.stderr)
        return 2
    print(f"pulling {r.name} from {r.source} ...")
    path = snapshot_download(repo_id=r.source)
    print(f"cached at {path}")
    return 0


def _cmd_run(args) -> int:
    # No torch gate here: the model runs in its worker venv, which owns
    # torch. A base install (pyyaml only) runs models fine once the
    # worker env is built; a failed build reports itself clearly.
    from ollada.runtime.manager import ModelManager

    if args.length is not None and args.length < 1:
        print("length must be a positive number of tokens", file=sys.stderr)
        return 2
    manager = ModelManager()
    # The length lever: a diffusion canvas is sized BEFORE generation,
    # and its cost scales with the size — small canvas for quick asks,
    # large when the answer needs room (the recipe default otherwise).
    session_length = args.length

    def _overrides() -> dict | None:
        return {"gen_length": session_length} if session_length else None

    # Watch mode is opt-in (OLLADA_WATCH=1). The live canvas is a nice
    # effect, but its in-place ANSI redraw proved unreliable in VS
    # Code's terminal: frames survived below the prompt after the run
    # ended, and two redraw fixes (shrink-clear, stream-measured width)
    # did not change the symptom — the foreground byte stream replays
    # clean in a terminal emulator, so the residue comes from outside
    # the run (a stale process on the terminal, or the terminal
    # replaying buffered frames late). Default runs stay clean
    # (spinner + answer); OLLADA_NO_WATCH still forces it off.
    watch = None
    if "OLLADA_WATCH" in os.environ and "OLLADA_NO_WATCH" not in os.environ:
        try:
            if sys.stderr.isatty():
                watch = _CanvasWatch()
        except Exception:
            watch = None

    def _run_chat(messages):
        thinking = _Thinking()
        with thinking:
            if watch is not None:
                watch.on_first_frame = thinking.extinguish
            try:
                result = manager.chat(
                    args.name,
                    messages,
                    overrides=_overrides(),
                    force=args.force,
                    on_step=watch,
                )
            finally:
                if watch is not None:
                    watch.finish()
        return result, thinking.elapsed

    if args.prompt:
        result, elapsed = _run_chat([{"role": "user", "content": args.prompt}])
        print(result.text)
        print(f"({elapsed:.1f}s)", file=sys.stderr)
        return 0
    print(f"{args.name} — type a message, Ctrl-C to quit · /length <n> sets the canvas size")
    history: list[dict] = []
    try:
        while True:
            text = input(">>> ").strip()
            if not text:
                continue
            try:
                new_length = _parse_length_command(text)
            except ValueError as exc:
                print(exc)
                continue
            if new_length is not None:
                session_length = new_length
                print(f"canvas length: {session_length} tokens")
                continue
            history.append({"role": "user", "content": text})
            result, elapsed = _run_chat(history)
            history.append({"role": "assistant", "content": result.text})
            print(result.text)
            print(f"({elapsed:.1f}s)", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


def _cmd_recommend(_args) -> int:
    hw = fit.detect_hardware()
    if hw.gpu_name:
        vram = f"{hw.vram_total_gb:.1f} GB VRAM" if hw.vram_total_gb is not None else "VRAM unknown"
        print(f"this machine: {hw.gpu_name} — {vram}")
    else:
        print("this machine: no GPU detected — verdicts unavailable, peaks shown for reference")
    if hw.ram_total_gb is not None:
        print(f"host RAM: {hw.ram_total_gb:.1f} GB")
    print()
    print(f"{'NAME':<16} {'VERDICT':<10} {'PEAK GB':<9} SOURCE")
    for r in load_catalog():
        report = fit.assess(r, hardware=hw)
        peak = f"{report.peak_gb:.2f}" if report.peak_gb is not None else "—"
        print(f"{r.name:<16} {report.verdict:<10} {peak:<9} {report.peak_source}")
        for note in report.notes:
            print(f"{'':<16} note: {note}")
    return 0


def _cmd_workers(args) -> int:
    from ollada.recipes import load_catalog
    from ollada.workers import list_envs, rebuild_env

    if args.workers_command == "rebuild":
        from ollada.recipes import get_recipe

        recipe = get_recipe(args.name)
        rebuild_env(recipe)
        print(f"rebuilt worker env for {recipe.family}")
        return 0
    rows = list_envs(load_catalog())
    print(f"{'FAMILY':<12} {'BUILT':<8} {'TRANSFORMERS':<14} PATH")
    for row in rows:
        print(f"{row['family']:<12} "
              f"{'yes' if row['built'] else 'no':<8} "
              f"{row['transformers'] or '—':<14} {row['path']}")
    return 0


def _cmd_serve(args) -> int:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        print('serving needs: pip install "ollada[server]"', file=sys.stderr)
        return 2
    from ollada.server import serve

    print(f"ollada serving on http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port)
    return 0


def _cmd_ps(args) -> int:
    import json
    import urllib.request

    url = f"http://127.0.0.1:{args.port}/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            print(json.loads(resp.read()))
    except Exception:
        print("no ollada daemon answering — is `ollada serve` running?")
        return 1
    return 0


def _cmd_rm(args) -> int:
    r = get_recipe(args.name)
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        # huggingface-hub is a base dependency; see _cmd_pull.
        print('this needs huggingface-hub: pip install huggingface-hub', file=sys.stderr)
        return 2
    found = False
    for repo in list(scan_cache_dir().repos):
        if repo.repo_id == r.source:
            print(f"found cached copy of {r.source} ({repo.size_on_disk_str})")
            print("delete it with:  huggingface-cli delete-cache  (interactive, safe)")
            found = True
    if not found:
        print(f"{r.source} is not in the local cache")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ollada", description=__doc__)
    parser.add_argument("--version", "-V", action="version", version=f"ollada {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="show the shelf").set_defaults(fn=_cmd_list)

    p = sub.add_parser("show", help="show one model's recipe")
    p.add_argument("name")
    p.set_defaults(fn=_cmd_show)

    p = sub.add_parser("pull", help="download a model's weights")
    p.add_argument("name")
    p.set_defaults(fn=_cmd_pull)

    p = sub.add_parser("run", help="chat with a model (one-shot with a prompt arg)")
    p.add_argument("name")
    p.add_argument("prompt", nargs="?")
    p.add_argument("--force", action="store_true", help="load even if the fit check refuses")
    p.add_argument(
        "--length",
        type=int,
        default=None,
        metavar="N",
        help="canvas size in tokens (default: the recipe's gen_length)",
    )
    p.set_defaults(fn=_cmd_run)

    sub.add_parser("recommend", help="which shelf models fit this machine").set_defaults(
        fn=_cmd_recommend
    )

    p = sub.add_parser("serve", help="run the local API daemon")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11435)
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("ps", help="ask the daemon what is loaded")
    p.add_argument("--port", type=int, default=11435)
    p.set_defaults(fn=_cmd_ps)

    p = sub.add_parser("rm", help="locate a model's cached weights")
    p.add_argument("name")
    p.set_defaults(fn=_cmd_rm)

    p = sub.add_parser("workers", help="inspect or rebuild per-family worker envs")
    wsub = p.add_subparsers(dest="workers_command")
    wsub.add_parser("list", help="show worker envs and their transformers versions")
    pr = wsub.add_parser("rebuild", help="delete and rebuild one family's worker env")
    pr.add_argument("name", help="a model name, e.g. dream:7b")
    p.set_defaults(fn=_cmd_workers)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 0
    try:
        return args.fn(args)
    except RecipeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except fit.FitError as exc:
        print(f"fit check: {exc}", file=sys.stderr)
        return 2
    except WorkerEnvError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
