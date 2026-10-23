"""ollada CLI — list / show / pull / run / serve / ps / rm / recommend.

The base install is dependency-light (pyyaml only). GPU work needs
`pip install "ollada[gpu]"`; serving needs `pip install "ollada[server]"`.
Commands that need a missing extra say so instead of crashing.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from ollada import __version__, fit, measurements
from ollada.recipes import RecipeError, get_recipe, load_catalog

GPU_HINT = 'this command needs the GPU stack: pip install "ollada[gpu]"'


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

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            try:
                self._stream.write("\r" + " " * 24 + "\r")
                self._stream.flush()
            except Exception:
                pass
        return False


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
        print(GPU_HINT, file=sys.stderr)
        return 2
    print(f"pulling {r.name} from {r.source} ...")
    path = snapshot_download(repo_id=r.source)
    print(f"cached at {path}")
    return 0


def _cmd_run(args) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print(GPU_HINT, file=sys.stderr)
        return 2
    from ollada.runtime.manager import ModelManager

    manager = ModelManager()
    if args.prompt:
        with _Thinking() as thinking:
            result = manager.chat(
                args.name, [{"role": "user", "content": args.prompt}], force=args.force
            )
        print(result.text)
        print(f"({thinking.elapsed:.1f}s)", file=sys.stderr)
        return 0
    print(f"{args.name} — type a message, Ctrl-C to quit")
    history: list[dict] = []
    try:
        while True:
            text = input(">>> ").strip()
            if not text:
                continue
            history.append({"role": "user", "content": text})
            with _Thinking() as thinking:
                result = manager.chat(args.name, history, force=args.force)
            history.append({"role": "assistant", "content": result.text})
            print(result.text)
            print(f"({thinking.elapsed:.1f}s)", file=sys.stderr)
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
        print(GPU_HINT, file=sys.stderr)
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


if __name__ == "__main__":
    raise SystemExit(main())
