"""ollada CLI — 0.0.1 is a planning release; the runtime is in development."""

import sys

from ollada import __version__

MESSAGE = """ollada {version} — the shelf for open-source diffusion language models.

Status: planning release. The runtime (pull / run / serve) is in development.
Plan: https://github.com/OpenSourceRulesz/ollada/blob/main/PLAN.md"""


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--version", "-V", "version"):
        print(f"ollada {__version__}")
        return 0
    print(MESSAGE.format(version=__version__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
