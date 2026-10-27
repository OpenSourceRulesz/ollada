from ollada.cli import main


def test_version(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:  # argparse --version exits
        assert exc.code == 0
    assert "ollada" in capsys.readouterr().out


def test_list_shows_the_shelf(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for name in ("dream:7b", "sdar:4b", "diffcoder:8b"):
        assert name in out


def test_show_prints_recipe_and_honest_verification_state(capsys):
    assert main(["show", "sdar:4b"]) == 0
    out = capsys.readouterr().out
    assert "JetLM/SDAR-4B-Chat" in out
    assert "Tesla T4" in out  # sdar:4b verified on T4, 2026-10-21
    # the whole v0.1 shelf is verified as of 2026-10-21:
    assert main(["show", "diffcoder:8b"]) == 0
    assert "Tesla T4" in capsys.readouterr().out


def test_show_unknown_model_is_a_clean_error(capsys):
    assert main(["show", "nope:1b"]) == 2
    assert "not on the shelf" in capsys.readouterr().err


def test_show_includes_a_fit_line(capsys):
    assert main(["show", "dream:7b"]) == 0
    assert "fit here:" in capsys.readouterr().out


def test_recommend_lists_the_shelf_with_verdicts(capsys):
    assert main(["recommend"]) == 0
    out = capsys.readouterr().out
    for name in ("dream:7b", "sdar:4b", "diffcoder:8b"):
        assert name in out
    assert "VERDICT" in out
    assert "9.80" in out  # dream's verified peak is quoted with its source


class _FakeStream:
    def __init__(self, tty):
        self._tty = tty
        self.writes = []

    def isatty(self):
        return self._tty

    def write(self, text):
        self.writes.append(text)

    def flush(self):
        pass


def test_thinking_indicator_animates_on_a_tty_and_clears():
    import time

    from ollada.cli import _Thinking

    stream = _FakeStream(tty=True)
    with _Thinking(stream=stream) as thinking:
        time.sleep(0.6)
    output = "".join(stream.writes)
    assert "working" in output
    assert thinking.elapsed >= 0.6
    # The line is cleared on exit so the answer prints on a clean row.
    assert stream.writes[-1].strip() == ""


def test_thinking_indicator_is_silent_when_piped():
    from ollada.cli import _Thinking

    stream = _FakeStream(tty=False)
    with _Thinking(stream=stream):
        pass
    assert stream.writes == []


def test_parse_length_command():
    import pytest

    from ollada.cli import _parse_length_command

    assert _parse_length_command("/length 512") == 512
    assert _parse_length_command("  /length 64 ") == 64
    assert _parse_length_command("hello") is None
    assert _parse_length_command("/lengthy read, great post") is None
    with pytest.raises(ValueError):
        _parse_length_command("/length")
    with pytest.raises(ValueError):
        _parse_length_command("/length abc")
    with pytest.raises(ValueError):
        _parse_length_command("/length 0")
    with pytest.raises(ValueError):
        _parse_length_command("/length -5")


def test_canvas_watch_draws_frames_and_finishes_clean():
    from ollada.cli import _CanvasWatch

    stream = _FakeStream(tty=True)
    watch = _CanvasWatch(stream=stream)
    fired = []
    watch.on_first_frame = lambda: fired.append(True)

    watch([(" The", "done"), (" cat", "tent"), ("", "mask")], 0, 4, 0, 4)
    assert fired == [True]
    watch._last_draw = 0.0  # defeat the throttle for the second frame
    watch([(" The", "done"), (" cat", "done"), (" sat", "done")], 0, 4, 3, 4)
    watch.finish()

    out = "".join(stream.writes)
    assert "The" in out and "cat" in out and "░" in out
    # No counter line, and no cursor-up sequence anywhere: the
    # single-line renderer is the whole VS Code fix.
    assert "block " not in out
    assert "step " not in out
    import re

    assert re.search(r"\033\[\d+A", out) is None
    assert "\033[2K" in out  # single-row clear sequences were used
    assert watch._lines == 0 and watch._started is False


def test_canvas_watch_never_moves_cursor_up_and_clears_one_row():
    from ollada.cli import _CanvasWatch

    stream = _FakeStream(tty=True)
    watch = _CanvasWatch(stream=stream)
    watch._draw("aaaa bbbb cccc")
    assert watch._lines == 1
    stream.writes.clear()
    watch._draw("short")
    out = "".join(stream.writes)
    import re

    assert re.search(r"\033\[\d+A", out) is None
    assert out == "\r\033[2Kshort"
    stream.writes.clear()
    watch.finish()
    assert "".join(stream.writes) == "\r\033[2K\r"
    assert watch._lines == 0


def test_canvas_watch_caps_trailing_mask_run():
    from ollada.cli import _CanvasWatch

    stream = _FakeStream(tty=True)
    watch = _CanvasWatch(stream=stream)
    # A mostly-masked canvas must not fill the row with ░: at
    # most 20 masks, plus the "…" that says more canvas remains.
    pieces = [("Hello world", "done")] + [("", "mask")] * 100
    line = watch._render(pieces, 80)
    assert line.count("░") == 20
    assert line.endswith("…")


def test_canvas_watch_window_follows_the_frontier():
    from ollada.cli import _CanvasWatch

    stream = _FakeStream(tty=True)
    watch = _CanvasWatch(stream=stream)
    # 200 committed chars, then a tentative guess: the window must
    # show the frontier, not just the head of the canvas, and must
    # fit in one terminal row (no newline in the frame).
    pieces = [("x" * 200, "done"), (" new", "tent"), ("", "mask")]
    line = watch._render(pieces, 40)
    assert "new" in line
    assert "\n" not in line
    import re

    plain = re.sub(r"\033\[[0-9;]*m", "", line)
    assert len(plain) <= 40


def test_canvas_watch_width_prefers_its_own_stream_fd(monkeypatch):
    import os

    from ollada.cli import _CanvasWatch

    class _FdStream(_FakeStream):
        def fileno(self):
            return 7

    monkeypatch.setattr(
        os, "get_terminal_size",
        lambda fd: os.terminal_size((123, 40)) if fd == 7 else (_ for _ in ()).throw(OSError()),
    )
    watch = _CanvasWatch(stream=_FdStream(tty=True))
    assert watch._width() == 122


def test_canvas_watch_width_falls_back_without_a_tty_fd(monkeypatch):
    import os

    from ollada.cli import _CanvasWatch

    monkeypatch.setattr(os, "get_terminal_size", lambda fd: (_ for _ in ()).throw(OSError()))
    watch = _CanvasWatch(stream=_FakeStream(tty=True))
    # No fd to measure: falls back to shutil's width, still sane.
    assert watch._width() >= 40
