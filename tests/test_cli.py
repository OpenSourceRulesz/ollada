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
