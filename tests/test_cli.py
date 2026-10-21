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
    # ...while an unverified model still says provisional:
    assert main(["show", "diffcoder:8b"]) == 0
    assert "provisional" in capsys.readouterr().out


def test_show_unknown_model_is_a_clean_error(capsys):
    assert main(["show", "nope:1b"]) == 2
    assert "not on the shelf" in capsys.readouterr().err
