"""CLI tests."""

import json
import subprocess
import sys


def _run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "image_analyser", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_version_flag(tmp_path):
    from importlib.metadata import version

    p = _run("--version")
    assert p.returncode == 0
    pkg_version = version("image-analyser")
    assert pkg_version in p.stdout or pkg_version in p.stderr


def test_help_flag():
    p = _run("--help")
    assert p.returncode == 0
    assert "image-analyser" in p.stdout.lower()
    assert "serve" in p.stdout.lower()


def test_analyse_emits_json(fixtures_dir):
    p = _run(str(fixtures_dir / "1x1.png"), "--json")
    assert p.returncode == 0
    body = json.loads(p.stdout)
    assert body["format"] == "PNG"
    assert body["resolution"] == [1, 1]


def test_missing_file_exits_2(tmp_path):
    p = _run(str(tmp_path / "nope.png"))
    assert p.returncode == 2


def test_skip_and_only_mutex_exits_2(fixtures_dir):
    p = _run(str(fixtures_dir / "1x1.png"), "--skip", "caption", "--only", "metadata")
    assert p.returncode == 2
