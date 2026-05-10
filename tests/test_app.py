# tests/test_app.py
"""FastAPI endpoint tests."""

from importlib.metadata import version as _v

from fastapi.testclient import TestClient

from image_analyser.app import app

client = TestClient(app)


def test_health_returns_ok_and_version():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": _v("image-analyser")}


def test_root_returns_service_info():
    r = client.get("/")
    body = r.json()
    assert r.status_code == 200
    assert body["service"] == "image-analyser"
    assert body["version"] == _v("image-analyser")
    assert "/analyse" in body["endpoints"]


def test_analyse_multipart(fixtures_dir):
    with (fixtures_dir / "1x1.png").open("rb") as f:
        r = client.post("/analyse", files={"file": ("1x1.png", f, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["format"] == "PNG"
    assert body["resolution"] == [1, 1]
    assert body["version"] == _v("image-analyser")


def test_analyse_json_path(fixtures_dir):
    r = client.post("/analyse", json={"path": str(fixtures_dir / "1x1.png")})
    assert r.status_code == 200
    assert r.json()["format"] == "PNG"


def test_analyse_missing_path_returns_404():
    r = client.post("/analyse", json={"path": "/does/not/exist.png"})
    assert r.status_code == 404


def test_analyse_unsupported_format_returns_400(tmp_path):
    bad = tmp_path / "not-image.txt"
    bad.write_text("hello")
    r = client.post("/analyse", json={"path": str(bad)})
    assert r.status_code == 400


def test_skip_only_mutex_returns_400(fixtures_dir):
    r = client.post(
        "/analyse",
        json={"path": str(fixtures_dir / "1x1.png"), "skip": ["caption"], "only": ["metadata"]},
    )
    assert r.status_code == 400
