import json

import numpy as np
import pytest
from PIL import Image

from verify import cli, config
from verify.layers import layout, pixel, temporal
from verify.verdict import decide


def make_png(path, value, size=(64, 48)):
    Image.fromarray(np.full((size[1], size[0], 3), value, np.uint8)).save(path)
    return path


def make_gif(path, frames):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=50, loop=0)
    return path


def test_captures_root_requires_setting(monkeypatch):
    monkeypatch.delenv(config.ENV_CAPTURES, raising=False)
    with pytest.raises(SystemExit):
        config.captures_root()


def test_pair_paths_layout(tmp_path):
    ref, cand = config.pair_paths(tmp_path, "ios", "button_light", "gif")
    assert ref == tmp_path / "ios/maui/button_light.gif"
    assert cand == tmp_path / "ios/cpp/button_light.gif"


def test_pixel_identical_is_zero(tmp_path):
    a = make_png(tmp_path / "a.png", 128)
    r = pixel.compare(a, a)
    assert r["score"] == pytest.approx(0, abs=1e-6)
    assert r["details"]["size_mismatch"] is False


def test_pixel_localised_change_scores_high(tmp_path):
    a = make_png(tmp_path / "a.png", 128)
    arr = np.full((48, 64, 3), 128, np.uint8)
    arr[:12, :16] = 255  # one tile-sized block changed
    b = tmp_path / "b.png"
    Image.fromarray(arr).save(b)
    r = pixel.compare(a, b)
    assert r["score"] > 0.3
    assert r["details"]["max_tile"] >= r["details"]["mean_tile"]


def test_pixel_size_mismatch_is_flagged(tmp_path):
    a = make_png(tmp_path / "a.png", 128, size=(64, 48))
    b = make_png(tmp_path / "b.png", 128, size=(32, 24))
    assert pixel.compare(a, b)["details"]["size_mismatch"] is True


def test_temporal_identical_gif_is_near_zero(tmp_path):
    frames = [np.full((48, 64), v, np.uint8) for v in (0, 0, 255, 255)]
    frames = [np.stack([f] * 3, -1) for f in frames]
    g = make_gif(tmp_path / "g.gif", frames)
    assert temporal.compare(g, g)["score"] < 1e-3


def test_kind_detection(tmp_path):
    p = make_png(tmp_path / "a.png", 0)
    g = make_gif(tmp_path / "a.gif", [np.zeros((4, 4, 3), np.uint8)])
    assert cli.kind_of(p, p) == "static"
    assert cli.kind_of(g, g) == "gif"
    with pytest.raises(SystemExit):
        cli.kind_of(p, g)


def test_layout_is_unavailable_until_format_defined():
    assert layout.compare()["available"] is False
    assert layout.compare("a.json", "b.json")["available"] is False


def test_decide_bands():
    t = {"static": {"diff": 0.3, "minor": 0.15}}
    ok = lambda s: {"pixel": {"available": True, "score": s}}
    assert decide("static", ok(0.5), t)["verdict"] == "DIFFERENT"
    assert decide("static", ok(0.2), t)["verdict"] == "MINOR"
    assert decide("static", ok(0.01), t)["verdict"] == "SAME"
    assert decide("static", {"pixel": {"available": False}}, t)["verdict"] is None


def test_cli_writes_json(tmp_path):
    a = make_png(tmp_path / "a.png", 128)
    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps({"static": {"score": "pixel.score", "diff": 0.3, "minor": 0.15},
                               "gif": {"score": "temporal.score", "diff": 0.1, "minor": 0.05}}))
    out = tmp_path / "out.json"
    assert cli.main(["verify", str(a), str(a), "--thresholds", str(thr), "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["verdict"] == "SAME" and d["kind"] == "static"
    assert set(d["layers"]) == {"pixel", "layout"}
