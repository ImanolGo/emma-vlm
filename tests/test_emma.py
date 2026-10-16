import json

import numpy as np
import pytest
from conftest import GOOD_REPLY, SPEED, expected_local, fake_client

from emma.data import NuImagesLoader
from emma.evaluation import evaluate
from emma.geometry import (
    calculate_metrics,
    constant_velocity_baseline,
    derive_command,
    global_to_ego,
)
from emma.model import EMMA, PredictionError, extract_json
from emma.visualizer import EMMAVisualizer, project_to_image

FT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ----------------------------------------------------------------- geometry

def test_global_to_ego_identity_and_rotation():
    from pyquaternion import Quaternion

    q = list(Quaternion(axis=[0, 0, 1], angle=np.pi / 2).elements)  # facing +y globally
    local = global_to_ego([[10.0, 25.0]], [10.0, 20.0], q)
    np.testing.assert_allclose(local, [[5.0, 0.0]], atol=1e-9)  # 5 m straight ahead


def test_metrics_handle_length_mismatch():
    m = calculate_metrics({"trajectory": [[1, 0]] * 6}, {"trajectory": [[1, 0]] * 4})
    assert m["ade"] == 0 and m["horizon_s"] == 2.0 and "l2_3s" not in m


def test_derive_command():
    assert derive_command([[10, 0.5]]) == "go straight"
    assert derive_command([[10, 5]]) == "turn left"
    assert derive_command([[10, -5]]) == "turn right"


def test_constant_velocity_baseline():
    b = constant_velocity_baseline([[-2, 0], [-1, 0]], [-1.0, -0.5], FT)
    np.testing.assert_allclose(b[-1], [6.0, 0.0])


# ----------------------------------------------------------------- loader

def test_loader_filters_cameras_and_uses_ego_frame(fake_nuimages):
    loader = NuImagesLoader(str(fake_nuimages))
    assert loader.sample_tokens == ["sample_CAM_FRONT"]
    s = loader.get_sample_data("sample_CAM_FRONT")

    assert s.image.shape == (900, 1600, 3)
    np.testing.assert_allclose(s.history_times, [-2.0, -1.5, -1.0, -0.5])
    for t, p in zip(s.history_times, s.history_xy):
        np.testing.assert_allclose(p, expected_local(t), atol=1e-6)
    gt = s.ground_truth["trajectory"]
    assert gt.shape == (6, 2)
    for t, p in zip(FT, gt):
        np.testing.assert_allclose(p, expected_local(t), atol=1e-6)
    # values are small, ego-relative numbers, not global map coordinates
    assert np.abs(gt).max() < 30
    assert s.speed == SPEED
    assert s.left_hand_traffic and s.location == "singapore-onenorth"
    # 0.1 rad/s for 3 s at 8 m/s -> lateral offset 80*(1-cos 0.3) = 3.57 m
    assert s.command == "turn left"


def test_loader_context_frames(fake_nuimages):
    s = NuImagesLoader(str(fake_nuimages), context_frames=2).get_sample_data("sample_CAM_FRONT")
    assert len(s.context_images) == 2 and s.context_times == [-1.0, -0.5]


def test_projection_points_ahead_land_below_horizon(fake_nuimages):
    s = NuImagesLoader(str(fake_nuimages)).get_sample_data("sample_CAM_FRONT")
    uv = project_to_image(np.array([[10.0, 0.0], [40.0, 0.0], [-5.0, 0.0]]), s.calibration)
    assert 491 < uv[1, 1] < uv[0, 1] < 900  # nearer -> lower in image
    assert abs(uv[0, 0] - 816) < 1  # straight ahead -> image center column
    assert np.isnan(uv[2]).all()  # behind camera


# ----------------------------------------------------------------- model

def test_extract_json_variants():
    obj = {"trajectory": [[1, 2]]}
    assert extract_json(json.dumps(obj)) == obj
    assert extract_json("Sure!\n```json\n" + json.dumps(obj) + "\n```") == obj
    assert extract_json("Here you go: " + json.dumps(obj) + " Hope it helps {}") == obj
    with pytest.raises(PredictionError):
        extract_json("no json here")


def test_predict_builds_openrouter_request_and_parses(fake_nuimages):
    client, comp = fake_client("```json\n" + GOOD_REPLY + "\n```")
    emma = EMMA(api_key=None, client=client)
    s = NuImagesLoader(str(fake_nuimages), context_frames=1).get_sample_data("sample_CAM_FRONT")
    pred = emma.predict_sample(s, FT)

    assert len(pred["trajectory"]) == 6 and pred["_meta"]["prompt_tokens"] == 1500
    req = comp.calls[0]
    assert req["model"] == "google/gemini-3.8-flash"
    assert req["response_format"]["type"] == "json_schema"
    assert req["extra_body"]["reasoning"]["effort"] == "low"
    sys_msg, user_msg = req["messages"]
    assert sys_msg["role"] == "system"
    images = [c for c in user_msg["content"] if c["type"] == "image_url"]
    assert len(images) == 2 and images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    text = user_msg["content"][0]["text"]
    for needle in ("LEFT side", "8.00 m/s", "t=-0.5s", "turn left", "0.5, 1.0, 1.5, 2.0, 2.5, 3.0"):
        assert needle in text, needle


def test_predict_retries_on_garbage_then_succeeds():
    client, comp = fake_client("I cannot comply", GOOD_REPLY)
    emma = EMMA(api_key=None, client=client)
    pred = emma.predict_trajectory(image=np.zeros((90, 160, 3), np.uint8), history_xy=[[-1, 0]])
    assert pred["_meta"]["attempts"] == 2 and len(comp.calls) == 2


def test_structured_output_fallback():
    client, comp = fake_client(RuntimeError("Provider does not support response_format json_schema"), GOOD_REPLY)
    emma = EMMA(api_key=None, client=client)
    emma.predict_trajectory(image=np.zeros((90, 160, 3), np.uint8))
    assert comp.calls[1]["response_format"] == {"type": "json_object"}


def test_missing_key_is_clear_error():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        EMMA(api_key=None)


# ----------------------------------------------------------------- eval + viz

def test_evaluate_end_to_end(fake_nuimages, tmp_path):
    client, _ = fake_client(GOOD_REPLY)
    emma = EMMA(api_key=None, client=client)
    loader = NuImagesLoader(str(fake_nuimages))
    summary = evaluate(emma, loader.get_samples(), FT, tmp_path, concurrency=2)
    assert summary["num_success"] == 1
    assert summary["model"]["ade_mean"] < 1.0  # GOOD_REPLY roughly follows the GT arc
    assert "ade_mean" in summary["constant_velocity_baseline"]
    assert (tmp_path / "evaluation_metrics.json").exists()


def test_evaluate_survives_api_errors(fake_nuimages, tmp_path):
    client, _ = fake_client(RuntimeError("boom"))
    emma = EMMA(api_key=None, client=client)
    summary = evaluate(emma, NuImagesLoader(str(fake_nuimages)).get_samples(), FT, tmp_path)
    assert summary["num_success"] == 0 and summary["errors"]


def test_visualizer_writes_png(fake_nuimages, tmp_path):
    s = NuImagesLoader(str(fake_nuimages)).get_sample_data("sample_CAM_FRONT")
    pred = json.loads(GOOD_REPLY)
    out = tmp_path / "vis.png"
    EMMAVisualizer().visualize_prediction(s.image, pred, s.ground_truth, out, s.calibration,
                                          s.history_xy, constant_velocity_baseline(
                                              s.history_xy, s.history_times, FT, s.speed))
    assert out.stat().st_size > 10_000
