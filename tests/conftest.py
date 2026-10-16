"""Builds a tiny synthetic nuImages dataset on disk so the loader can be tested
without downloading anything."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pyquaternion import Quaternion

VERSION = "v1.0-mini"
HEADING = math.radians(30)  # ego heading in the global frame
SPEED = 8.0  # m/s
YAW_RATE = 0.1  # rad/s, turning left
ORIGIN = np.array([400.0, 1100.0])


def ego_global_pose(t: float):
    """Constant speed + constant yaw rate arc starting at ORIGIN at t=0."""
    if abs(YAW_RATE) < 1e-9:
        dx, dy = SPEED * t, 0.0
    else:
        r = SPEED / YAW_RATE
        dx, dy = r * math.sin(YAW_RATE * t), r * (1 - math.cos(YAW_RATE * t))
    c, s = math.cos(HEADING), math.sin(HEADING)
    gx, gy = ORIGIN + np.array([c * dx - s * dy, s * dx + c * dy])
    yaw = HEADING + YAW_RATE * t
    return [gx, gy, 0.0], list(Quaternion(axis=[0, 0, 1], angle=yaw).elements)


def expected_local(t: float):
    if abs(YAW_RATE) < 1e-9:
        return np.array([SPEED * t, 0.0])
    r = SPEED / YAW_RATE
    return np.array([r * math.sin(YAW_RATE * t), r * (1 - math.cos(YAW_RATE * t))])


@pytest.fixture(scope="session")
def fake_nuimages(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("nuimages")
    tables = {
        k: []
        for k in [
            "attribute",
            "calibrated_sensor",
            "category",
            "ego_pose",
            "log",
            "object_ann",
            "sample",
            "sample_data",
            "sensor",
            "surface_ann",
        ]
    }
    tables["log"].append(
        {
            "token": "log0",
            "logfile": "x",
            "vehicle": "n008",
            "date_captured": "2018-08-01",
            "location": "singapore-onenorth",
        }
    )
    # front camera: realistic nuScenes-like intrinsics/extrinsics
    cam_rot = Quaternion(matrix=np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float))  # cam->ego
    for ch in ("CAM_FRONT", "CAM_BACK"):
        tables["sensor"].append({"token": f"sensor_{ch}", "channel": ch, "modality": "camera"})
        tables["calibrated_sensor"].append(
            {
                "token": f"cs_{ch}",
                "sensor_token": f"sensor_{ch}",
                "translation": [1.7, 0.0, 1.5],
                "rotation": list(cam_rot.elements),
                "camera_intrinsic": [[1266.0, 0, 816.0], [0, 1266.0, 491.0], [0, 0, 1]],
                "camera_distortion": [0, 0, 0, 0, 0],
            }
        )

    img = np.zeros((900, 1600, 3), dtype=np.uint8)
    img[:450] = (135, 180, 235)
    img[450:] = (90, 90, 90)
    for ch in ("CAM_FRONT", "CAM_BACK"):
        times = [k * 0.5 for k in range(-6, 7)]
        sd_tokens = [f"sd_{ch}_{k}" for k in range(13)]
        for k, t in enumerate(times):
            trans, rot = ego_global_pose(t)
            ep = f"ep_{ch}_{k}"
            tables["ego_pose"].append(
                {
                    "token": ep,
                    "translation": trans,
                    "rotation": rot,
                    "timestamp": int(1e15 + t * 1e6),
                    "speed": SPEED,
                    "acceleration": [0.0, 0.0, 0.0],
                    "rotation_rate": [0.0, 0.0, YAW_RATE],
                    "rotation_rate_frame": "ego",
                }
            )
            kind = "samples" if k == 6 else "sweeps"
            fname = f"{kind}/{ch}/{ch}_{k}.jpg"
            (root / kind / ch).mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(root / fname)
            tables["sample_data"].append(
                {
                    "token": sd_tokens[k],
                    "sample_token": f"sample_{ch}",
                    "ego_pose_token": ep,
                    "calibrated_sensor_token": f"cs_{ch}",
                    "filename": fname,
                    "fileformat": "jpg",
                    "width": 1600,
                    "height": 900,
                    "timestamp": int(1e15 + t * 1e6),
                    "is_key_frame": k == 6,
                    "prev": sd_tokens[k - 1] if k > 0 else "",
                    "next": sd_tokens[k + 1] if k < 12 else "",
                }
            )
        tables["sample"].append(
            {
                "token": f"sample_{ch}",
                "timestamp": int(1e15),
                "log_token": "log0",
                "key_camera_token": sd_tokens[6],
            }
        )

    (root / VERSION).mkdir()
    for name, rows in tables.items():
        (root / VERSION / f"{name}.json").write_text(json.dumps(rows))
    return root


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1500, completion_tokens=300, cost=0.002),
        )


def fake_client(*replies):
    comp = FakeCompletions(replies)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


GOOD_REPLY = json.dumps(
    {
        "scene_description": "Clear day, straight two-lane road.",
        "critical_objects": [
            {"type": "car", "position": [20.0, 0.5], "velocity": [6.0, 0.0], "description": "lead car"}
        ],
        "meta_decision": {"longitudinal": "keep speed", "lateral": "keep lane"},
        "reasoning": "Follow the lane at current speed.",
        "trajectory": [[4.0, 0.1], [8.0, 0.4], [12.0, 0.9], [16.0, 1.6], [19.9, 2.5], [23.8, 3.6]],
    }
)
