"""Integration test for the nuScenes loader.

Runs against a real nuScenes dataset when one is available (``NUSCENES_DATAROOT``
or ``~/Documents/nuscenes``) and is skipped otherwise, so CI without the data
still passes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from emma.data import NuScenesLoader


def _find_root() -> Path | None:
    candidates = []
    if os.environ.get("NUSCENES_DATAROOT"):
        candidates.append(Path(os.environ["NUSCENES_DATAROOT"]).expanduser())
    candidates.append(Path.home() / "Documents" / "nuscenes")
    for c in candidates:
        if (c / "v1.0-mini").exists():
            return c
    return None


ROOT = _find_root()

pytestmark = pytest.mark.skipif(ROOT is None, reason="nuScenes dataset not available")


def test_nuscenes_loader_real_dataset():
    loader = NuScenesLoader(str(ROOT), version="v1.0-mini")
    assert len(loader.sample_tokens) > 100  # nuScenes mini has hundreds of samples

    # The very first record has no past history; pick one with a full window.
    s = next(
        (
            d
            for d in (loader.get_sample_data(t) for t in loader.sample_tokens[10:40])
            if len(d.history_times) == 4
        ),
        None,
    )
    assert s is not None
    assert s.channel == "CAM_FRONT"
    assert s.image.ndim == 3 and s.image.shape[2] == 3

    # ego-frame history resampled to 0.5 s, oldest first, ending at -0.5 s
    np.testing.assert_allclose(s.history_times, [-2.0, -1.5, -1.0, -0.5])
    assert s.history_xy.shape == (4, 2)

    # 3 s ground-truth future at 0.5 s spacing, ego-relative (not map coords)
    gt = s.ground_truth["trajectory"]
    assert gt.shape == (6, 2)
    assert np.abs(gt).max() < 60
    assert s.command in ("go straight", "turn left", "turn right")

    # motion is derived (nuScenes ego_pose has no speed field)
    assert s.speed is not None and s.speed >= 0
    assert s.yaw_rate is not None
