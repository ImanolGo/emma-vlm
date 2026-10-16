"""Coordinate transforms, trajectory metrics and simple baselines.

Conventions (nuScenes ego frame): x points forward, y points left, units are meters,
(0, 0) is the ego pose at the key frame.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion

# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #


def global_to_ego(
    points: npt.ArrayLike,
    ego_translation: Sequence[float],
    ego_rotation: Sequence[float],
) -> np.ndarray:
    """Transform global (map) xy or xyz points into the ego frame of a given pose.

    ``ego_rotation`` is a [w, x, y, z] quaternion as stored in nuScenes/nuImages.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.size == 0:
        return np.zeros((0, 2))
    dims = pts.shape[1]
    pts3 = np.zeros((len(pts), 3))
    pts3[:, :dims] = pts[:, :3]
    t = np.zeros(3)
    t[:dims] = np.asarray(ego_translation, dtype=float)[:dims]
    rot_inv = Quaternion(ego_rotation).inverse.rotation_matrix
    local = (rot_inv @ (pts3 - t).T).T
    return local[:, :dims]


def resample_trajectory(
    times: npt.ArrayLike, points: npt.ArrayLike, target_times: npt.ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate an (N, 2) trajectory onto ``target_times``.

    Target times outside the observed range are dropped (no extrapolation).
    Returns ``(kept_times, points)``.
    """
    times = np.asarray(times, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    target = np.asarray(target_times, dtype=float)
    if len(times) == 0:
        return np.zeros(0), np.zeros((0, 2))
    order = np.argsort(times)
    times, points = times[order], points[order]
    keep = target[(target >= times[0] - 1e-3) & (target <= times[-1] + 1e-3)]
    if len(keep) == 0:
        return keep, np.zeros((0, 2))
    out = np.stack([np.interp(keep, times, points[:, i]) for i in range(2)], axis=1)
    return keep, out


# --------------------------------------------------------------------------- #
# Driving command
# --------------------------------------------------------------------------- #


def derive_command(future_xy: npt.ArrayLike, lateral_threshold: float = 2.0) -> str:
    """Derive a high-level routing command from the future ego trajectory.

    Same convention as the nuScenes open-loop planning benchmarks (UniAD/VAD):
    turn if the final lateral offset exceeds ~2 m.
    """
    fut = np.asarray(future_xy, dtype=float).reshape(-1, 2)
    if len(fut) == 0:
        return "go straight"
    lateral = fut[-1, 1]
    if lateral > lateral_threshold:
        return "turn left"
    if lateral < -lateral_threshold:
        return "turn right"
    return "go straight"


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def constant_velocity_baseline(
    history_xy: npt.ArrayLike,
    history_times: npt.ArrayLike,
    future_times: npt.ArrayLike,
    speed: Optional[float] = None,
) -> np.ndarray:
    """Straight-ahead constant-velocity rollout from the current pose (0, 0).

    A VLM planner should beat this. If it doesn't, the model isn't really using
    the image. Uses the measured ego speed when available, otherwise the last
    history displacement.
    """
    ft = np.asarray(future_times, dtype=float)
    hist = np.asarray(history_xy, dtype=float).reshape(-1, 2)
    ht = np.asarray(history_times, dtype=float)
    if speed is not None and np.isfinite(speed):
        v = np.array([float(speed), 0.0])
    elif len(hist) and len(ht) and abs(ht[-1]) > 1e-6:
        v = (np.zeros(2) - hist[-1]) / (0.0 - ht[-1])
    else:
        v = np.zeros(2)
    return ft[:, None] * v[None, :]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _align(pred: npt.ArrayLike, gt: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=float).reshape(-1, 2)
    g = np.asarray(gt, dtype=float).reshape(-1, 2)
    n = min(len(p), len(g))
    return p[:n], g[:n]


def average_displacement_error(prediction: npt.ArrayLike, ground_truth: npt.ArrayLike) -> float:
    p, g = _align(prediction, ground_truth)
    if len(p) == 0:
        raise ValueError("Empty trajectory")
    return float(np.mean(np.linalg.norm(p - g, axis=1)))


def final_displacement_error(prediction: npt.ArrayLike, ground_truth: npt.ArrayLike) -> float:
    p, g = _align(prediction, ground_truth)
    if len(p) == 0:
        raise ValueError("Empty trajectory")
    return float(np.linalg.norm(p[-1] - g[-1]))


def trajectory_smoothness(trajectory: npt.ArrayLike, dt: float = 0.5) -> float:
    """Mean squared acceleration in (m/s^2)^2, including the step from (0, 0)."""
    t = np.vstack([np.zeros((1, 2)), np.asarray(trajectory, dtype=float).reshape(-1, 2)])
    if len(t) < 3:
        return 0.0
    acc = np.diff(t, n=2, axis=0) / dt**2
    return float(np.mean(np.sum(acc**2, axis=1)))


def calculate_metrics(prediction: Dict, ground_truth: Dict, dt: float = 0.5) -> Dict[str, float]:
    """ADE, FDE, nuScenes-style L2 at 1/2/3 s, and smoothness over the common horizon."""
    pred = np.asarray(prediction["trajectory"], dtype=float).reshape(-1, 2)
    gt = np.asarray(ground_truth["trajectory"], dtype=float).reshape(-1, 2)
    p, g = _align(pred, gt)
    if len(p) == 0:
        raise ValueError("No overlapping horizon between prediction and ground truth")

    errors = np.linalg.norm(p - g, axis=1)
    metrics: Dict[str, float] = {
        "ade": float(errors.mean()),
        "fde": float(errors[-1]),
        "smoothness": trajectory_smoothness(pred, dt),
        "horizon_s": float(len(p) * dt),
    }
    for sec in (1, 2, 3):
        idx = int(round(sec / dt)) - 1
        if idx < len(errors):
            metrics[f"l2_{sec}s"] = float(errors[idx])
    return metrics
