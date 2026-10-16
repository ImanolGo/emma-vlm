"""Dataset loading for EMMA-style trajectory prediction.

Supports two nuScenes-family datasets, both converting ego poses into the key
frame's ego frame (x forward, y left):

* **nuImages** (``NuImagesLoader``): every *sample* is one annotated key frame
  from one of six cameras plus up to 6 past and 6 future "sweeps" (~2 Hz) from
  the same camera, each with an ego pose. ``ego_pose`` carries ``speed``,
  ``acceleration`` and ``rotation_rate`` directly.

* **nuScenes** (``NuScenesLoader``): every *sample* links six camera key frames
  (~2 Hz) and each camera has ~12 Hz sweeps via ``prev``/``next``. nuScenes
  ``ego_pose`` has no motion fields, so speed/acceleration/yaw rate are derived
  from the ego trajectory.

In both cases we get a ~3 s ego history and a ~3 s future to use as ground truth.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from emma.geometry import derive_command, global_to_ego, resample_trajectory

logger = logging.getLogger(__name__)

# Cities in nuScenes/nuImages and which side of the road traffic drives on.
_LEFT_HAND_TRAFFIC = ("singapore",)


@dataclass
class CameraCalibration:
    intrinsic: np.ndarray  # (3, 3)
    translation: np.ndarray  # sensor position in ego frame (3,)
    rotation: List[float]  # sensor -> ego quaternion [w, x, y, z]


@dataclass
class SampleData:
    token: str
    channel: str
    image: np.ndarray  # RGB uint8 (H, W, 3), key frame
    history_xy: np.ndarray  # (T, 2) ego frame, oldest first, excludes current pose
    history_times: np.ndarray  # (T,) seconds, negative
    command: str
    speed: Optional[float] = None  # m/s at key frame
    acceleration: Optional[float] = None  # longitudinal m/s^2
    yaw_rate: Optional[float] = None  # rad/s, positive = turning left
    ground_truth: Optional[Dict] = None  # {"trajectory": (N, 2), "timestamps": (N,)}
    context_images: List[np.ndarray] = field(default_factory=list)  # past frames, oldest first
    context_times: List[float] = field(default_factory=list)
    calibration: Optional[CameraCalibration] = None
    location: str = ""
    left_hand_traffic: bool = False
    log_token: Optional[str] = None

    # Backwards-compatible accessors used by the old API
    @property
    def images(self) -> Dict[str, np.ndarray]:
        return {self.channel: self.image}

    @property
    def ego_history(self) -> np.ndarray:
        return self.history_xy


class _BaseLoader:
    """Shared sampling logic for nuImages and nuScenes datasets.

    Subclasses open the right devkit object and provide three hooks:
    :meth:`_candidate_sample_data`, :meth:`_sample_log` and :meth:`_ego_motion`.
    """

    dataset: object  # set by :meth:`_open`

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        camera: str = "CAM_FRONT",
        history_seconds: float = 2.0,
        future_seconds: float = 3.0,
        dt: float = 0.5,
        context_frames: int = 0,
        verbose: bool = False,
    ):
        self.dataroot = Path(dataroot).expanduser()
        if not self.dataroot.exists():
            raise ValueError(f"Dataset root not found: {self.dataroot}")
        self._open(version=version, verbose=verbose)
        self.camera = camera
        self.dt = dt
        self.history_times = -np.arange(round(history_seconds / dt), 0, -1) * dt  # e.g. [-2,-1.5,-1,-0.5]
        self.future_times = np.arange(1, round(future_seconds / dt) + 1) * dt  # [0.5 .. 3.0]
        self.context_frames = context_frames

    # ------------------------------------------------------------- subclass hooks

    def _open(self, version: str, verbose: bool) -> None:
        raise NotImplementedError

    def _candidate_sample_data(self, sample: Dict) -> List[Dict]:
        """Key-frame sample_data records of ``sample`` (one per camera)."""
        raise NotImplementedError

    def _sample_log(self, sample: Dict) -> Dict:
        raise NotImplementedError

    def _ego_motion(
        self, key_pose: Dict, history_times: np.ndarray, history_xy: np.ndarray
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """(speed, acceleration, yaw_rate); override to derive when absent."""
        return (
            _scalar(key_pose.get("speed")),
            _first(key_pose.get("acceleration")),
            _last(key_pose.get("rotation_rate")),
        )

    # ------------------------------------------------------------------ helpers

    def _get(self, table: str, token: str) -> Dict:
        return self.dataset.get(table, token)

    def _key_sample_data(self, sample: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        """The sample_data + sensor for ``self.camera`` in ``sample``."""
        for sd in self._candidate_sample_data(sample):
            cs = self._get("calibrated_sensor", sd["calibrated_sensor_token"])
            sensor = self._get("sensor", cs["sensor_token"])
            if sensor["channel"] == self.camera:
                return sd, sensor
        return None, None

    @cached_property
    def sample_tokens(self) -> List[str]:
        """Tokens of samples that have a ``self.camera`` key frame.

        Only ~1/6 of nuImages samples are front-camera key frames; the original
        code iterated over all samples and crashed with KeyError('CAM_FRONT') on
        the rest.
        """
        tokens = []
        for sample in self.dataset.sample:
            sd, _ = self._key_sample_data(sample)
            if sd is not None:
                tokens.append(sample["token"])
        logger.info("Found %d %s samples (of %d total)", len(tokens), self.camera, len(self.dataset.sample))
        return tokens

    def get_samples(self, num_samples: Optional[int] = None) -> Iterator[SampleData]:
        tokens = self.sample_tokens[:num_samples] if num_samples else self.sample_tokens
        for token in tokens:
            try:
                yield self.get_sample_data(token)
            except Exception as e:  # keep going on a bad sample
                logger.warning("Skipping sample %s: %s", token, e)

    def _chain(self, key_sd: Dict) -> List[Dict]:
        """All sample_datas of this camera, chronological (sweeps included)."""
        before, after = [], []
        cur = key_sd
        while cur["prev"]:
            cur = self._get("sample_data", cur["prev"])
            before.append(cur)
        cur = key_sd
        while cur["next"]:
            cur = self._get("sample_data", cur["next"])
            after.append(cur)
        return before[::-1] + [key_sd] + after

    def _load_image(self, sd: Dict) -> np.ndarray:
        path = self.dataroot / sd["filename"]
        if not path.exists():
            raise FileNotFoundError(f"Image missing: {path}")
        return np.array(Image.open(path).convert("RGB"))

    def get_sample_data(self, sample_token: str) -> SampleData:
        sample = self._get("sample", sample_token)
        key_sd, sensor = self._key_sample_data(sample)
        if key_sd is None:
            raise ValueError(f"Sample {sample_token} has no {self.camera} key frame")
        cs = self._get("calibrated_sensor", key_sd["calibrated_sensor_token"])
        log = self._sample_log(sample)

        key_pose = self._get("ego_pose", key_sd["ego_pose_token"])
        t0 = key_sd["timestamp"]

        chain = self._chain(key_sd)
        times, xy = [], []
        for sd in chain:
            pose = self._get("ego_pose", sd["ego_pose_token"])
            times.append((sd["timestamp"] - t0) / 1e6)
            xy.append(pose["translation"][:2])
        times_arr = np.asarray(times)
        local = global_to_ego(np.asarray(xy), key_pose["translation"][:2], key_pose["rotation"])

        past = times_arr <= 1e-6
        future = times_arr >= -1e-6
        hist_t, hist_xy = resample_trajectory(times_arr[past], local[past], self.history_times)
        fut_t, fut_xy = resample_trajectory(times_arr[future], local[future], self.future_times)

        context_images, context_times = [], []
        if self.context_frames > 0:
            past_sds = [(t, sd) for t, sd in zip(times_arr, chain) if t < -1e-6]
            for t, sd in past_sds[-self.context_frames :]:
                try:
                    context_images.append(self._load_image(sd))
                    context_times.append(float(t))
                except FileNotFoundError as e:
                    logger.debug("%s", e)

        speed, acceleration, yaw_rate = self._ego_motion(key_pose, hist_t, hist_xy)
        location = log.get("location", "")
        return SampleData(
            token=sample_token,
            channel=sensor["channel"],
            image=self._load_image(key_sd),
            history_xy=hist_xy,
            history_times=hist_t,
            command=derive_command(fut_xy),
            speed=speed,
            acceleration=acceleration,
            yaw_rate=yaw_rate,
            ground_truth={"trajectory": fut_xy, "timestamps": fut_t},
            context_images=context_images,
            context_times=context_times,
            calibration=CameraCalibration(
                intrinsic=np.asarray(cs["camera_intrinsic"], dtype=float),
                translation=np.asarray(cs["translation"], dtype=float),
                rotation=list(cs["rotation"]),
            ),
            location=location,
            left_hand_traffic=any(c in location for c in _LEFT_HAND_TRAFFIC),
            log_token=log["token"],
        )


class NuImagesLoader(_BaseLoader):
    """Loads nuImages samples (ego frame; speed/yaw read from ``ego_pose``)."""

    def _open(self, version: str, verbose: bool) -> None:
        from nuimages import NuImages  # heavy import, keep it local

        self.dataset = NuImages(dataroot=str(self.dataroot), version=version, verbose=verbose, lazy=True)
        self.nuim = self.dataset  # backwards-compatible attribute

    def _candidate_sample_data(self, sample: Dict) -> List[Dict]:
        return [self._get("sample_data", sample["key_camera_token"])]

    def _sample_log(self, sample: Dict) -> Dict:
        return self._get("log", sample["log_token"])


class NuScenesLoader(_BaseLoader):
    """Loads nuScenes samples (ego frame; speed/yaw derived from ego poses)."""

    def _open(self, version: str, verbose: bool) -> None:
        from nuscenes.nuscenes import NuScenes  # heavy import, keep it local

        self.dataset = NuScenes(version=version, dataroot=str(self.dataroot), verbose=verbose)
        self.nusc = self.dataset  # convenience alias

    def _candidate_sample_data(self, sample: Dict) -> List[Dict]:
        # nuScenes sample["data"] maps channel -> sample_data token (older
        # devkits used a plain list of tokens); handle both.
        data = sample["data"]
        tokens = data.values() if isinstance(data, dict) else data
        return [self._get("sample_data", t) for t in tokens]

    def _sample_log(self, sample: Dict) -> Dict:
        scene = self._get("scene", sample["scene_token"])
        return self._get("log", scene["log_token"])

    def _ego_motion(self, key_pose: Dict, history_times: np.ndarray, history_xy: np.ndarray):
        """Derive speed/acceleration/yaw rate from the ego trajectory.

        nuScenes ``ego_pose`` has no motion fields (unlike nuImages), so we fit
        the last two history segments and the current one (0, 0).
        """
        ts = np.asarray(history_times, dtype=float)
        pts = np.asarray(history_xy, dtype=float).reshape(-1, 2)
        if len(ts) < 2:
            return None, None, None
        ts = np.append(ts, 0.0)
        pts = np.vstack([pts, np.zeros(2)])
        order = np.argsort(ts)
        ts, pts = ts[order], pts[order]
        dt = ts[-1] - ts[-2]
        if dt <= 1e-6:
            return None, None, None
        v_now = (pts[-1] - pts[-2]) / dt
        v_prev = (pts[-2] - pts[-3]) / (ts[-2] - ts[-3]) if len(ts) >= 3 else v_now
        speed = float(np.linalg.norm(v_now))
        prev_speed = float(np.linalg.norm(v_prev))
        acceleration = (speed - prev_speed) / dt
        yaw_now = math.atan2(v_now[1], v_now[0])
        yaw_prev = math.atan2(v_prev[1], v_prev[0])
        yaw_rate = _wrap_angle(yaw_now - yaw_prev) / dt
        return speed, acceleration, yaw_rate


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _scalar(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _first(v) -> Optional[float]:
    return _scalar(v[0]) if isinstance(v, (list, tuple)) and v else _scalar(v)


def _last(v) -> Optional[float]:
    return _scalar(v[-1]) if isinstance(v, (list, tuple)) and v else _scalar(v)
