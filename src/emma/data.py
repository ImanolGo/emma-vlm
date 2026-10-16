"""nuImages data loading for EMMA-style trajectory prediction.

Every nuImages *sample* is one annotated key frame from one of six cameras plus
up to 6 past and 6 future "sweeps" (~2 Hz) from the same camera, each with an
ego pose. That gives us a ~3 s ego history and a ~3 s future to use as ground
truth, all expressed in the ego frame of the key frame (x forward, y left).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterator, List, Optional

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


class NuImagesLoader:
    """Loads nuImages samples and converts poses into the key-frame ego frame."""

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
        from nuimages import NuImages  # heavy import, keep it local

        self.dataroot = Path(dataroot).expanduser()
        if not self.dataroot.exists():
            raise ValueError(f"nuImages dataroot not found: {self.dataroot}")
        self.nuim = NuImages(dataroot=str(self.dataroot), version=version, verbose=verbose, lazy=True)
        self.camera = camera
        self.dt = dt
        self.history_times = -np.arange(round(history_seconds / dt), 0, -1) * dt  # e.g. [-2,-1.5,-1,-0.5]
        self.future_times = np.arange(1, round(future_seconds / dt) + 1) * dt  # [0.5 .. 3.0]
        self.context_frames = context_frames

    # ------------------------------------------------------------------ index

    @cached_property
    def sample_tokens(self) -> List[str]:
        """Tokens of samples whose key frame comes from ``self.camera``.

        Only ~1/6 of nuImages samples are front-camera key frames; the original
        code iterated over all samples and crashed with KeyError('CAM_FRONT') on
        the rest.
        """
        tokens = []
        for sample in self.nuim.sample:
            sd = self.nuim.get("sample_data", sample["key_camera_token"])
            cs = self.nuim.get("calibrated_sensor", sd["calibrated_sensor_token"])
            sensor = self.nuim.get("sensor", cs["sensor_token"])
            if sensor["channel"] == self.camera:
                tokens.append(sample["token"])
        logger.info("Found %d %s samples (of %d total)", len(tokens), self.camera, len(self.nuim.sample))
        return tokens

    def get_samples(self, num_samples: Optional[int] = None) -> Iterator[SampleData]:
        tokens = self.sample_tokens[:num_samples] if num_samples else self.sample_tokens
        for token in tokens:
            try:
                yield self.get_sample_data(token)
            except Exception as e:  # keep going on a bad sample
                logger.warning("Skipping sample %s: %s", token, e)

    # ------------------------------------------------------------------ sample

    def _chain(self, key_sd: Dict) -> List[Dict]:
        """All sample_datas of this sample (same camera), chronological."""
        before, after = [], []
        cur = key_sd
        while cur["prev"]:
            cur = self.nuim.get("sample_data", cur["prev"])
            before.append(cur)
        cur = key_sd
        while cur["next"]:
            cur = self.nuim.get("sample_data", cur["next"])
            after.append(cur)
        return before[::-1] + [key_sd] + after

    def _load_image(self, sd: Dict) -> np.ndarray:
        path = self.dataroot / sd["filename"]
        if not path.exists():
            raise FileNotFoundError(f"Image missing: {path}")
        return np.array(Image.open(path).convert("RGB"))

    def get_sample_data(self, sample_token: str) -> SampleData:
        sample = self.nuim.get("sample", sample_token)
        key_sd = self.nuim.get("sample_data", sample["key_camera_token"])
        cs = self.nuim.get("calibrated_sensor", key_sd["calibrated_sensor_token"])
        sensor = self.nuim.get("sensor", cs["sensor_token"])
        log = self.nuim.get("log", sample["log_token"])

        key_pose = self.nuim.get("ego_pose", key_sd["ego_pose_token"])
        t0 = key_sd["timestamp"]

        chain = self._chain(key_sd)
        times, xy = [], []
        for sd in chain:
            pose = self.nuim.get("ego_pose", sd["ego_pose_token"])
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

        location = log.get("location", "")
        return SampleData(
            token=sample_token,
            channel=sensor["channel"],
            image=self._load_image(key_sd),
            history_xy=hist_xy,
            history_times=hist_t,
            command=derive_command(fut_xy),
            speed=_scalar(key_pose.get("speed")),
            acceleration=_first(key_pose.get("acceleration")),
            yaw_rate=_last(key_pose.get("rotation_rate")),
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
            log_token=sample["log_token"],
        )


def _scalar(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _first(v) -> Optional[float]:
    return _scalar(v[0]) if isinstance(v, (list, tuple)) and v else _scalar(v)


def _last(v) -> Optional[float]:
    return _scalar(v[-1]) if isinstance(v, (list, tuple)) and v else _scalar(v)
