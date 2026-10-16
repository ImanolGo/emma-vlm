"""Matplotlib visualisation: camera view with projected paths, BEV, and text."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Rectangle  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_COLORS = {
    "trajectory": "#2ecc71",
    "gt_trajectory": "#e74c3c",
    "baseline": "#95a5a6",
    "history": "#34495e",
    "vehicle": "#3498db",
    "pedestrian": "#f1c40f",
    "cyclist": "#9b59b6",
    "other": "#e67e22",
    "ego": "#2c3e50",
}
_TYPE_GROUP = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "vehicle": "vehicle",
    "van": "vehicle",
    "motorcycle": "cyclist",
    "bicycle": "cyclist",
    "cyclist": "cyclist",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
}


def project_to_image(points_ego: np.ndarray, calibration: Any, z: float = 0.0) -> np.ndarray:
    """Project ego-frame ground points (N, 2) to pixel coords (N, 2); NaN if behind camera.

    Uses the real nuImages camera calibration instead of the old hand-tuned
    approximation.
    """
    pts = np.asarray(points_ego, dtype=float).reshape(-1, 2)
    xyz = np.column_stack([pts, np.full(len(pts), z)])
    rot = Quaternion(calibration.rotation).rotation_matrix  # sensor -> ego
    cam = (rot.T @ (xyz - calibration.translation).T).T
    out = np.full((len(pts), 2), np.nan)
    front = cam[:, 2] > 0.5
    if front.any():
        uvw = (calibration.intrinsic @ cam[front].T).T
        out[front] = uvw[:, :2] / uvw[:, 2:3]
    return out


class EMMAVisualizer:
    def __init__(self, cfg: Optional[Mapping] = None):
        cfg = dict(cfg or {})
        self.colors = {**DEFAULT_COLORS, **dict(cfg.get("colors") or {})}
        self.max_range = float(cfg.get("max_range", 50))
        self.fig_size = tuple(cfg.get("fig_size", (18, 10)))
        self.dpi = int(cfg.get("dpi", 120))

    def _color(self, obj_type: str) -> str:
        return self.colors.get(_TYPE_GROUP.get(obj_type, obj_type), self.colors["other"])

    def visualize_prediction(
        self,
        front_image: np.ndarray,
        prediction: Dict,
        ground_truth: Optional[Dict] = None,
        save_path: Optional[Path] = None,
        calibration: Any = None,
        history: Optional[np.ndarray] = None,
        baseline: Optional[np.ndarray] = None,
        title: str = "",
    ) -> plt.Figure:
        fig = plt.figure(figsize=self.fig_size)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3], width_ratios=[1.6, 1])
        self._plot_camera_view(fig.add_subplot(gs[0, 0]), front_image, prediction, ground_truth, calibration)
        self._plot_birds_eye_view(fig.add_subplot(gs[0, 1]), prediction, ground_truth, history, baseline)
        self._plot_text_description(fig.add_subplot(gs[1, :]), prediction)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=self.dpi)
            plt.close(fig)
            logger.info("Visualization saved to %s", save_path)
        return fig

    def _plot_camera_view(self, ax, image, prediction, ground_truth, calibration) -> None:
        # NOTE: images are loaded with PIL and are already RGB. The old code ran
        # cv2.COLOR_BGR2RGB on them, which swapped red and blue.
        ax.imshow(image)
        h, w = image.shape[:2]
        ax.set_title("Camera view (projected paths)")
        if calibration is not None:

            def draw(traj, color, label):
                pts = np.vstack([[0.0, 0.0], np.asarray(traj, dtype=float).reshape(-1, 2)])
                uv = project_to_image(pts, calibration)
                ok = ~np.isnan(uv).any(axis=1)
                if ok.sum() >= 2:
                    ax.plot(
                        uv[ok, 0],
                        uv[ok, 1],
                        "-o" if label == "prediction" else "--o",
                        color=color,
                        lw=3 if label == "prediction" else 2,
                        ms=4,
                        label=label,
                    )

            draw(prediction["trajectory"], self.colors["trajectory"], "prediction")
            if ground_truth is not None and len(ground_truth.get("trajectory", [])):
                draw(ground_truth["trajectory"], self.colors["gt_trajectory"], "ground truth")
            for obj in prediction.get("critical_objects", []):
                uv = project_to_image(np.array([obj["position"]]), calibration)[0]
                if not np.isnan(uv).any() and 0 <= uv[0] < w and 0 <= uv[1] < h:
                    ax.add_patch(Circle(uv, radius=12, color=self._color(obj["type"]), alpha=0.8))
                    ax.text(
                        uv[0] + 15,
                        uv[1],
                        f"{obj['type']} {obj['position'][0]:.0f}m",
                        color="white",
                        fontsize=8,
                        backgroundcolor="black",
                    )
            ax.legend(loc="lower right")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

    def _plot_birds_eye_view(self, ax, prediction, ground_truth, history, baseline) -> None:
        """BEV with forward (+x) pointing up and left (+y) pointing left."""
        ax.set_title("Bird's eye view (ego frame)")
        ax.set_xlabel("lateral  (m, left is left)")
        ax.set_ylabel("forward x (m)")

        def plot(traj, color, label, style="-o"):
            t = np.asarray(traj, dtype=float).reshape(-1, 2)
            if len(t):
                t = np.vstack([[0, 0], t]) if label != "history" else np.vstack([t, [0, 0]])
                ax.plot(-t[:, 1], t[:, 0], style, color=color, lw=2, ms=4, label=label)

        ax.add_patch(Rectangle((-0.95, -1.0), 1.9, 4.5, color=self.colors["ego"], alpha=0.7))
        if history is not None:
            plot(history, self.colors["history"], "history", "--o")
        if baseline is not None:
            plot(baseline, self.colors["baseline"], "const. velocity", ":o")
        plot(prediction["trajectory"], self.colors["trajectory"], "prediction")
        if ground_truth is not None and len(ground_truth.get("trajectory", [])):
            plot(ground_truth["trajectory"], self.colors["gt_trajectory"], "ground truth", "--o")

        for obj in prediction.get("critical_objects", []):
            x, y = obj["position"]
            if abs(x) > self.max_range or abs(y) > self.max_range:
                continue
            c = self._color(obj["type"])
            ax.add_patch(Circle((-y, x), radius=1.0, color=c, alpha=0.7))
            vx, vy = obj.get("velocity", [0, 0])
            if abs(vx) + abs(vy) > 0.1:
                ax.add_patch(FancyArrow(-y, x, -vy, vx, width=0.25, color=c, length_includes_head=True))

        r = self.max_range
        ax.set_xlim(-r / 2, r / 2)
        ax.set_ylim(-10, r)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    def _plot_text_description(self, ax, prediction) -> None:
        ax.axis("off")
        meta = prediction.get("meta_decision") or {}
        objs = (
            "\n".join(
                f"  - {o['type']} at ({o['position'][0]:.1f}, {o['position'][1]:.1f}) m: {o.get('description', '')}"
                for o in prediction.get("critical_objects", [])[:6]
            )
            or "  (none)"
        )
        info = prediction.get("_meta", {})
        text = "\n".join(
            [
                "Scene: " + textwrap.fill(prediction.get("scene_description", ""), 160),
                f"Decision: {meta.get('longitudinal', '?')} / {meta.get('lateral', '?')}",
                "Critical objects:\n" + objs,
                "Reasoning: " + textwrap.fill(prediction.get("reasoning", ""), 160),
                f"Model: {info.get('model', '?')}  latency: {info.get('latency_s', '?')} s",
            ]
        )
        ax.text(0.01, 0.98, text, transform=ax.transAxes, va="top", fontsize=9, family="monospace")
