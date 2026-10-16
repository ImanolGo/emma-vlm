"""EMMA-style planner backed by any vision model on OpenRouter."""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3.8-flash"

_PROMPT_DIR = Path(__file__).parent / "prompts"

# JSON schema used for structured outputs (OpenRouter `response_format`).
_NUM_PAIR = {"type": "array", "items": {"type": "number"}}
PREDICTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene_description", "critical_objects", "meta_decision", "reasoning", "trajectory"],
    "properties": {
        "scene_description": {"type": "string"},
        "critical_objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "position", "velocity", "description"],
                "properties": {
                    "type": {"type": "string"},
                    "position": _NUM_PAIR,
                    "velocity": _NUM_PAIR,
                    "description": {"type": "string"},
                },
            },
        },
        "meta_decision": {
            "type": "object",
            "additionalProperties": False,
            "required": ["longitudinal", "lateral"],
            "properties": {"longitudinal": {"type": "string"}, "lateral": {"type": "string"}},
        },
        "reasoning": {"type": "string"},
        "trajectory": {"type": "array", "items": _NUM_PAIR},
    },
}


class PredictionError(RuntimeError):
    """The model answered, but not with a usable prediction."""

    def __init__(self, message: str, raw: Optional[str] = None):
        super().__init__(message)
        self.raw = raw


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def encode_image(image: np.ndarray, max_side: int = 1280, quality: int = 90) -> str:
    """RGB array -> base64 JPEG data URL, downscaled so the long side <= max_side.

    The original sent a full-resolution PNG (~2-3 MB base64 for 1600x900),
    which is slow and costly with no accuracy benefit.
    """
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).convert("RGB")
    if max_side and max(pil.size) > max_side:
        pil.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object out of a model reply.

    Handles markdown fences and leading/trailing prose. The original did a bare
    ``json.loads(reply)``, which fails whenever the model adds a sentence or a
    ```json fence — one of the main reasons nothing worked.
    """
    if not text or not text.strip():
        raise PredictionError("Empty response from model", raw=text)
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE_RE.findall(text)]
    start = text.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise PredictionError("Could not parse JSON from model response", raw=text)


def _pair(v: Any, default: Sequence[float] = (0.0, 0.0)) -> List[float]:
    try:
        x, y = float(v[0]), float(v[1])
        if np.isfinite(x) and np.isfinite(y):
            return [x, y]
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    return list(default)


def normalize_prediction(obj: Dict[str, Any], num_points: int) -> Dict[str, Any]:
    """Validate and coerce a raw model JSON into the prediction format."""
    traj_raw = obj.get("trajectory")
    if not isinstance(traj_raw, list) or not traj_raw:
        raise PredictionError("Response has no trajectory", raw=json.dumps(obj))
    traj = []
    for p in traj_raw:
        if isinstance(p, dict):
            p = [p.get("x"), p.get("y")]
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError) as e:
            raise PredictionError(f"Bad waypoint {p!r}", raw=json.dumps(obj)) from e
        if not (np.isfinite(x) and np.isfinite(y)):
            raise PredictionError(f"Non-finite waypoint {p!r}", raw=json.dumps(obj))
        traj.append([x, y])
    if len(traj) != num_points:
        logger.warning("Model returned %d waypoints, expected %d", len(traj), num_points)
        traj = traj[:num_points]

    objects = []
    for o in obj.get("critical_objects") or []:
        if not isinstance(o, dict):
            continue
        objects.append(
            {
                "type": str(o.get("type", "other")).lower(),
                "position": _pair(o.get("position")),
                "velocity": _pair(o.get("velocity")),
                "description": str(o.get("description", "")),
            }
        )
    meta = obj.get("meta_decision") if isinstance(obj.get("meta_decision"), dict) else {}
    return {
        "scene_description": str(obj.get("scene_description", "")),
        "critical_objects": objects,
        "meta_decision": {
            "longitudinal": str(meta.get("longitudinal", "")),
            "lateral": str(meta.get("lateral", "")),
        },
        "reasoning": str(obj.get("reasoning", "")),
        "trajectory": traj,
    }


def _fmt(v: Optional[float], unit: str) -> str:
    return "unknown" if v is None else f"{v:.2f} {unit}"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class EMMA:
    """Vision-language planner using the OpenRouter (OpenAI-compatible) API."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = DEFAULT_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        reasoning_effort: Optional[str] = "low",
        structured_output: bool = True,
        image_max_side: int = 1280,
        image_quality: int = 90,
        timeout: float = 90.0,
        max_retries: int = 3,
        parse_retries: int = 1,
        provider: Optional[Dict[str, Any]] = None,
        app_name: str = "EMMA-VLM",
        app_url: str = "https://github.com/ImanolGo/emma-vlm",
        client: Any = None,
    ):
        if client is None:
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set (see .env.example)")
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,  # SDK retries 429/5xx with backoff
                default_headers={"HTTP-Referer": app_url, "X-Title": app_name},
            )
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.structured_output = structured_output
        self.image_max_side = image_max_side
        self.image_quality = image_quality
        self.parse_retries = parse_retries
        self.provider = provider

        self.system_prompt = (_PROMPT_DIR / "system.txt").read_text()
        self.analysis_template = Template((_PROMPT_DIR / "analysis.txt").read_text())

    # ---------------------------------------------------------------- prompt

    def format_prompt(
        self,
        history_xy: np.ndarray,
        history_times: np.ndarray,
        command: str,
        future_times: Sequence[float],
        speed: Optional[float] = None,
        acceleration: Optional[float] = None,
        yaw_rate: Optional[float] = None,
        camera: str = "CAM_FRONT",
        location: str = "",
        left_hand_traffic: bool = False,
        context_times: Sequence[float] = (),
    ) -> str:
        lines = [f"  t={t:+.1f}s: ({x:.2f}, {y:.2f})" for t, (x, y) in zip(history_times, history_xy)]
        lines.append("  t=+0.0s: (0.00, 0.00)")
        context_note = ""
        if context_times:
            ts = ", ".join(f"{t:+.1f}s" for t in context_times)
            context_note = f" (images in order: past frames at {ts}, then the current frame last)"
        return self.analysis_template.substitute(
            camera=camera,
            context_note=context_note,
            location=location or "unknown",
            traffic_side=(
                "vehicles drive on the LEFT side of the road"
                if left_hand_traffic
                else "vehicles drive on the RIGHT side of the road"
            ),
            command=command,
            speed=_fmt(speed, "m/s"),
            acceleration=_fmt(acceleration, "m/s^2"),
            yaw_rate=_fmt(yaw_rate, "rad/s (positive = turning left)"),
            history="\n".join(lines),
            horizon=f"{future_times[-1]:.1f}",
            num_points=len(future_times),
            future_times=", ".join(f"{t:.1f}" for t in future_times),
        )

    # ---------------------------------------------------------------- call

    def _request_kwargs(self, messages: List[Dict], structured: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        extra: Dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning"] = {"effort": self.reasoning_effort, "exclude": True}
        if self.provider:
            extra["provider"] = dict(self.provider)
        if extra:
            kwargs["extra_body"] = extra
        if structured:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "emma_prediction", "strict": True, "schema": PREDICTION_SCHEMA},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _complete(self, messages: List[Dict]) -> tuple[str, Any]:
        structured = self.structured_output
        try:
            resp = self.client.chat.completions.create(**self._request_kwargs(messages, structured))
        except Exception as e:  # provider may not support json_schema
            msg = str(e).lower()
            if structured and any(k in msg for k in ("response_format", "json_schema", "structured")):
                logger.warning("Structured output rejected (%s); retrying with json_object", e)
                self.structured_output = False
                resp = self.client.chat.completions.create(**self._request_kwargs(messages, False))
            else:
                raise
        if not getattr(resp, "choices", None):
            err = getattr(resp, "error", None)
            raise PredictionError(f"No choices in response: {err}")
        choice = resp.choices[0]
        text = choice.message.content or ""
        if getattr(choice, "finish_reason", None) == "length":
            logger.warning("Response truncated at max_tokens=%d; consider raising it", self.max_tokens)
        return text, resp

    def predict_sample(self, sample: Any, future_times: Sequence[float]) -> Dict[str, Any]:
        """Predict for a :class:`emma.data.SampleData`."""
        return self.predict_trajectory(
            image=sample.image,
            history_xy=sample.history_xy,
            history_times=sample.history_times,
            command=sample.command,
            future_times=future_times,
            context_images=sample.context_images,
            context_times=sample.context_times,
            speed=sample.speed,
            acceleration=sample.acceleration,
            yaw_rate=sample.yaw_rate,
            camera=sample.channel,
            location=sample.location,
            left_hand_traffic=sample.left_hand_traffic,
        )

    def predict_trajectory(
        self,
        image: Optional[np.ndarray] = None,
        history_xy: Optional[np.ndarray] = None,
        history_times: Optional[np.ndarray] = None,
        command: str = "go straight",
        future_times: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        context_images: Sequence[np.ndarray] = (),
        context_times: Sequence[float] = (),
        speed: Optional[float] = None,
        acceleration: Optional[float] = None,
        yaw_rate: Optional[float] = None,
        camera: str = "CAM_FRONT",
        location: str = "",
        left_hand_traffic: bool = False,
        *,
        camera_images: Optional[Dict[str, np.ndarray]] = None,
        ego_history: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Predict a trajectory. Returns the parsed prediction plus a ``_meta`` block.

        ``camera_images`` / ``ego_history`` are accepted for the old API.
        """
        if image is None and camera_images:
            image = camera_images.get("CAM_FRONT", next(iter(camera_images.values())))
        if image is None:
            raise ValueError("An image is required")
        if history_xy is None:
            history_xy = ego_history if ego_history is not None else np.zeros((0, 2))
        history_xy = np.asarray(history_xy, dtype=float).reshape(-1, 2)
        if history_times is None:
            history_times = -np.arange(len(history_xy), 0, -1) * 0.5
        future_times = [float(t) for t in future_times]
        context_images = list(context_images)

        prompt = self.format_prompt(
            history_xy,
            np.asarray(history_times, dtype=float),
            command,
            future_times,
            speed=speed,
            acceleration=acceleration,
            yaw_rate=yaw_rate,
            camera=camera,
            location=location,
            left_hand_traffic=left_hand_traffic,
            context_times=list(context_times) if context_images else (),
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in context_images + [image]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image(img, self.image_max_side, self.image_quality)},
                }
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        last_err: Optional[Exception] = None
        for attempt in range(self.parse_retries + 1):
            t0 = time.perf_counter()
            text, resp = self._complete(messages)
            latency = time.perf_counter() - t0
            try:
                pred = normalize_prediction(extract_json(text), len(future_times))
            except PredictionError as e:
                last_err = e
                logger.warning("Unparseable response (attempt %d): %s", attempt + 1, e)
                continue
            usage = getattr(resp, "usage", None)
            pred["_meta"] = {
                "model": getattr(resp, "model", self.model),
                "latency_s": round(latency, 3),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "cost_usd": getattr(usage, "cost", None),
                "attempts": attempt + 1,
            }
            pred["_meta"]["future_times"] = future_times
            return pred
        raise last_err or PredictionError("Prediction failed")


# Backwards-compatible name
ClaudeEMMA = EMMA
