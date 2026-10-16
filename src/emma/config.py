"""Configuration loading (OmegaConf + .env)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: Optional[Path] = None, overrides: Optional[List[str]] = None) -> DictConfig:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        load_dotenv()  # also a .env in the current directory
    except ImportError:
        pass
    cfg = OmegaConf.load(path or DEFAULT_CONFIG)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def future_times(cfg: DictConfig) -> List[float]:
    dt = float(cfg.data.dt)
    return [float(t) for t in np.arange(1, round(cfg.data.future_seconds / dt) + 1) * dt]


def build_model(cfg: DictConfig) -> Any:
    from emma.model import EMMA

    m = cfg.model
    provider: Optional[Dict[str, Any]] = (
        OmegaConf.to_container(m.provider) if m.get("provider") else None  # type: ignore[assignment]
    )
    return EMMA(
        api_key=m.api_key,
        model=m.name,
        base_url=m.base_url,
        max_tokens=m.max_tokens,
        temperature=m.temperature,
        reasoning_effort=m.reasoning_effort,
        structured_output=m.structured_output,
        image_max_side=m.image_max_side,
        image_quality=m.image_quality,
        timeout=m.timeout,
        max_retries=m.max_retries,
        parse_retries=m.parse_retries,
        provider=provider,
    )


def build_loader(cfg: DictConfig) -> Any:
    from emma.data import NuImagesLoader, NuScenesLoader

    d = cfg.data
    dataset = str(d.get("dataset", "nuimages")).lower()
    if dataset == "nuscenes":
        cls, root = NuScenesLoader, d.nuscenes_root
    elif dataset == "nuimages":
        cls, root = NuImagesLoader, d.nuimages_root
    else:
        raise ValueError(f"Unknown data.dataset {dataset!r} (use 'nuimages' or 'nuscenes')")
    return cls(
        dataroot=root,
        version=d.version,
        camera=d.camera,
        history_seconds=d.history_seconds,
        future_seconds=d.future_seconds,
        dt=d.dt,
        context_frames=d.context_frames,
    )
