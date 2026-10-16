"""Batch evaluation: model vs. ground truth vs. a constant-velocity baseline."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from emma.geometry import calculate_metrics, constant_velocity_baseline

logger = logging.getLogger(__name__)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2))


def aggregate(values: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = sorted({k for v in values for k in v})
    for k in keys:
        arr = np.array([v[k] for v in values if v.get(k) is not None], dtype=float)
        if len(arr):
            out[f"{k}_mean"] = float(arr.mean())
            out[f"{k}_median"] = float(np.median(arr))
            out[f"{k}_std"] = float(arr.std())
    return out


def evaluate_one(model: Any, sample: Any, future_times: List[float]) -> Dict[str, Any]:
    gt = sample.ground_truth or {}
    record: Dict[str, Any] = {"token": sample.token, "command": sample.command, "speed": sample.speed}
    if len(gt.get("trajectory", [])) == 0:
        record["error"] = "no ground-truth future"
        return record

    baseline = constant_velocity_baseline(sample.history_xy, sample.history_times, future_times, sample.speed)
    record["baseline_metrics"] = calculate_metrics({"trajectory": baseline}, gt)
    try:
        pred = model.predict_sample(sample, future_times)
        record["prediction"] = pred
        record["metrics"] = calculate_metrics(pred, gt)
        record["meta"] = pred.get("_meta", {})
    except Exception as e:  # one bad sample must not kill the run
        logger.warning("Sample %s failed: %s", sample.token, e)
        record["error"] = f"{type(e).__name__}: {e}"
        raw = getattr(e, "raw", None)
        if raw:
            record["raw_response"] = raw[:4000]
    record["ground_truth"] = gt
    return record


def evaluate(
    model: Any,
    samples: Iterable[Any],
    future_times: List[float],
    output_dir: Optional[Path] = None,
    concurrency: int = 4,
    on_record: Optional[Callable[[Dict[str, Any], Any], None]] = None,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(evaluate_one, model, s, future_times): s for s in samples}
        for fut in as_completed(futures):
            sample = futures[fut]
            rec = fut.result()
            records.append(rec)
            if output_dir:
                save_json(rec, Path(output_dir) / "samples" / f"{rec['token']}.json")
            if on_record:
                on_record(rec, sample)

    ok = [r for r in records if "metrics" in r]
    summary: Dict[str, Any] = {
        "num_samples": len(records),
        "num_success": len(ok),
        "success_rate": len(ok) / len(records) if records else 0.0,
        "model": aggregate([r["metrics"] for r in ok]),
        # baseline on the same successful samples, so the comparison is fair
        "constant_velocity_baseline": aggregate([r["baseline_metrics"] for r in ok]),
        "by_command": {},
        "errors": [{"token": r["token"], "error": r["error"]} for r in records if "error" in r],
    }
    lat = [r["meta"].get("latency_s") for r in ok if r.get("meta", {}).get("latency_s")]
    cost = [r["meta"].get("cost_usd") for r in ok if r.get("meta", {}).get("cost_usd")]
    if lat:
        summary["latency_s_mean"] = float(np.mean(lat))
    if cost:
        summary["cost_usd_total"] = float(np.sum(cost))
    for cmd in sorted({r["command"] for r in ok}):
        sub = [r for r in ok if r["command"] == cmd]
        summary["by_command"][cmd] = {
            "n": len(sub),
            "model_ade": float(np.mean([r["metrics"]["ade"] for r in sub])),
            "baseline_ade": float(np.mean([r["baseline_metrics"]["ade"] for r in sub])),
        }
    if output_dir:
        save_json(summary, Path(output_dir) / "evaluation_metrics.json")
    return summary
