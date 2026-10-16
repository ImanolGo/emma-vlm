"""Command-line interface: `emma list | predict | evaluate | visualize | check`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from emma.config import build_loader, build_model, future_times, load_config

app = typer.Typer(
    help="EMMA-VLM: end-to-end driving trajectory prediction with vision LLMs (OpenRouter)",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger("emma")

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to a config YAML")
SetOpt = typer.Option(None, "--set", "-s", help="Config override, e.g. -s model.name=openai/gpt-6-luna")
ModelOpt = typer.Option(None, "--model", "-m", help="OpenRouter model id (overrides config)")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    # The old CLI never configured logging, so all logger.info output was silent.
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    for noisy in ("httpx", "openai", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _cfg(config: Optional[Path], overrides: Optional[List[str]], model: Optional[str] = None):
    ov = list(overrides or [])
    if model:
        ov.append(f"model.name={model}")
    return load_config(config, ov)


def _render(viz, sample, prediction, path: Path, cfg) -> None:
    from emma.geometry import constant_velocity_baseline

    ft = future_times(cfg)
    viz.visualize_prediction(
        front_image=sample.image,
        prediction=prediction,
        ground_truth=sample.ground_truth,
        calibration=sample.calibration,
        history=sample.history_xy,
        baseline=constant_velocity_baseline(sample.history_xy, sample.history_times, ft, sample.speed),
        title=f"{sample.token}  |  command: {sample.command}  |  speed: "
        f"{sample.speed if sample.speed is not None else float('nan'):.1f} m/s",
        save_path=path,
    )


@app.command("list")
def list_samples(
    limit: int = typer.Option(20, help="How many tokens to show"),
    config: Optional[Path] = ConfigOpt,
    overrides: Optional[List[str]] = SetOpt,
) -> None:
    """List usable sample tokens (front-camera key frames)."""
    cfg = _cfg(config, overrides)
    loader = build_loader(cfg)
    tokens = loader.sample_tokens
    console.print(f"{len(tokens)} {cfg.data.camera} samples in {cfg.data.version}")
    for t in tokens[:limit]:
        console.print(t)


@app.command()
def check(
    model: Optional[str] = ModelOpt,
    config: Optional[Path] = ConfigOpt,
    overrides: Optional[List[str]] = SetOpt,
) -> None:
    """Check API key + model with a tiny synthetic image (no dataset needed)."""
    import numpy as np

    cfg = _cfg(config, overrides, model)
    emma = build_model(cfg)
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[180:, :] = (90, 90, 90)  # road
    img[:180, :] = (135, 180, 235)  # sky
    img[180:, 310:330] = (255, 255, 255)  # lane marking
    pred = emma.predict_trajectory(
        image=img, history_xy=[[-10, 0], [-7.5, 0], [-5, 0], [-2.5, 0]], command="go straight", speed=5.0
    )
    console.print_json(json.dumps(pred))


@app.command()
def predict(
    sample_token: Optional[str] = typer.Argument(
        None, help="Sample token (default: first front-camera sample)"
    ),
    output_dir: Path = typer.Option(Path("outputs/predict"), help="Where to save JSON and PNG"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize"),
    model: Optional[str] = ModelOpt,
    config: Optional[Path] = ConfigOpt,
    overrides: Optional[List[str]] = SetOpt,
) -> None:
    """Predict a trajectory for one sample."""
    from emma.evaluation import save_json
    from emma.geometry import calculate_metrics
    from emma.visualizer import EMMAVisualizer

    cfg = _cfg(config, overrides, model)
    loader = build_loader(cfg)
    emma = build_model(cfg)
    token = sample_token or loader.sample_tokens[0]
    sample = loader.get_sample_data(token)

    with console.status(f"Asking {cfg.model.name}..."):
        pred = emma.predict_sample(sample, future_times(cfg))

    result = {
        "token": token,
        "command": sample.command,
        "prediction": pred,
        "ground_truth": sample.ground_truth,
    }
    if len(sample.ground_truth["trajectory"]):
        result["metrics"] = calculate_metrics(pred, sample.ground_truth)
    save_json(result, output_dir / f"{token}_prediction.json")
    if visualize:
        _render(
            EMMAVisualizer(cfg.visualization), sample, pred, output_dir / f"{token}_visualization.png", cfg
        )

    console.print(f"[bold]Scene:[/bold] {pred['scene_description']}")
    console.print(f"[bold]Decision:[/bold] {pred['meta_decision']}")
    console.print(f"[bold]Trajectory:[/bold] {pred['trajectory']}")
    console.print(f"[bold]Ground truth:[/bold] {sample.ground_truth['trajectory'].round(2).tolist()}")
    if "metrics" in result:
        console.print(f"[bold]Metrics:[/bold] {result['metrics']}")
    console.print(f"Saved to {output_dir}")


@app.command()
def evaluate(
    num_samples: int = typer.Option(10, help="Number of samples (0 = all)"),
    output_dir: Path = typer.Option(Path("outputs/evaluation")),
    concurrency: Optional[int] = typer.Option(None, help="Parallel API requests"),
    visualize: bool = typer.Option(False, "--visualize/--no-visualize", help="Save a PNG per sample"),
    model: Optional[str] = ModelOpt,
    config: Optional[Path] = ConfigOpt,
    overrides: Optional[List[str]] = SetOpt,
) -> None:
    """Evaluate on many samples against ground truth and a constant-velocity baseline."""
    from emma.evaluation import evaluate as run_eval
    from emma.visualizer import EMMAVisualizer

    cfg = _cfg(config, overrides, model)
    loader = build_loader(cfg)
    emma = build_model(cfg)
    viz = EMMAVisualizer(cfg.visualization) if visualize else None
    ft = future_times(cfg)

    def on_record(rec, sample):
        status = "[red]error[/red]" if "error" in rec else f"ADE {rec['metrics']['ade']:.2f} m"
        console.print(f"{rec['token']}  {rec['command']:<12} {status}")
        if viz and "prediction" in rec:
            _render(viz, sample, rec["prediction"], output_dir / "viz" / f"{rec['token']}.png", cfg)

    summary = run_eval(
        emma,
        loader.get_samples(num_samples or None),
        ft,
        output_dir,
        concurrency or cfg.evaluation.concurrency,
        on_record,
    )

    table = Table(title=f"{cfg.model.name}: {summary['num_success']}/{summary['num_samples']} succeeded")
    table.add_column("metric")
    table.add_column("model", justify="right")
    table.add_column("const. velocity", justify="right")
    for k in ("ade", "fde", "l2_1s", "l2_2s", "l2_3s", "smoothness"):
        m, b = summary["model"].get(f"{k}_mean"), summary["constant_velocity_baseline"].get(f"{k}_mean")
        if m is not None:
            table.add_row(k, f"{m:.3f}", f"{b:.3f}" if b is not None else "-")
    console.print(table)
    if "latency_s_mean" in summary:
        console.print(f"Mean latency: {summary['latency_s_mean']:.2f} s")
    console.print(f"Saved to {output_dir / 'evaluation_metrics.json'}")


@app.command()
def visualize(
    sample_token: str = typer.Argument(...),
    prediction_path: Path = typer.Argument(..., help="JSON written by `emma predict`"),
    output_path: Path = typer.Option(Path("outputs/visualization/vis.png")),
    config: Optional[Path] = ConfigOpt,
    overrides: Optional[List[str]] = SetOpt,
) -> None:
    """Re-render a saved prediction."""
    from emma.visualizer import EMMAVisualizer

    cfg = _cfg(config, overrides)
    data = json.loads(prediction_path.read_text())
    pred = data.get("prediction", data)
    sample = build_loader(cfg).get_sample_data(sample_token)
    _render(EMMAVisualizer(cfg.visualization), sample, pred, output_path, cfg)
    console.print(f"Saved to {output_path}")


@app.command("show-config")
def show_config(config: Optional[Path] = ConfigOpt, overrides: Optional[List[str]] = SetOpt) -> None:
    """Print the resolved config (API key hidden)."""
    cfg = _cfg(config, overrides)
    c = OmegaConf.to_container(cfg, resolve=True)
    if c["model"].get("api_key"):
        c["model"]["api_key"] = "***"
    console.print(OmegaConf.to_yaml(c))


if __name__ == "__main__":
    app()
