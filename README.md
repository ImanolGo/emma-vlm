# EMMA-VLM

An EMMA-style planner ([EMMA: End-to-End Multimodal Model for Autonomous Driving](https://arxiv.org/abs/2410.23262)) that runs on **any vision LLM via [OpenRouter](https://openrouter.ai)**. Given a front-camera image, the ego vehicle's recent motion and a routing command, the model describes the scene, lists critical objects, makes a driving decision and outputs a 3-second trajectory. Predictions are scored against nuImages ground truth and a constant-velocity baseline.

Default model: **`google/gemini-3.8-flash`** — fast (~2 s per sample), strong vision and spatial reasoning, structured JSON output, and the same family EMMA was built on. Swap it with `--model` (e.g. `openai/gpt-6-luna` or `anthropic/claude-haiku-5.5` for lower cost).

## Quick start

```bash
./scripts/setup.sh                 # creates .venv and installs the package (uv)
source .venv/bin/activate
# edit .env: OPENROUTER_API_KEY and NUIMAGES_DATAROOT

emma check                          # tests the key + model on a synthetic image (no dataset needed)
emma list                           # usable sample tokens (front-camera key frames)
emma predict                        # first sample -> outputs/predict/*.json + *.png
emma predict <token> --model openai/gpt-6-luna
emma evaluate --num-samples 50 --concurrency 8 --visualize
```

Any config value can be overridden: `emma evaluate -s data.context_frames=2 -s model.reasoning_effort=null`. See `config/config.yaml` and `emma show-config`.

## Data

Two nuScenes-family datasets are supported; pick with `data.dataset` (`nuimages` default, or `nuscenes`).

**nuImages** ([download](https://www.nuscenes.org/nuimages), the mini split is enough to start). Each sample is a key frame from one camera plus ~6 past and ~6 future sweeps at ~2 Hz with ego poses, which gives a 2 s history and a 3 s ground-truth future. Only `CAM_FRONT` key frames are used (about 1/6 of samples). `ego_pose` carries `speed`/`acceleration`/`rotation_rate` directly.

```
$NUIMAGES_DATAROOT/
    samples/   sweeps/   v1.0-mini/
```

**nuScenes** ([download](https://www.nuscenes.org/nuscenes); `v1.0-mini` is ~3.9 GB, `v1.0-trainval` is the full benchmark). Every sample links six camera key frames at 2 Hz with ~12 Hz sweeps, so it has far more samples and includes turns/lane changes. nuScenes `ego_pose` has no motion fields, so speed/acceleration/yaw rate are derived from the ego trajectory.

```
$NUSCENES_DATAROOT/
    samples/   sweeps/   v1.0-mini/
```

```bash
emma list  -s data.dataset=nuscenes            # NUSCENES_DATAROOT must be set
emma evaluate --num-samples 50 -s data.dataset=nuscenes
```

All positions are in the **ego frame at the key frame**: x forward, y left, meters, current position (0, 0).

## Metrics

`evaluate` reports ADE, FDE, L2 at 1/2/3 s (nuScenes open-loop style) and smoothness for the model **and** for a constant-velocity baseline on the same samples, plus a per-command breakdown, mean latency and cost. If the model doesn't beat the baseline, it isn't really using the image.

## Python API

```python
from emma.config import load_config, build_loader, build_model, future_times

cfg = load_config()
loader, model = build_loader(cfg), build_model(cfg)
sample = loader.get_sample_data(loader.sample_tokens[0])
pred = model.predict_sample(sample, future_times(cfg))
```

## Development

```bash
pytest          # runs against a synthetic nuImages dataset and a mocked API, no key needed
ruff check src tests
```

## License

MIT. If you use this, please cite the EMMA paper:

```bibtex
@article{hwang2024emma,
  title={EMMA: End-to-End Multimodal Model for Autonomous Driving},
  author={Hwang, Jyh-Jing and others},
  journal={arXiv preprint arXiv:2410.23262},
  year={2024}
}
```
