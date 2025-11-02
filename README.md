# FrozenLake-style Spatial Reasoning Benchmark

This repository contains utilities for generating and evaluating FrozenLake-style gridworld datasets to study multimodal reasoning. The project mirrors the experimental setup described in [Visual Planning via Large Vision-Language Models](https://arxiv.org/abs/2505.11409), with paired ASCII and image representations of each environment.

## Directory Layout

```
frozenlake_benchmark/
├── data/                # Generated datasets (gitignored; run the generator first)
├── report.ipynb         # Notebook for analysis and visualization
├── scripts/             # Helper shell scripts
└── src/                 # Python modules implementing the benchmark
```

## Requirements

* Python 3.10+
* [uv](https://docs.astral.sh/uv/) for dependency and virtual environment management
* [Pillow](https://pypi.org/project/Pillow/) for rendering PNG grids
* [huggingface_hub](https://pypi.org/project/huggingface-hub/) for remote inference (optional)
* [transformers](https://pypi.org/project/transformers/) and [PyTorch](https://pypi.org/project/torch/) for local text-only inference (optional)

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create/refresh the project environment and install dependencies:

```bash
uv sync --extra hpc
```

`uv sync` creates a `.venv` in the repository. Either activate it (`source .venv/bin/activate`) or prefix commands with `uv run ...` to ensure they use the locked dependencies.

## Dataset Generation

Run the generator to produce the train/test splits:

```bash
uv run ./frozenlake_benchmark/scripts/gen_all.sh --output-dir frozenlake_benchmark/data
```

This command creates `train.jsonl` and `test.jsonl`. Default grid sizes are 3×3 to 6×6 with 1,000 train and 250 test layouts per size (note that extremely small grids have a limited number of unique solvable layouts, so you may need to override counts for those sizes). Command-line flags allow you to customize grid sizes, dataset sizes, hole probability, and random seed. Pass `--save-renderings` if you also want ASCII/PNG dumps in `data/render/` (handy for qualitative inspection, but large and therefore ignored by Git).

To retroactively produce renderings for an existing split, use `render_dataset.py`:

```bash
uv run python -m frozenlake_benchmark.src.render_dataset \
  --dataset frozenlake_benchmark/data/test.jsonl \
  --output-dir frozenlake_benchmark/data/render/test
```

The helper mirrors the filename convention used by the generator and writes ASCII grids and PNG renderings side-by-side. The output directory is gitignored so you can regenerate it locally without polluting commits.

Use `--train-counts`/`--test-counts` to provide explicit `SIZE:COUNT` overrides when you want the total dataset to follow a specific split. For example, the committed dataset was generated via:

```bash
uv run python -m frozenlake_benchmark.src.generate_dataset \
  --output-dir frozenlake_benchmark/data \
  --train-counts 3:32 4:368 5:300 6:300 \
  --test-counts 3:8 4:92 5:75 6:75
```

Each dataset entry includes:

```json
{
  "grid_size": 4,
  "layout": ["FSFF", "FFFF", "FFFH", "FFFG"],
  "hole_prob": 0.2,
  "start": [0, 1],
  "goal": [3, 3],
  "optimal_actions": ["down", "down", "down", "right", "right"],
  "optimal_action_sequences": [
    ["down", "down", "down", "right", "right"],
    ["down", "down", "right", "down", "right"],
    ["down", "right", "down", "down", "right"],
    ["right", "down", "down", "down", "right"]
  ],
  "path_coords": [[0, 1], [1, 1], [2, 1], [3, 1], [3, 2], [3, 3]]
}
```

The start (`S`) and goal (`G`) tiles are sampled uniformly across the grid (subject to being distinct) so the agent must reason over diverse viewpoints. If renderings are saved, the records also include `ascii_path` and `image_path` fields pointing to the corresponding assets.

## Evaluation

Use `run_vlm_eval.py` to compare models across modalities. The script now supports three execution backends:

* `mock` – predicts the optimal plan for each board. Handy for smoke tests.
* `huggingface` – calls hosted inference endpoints via `huggingface_hub.InferenceClient`.
* `transformers` – loads a local `transformers` checkpoint (e.g., a downloaded Qwen2.5-VL model) and performs greedy decoding offline.

Example invocations:

```bash
# Dry run using the built-in oracle
uv run ./frozenlake_benchmark/scripts/eval_ascii.sh --backend mock --limit 5

# Remote evaluation with Hugging Face (requires API token)
HUGGING_FACE_HUB_TOKEN=... uv run ./frozenlake_benchmark/scripts/eval_ascii.sh \
  --backend huggingface \
  --model Qwen/Qwen2.5-Instruct \
  --token "$HUGGING_FACE_HUB_TOKEN" \
  --dataset frozenlake_benchmark/data/test.jsonl \
  --limit 10
```

Image-only evaluation still requires a local VLM runtime. Export the dataset and run in an environment with GPU access to evaluate `Qwen2.5-VL-Instruct-7B` or similar models.

### Offline ASCII evaluation with Qwen2.5-VL

To run the 3B Qwen2.5-VL checkpoint locally:

1. Install the optional dependencies (CPU-only environments need the `+cpu` PyTorch wheels):

   ```bash
   uv pip install 'torch==2.2.2+cpu' --index-url https://download.pytorch.org/whl/cpu
   uv pip install transformers qwen-vl-utils
   ```

2. Download the model weights via the Hugging Face CLI (or mirror the files manually if direct access is blocked):

   ```bash
   huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/qwen25vl-3b --local-dir-use-symlinks False
   ```

3. Run evaluation with the `transformers` backend, pointing `--model` at the local directory:

   ```bash
   uv run python -m frozenlake_benchmark.src.run_vlm_eval \
     --backend transformers \
     --model ./models/qwen25vl-3b \
     --dataset frozenlake_benchmark/data/test.jsonl \
     --output frozenlake_benchmark/data/eval/qwen25vl_3b_local.jsonl
   ```

The resulting JSONL summarises exact match, progress rate, and invalid action rate alongside the raw completions for post-hoc analysis.

For quick experiments we also ship a lightweight `sandbox.jsonl` split (8 layouts per grid size). The file is small enough for rapid iteration yet preserves diverse board configurations. Results produced by evaluation scripts can be stored under `frozenlake_benchmark/data/eval/` for tracking progress.

## Metrics

The following metrics are implemented in `src/metrics.py`:

* **Exact Match (EM)** – fraction of predictions matching the optimal action sequence exactly.
* **Progress Rate (PR)** – degree of overlap between the predicted and optimal paths.
* **Invalid Action Rate (IAR)** – fraction of illegal moves (off-grid or into holes).

## Notebook Report

`report.ipynb` provides a starter notebook to summarize evaluation results and visualize qualitative examples. Populate it with results saved from `run_vlm_eval.py` to compare ASCII and image performance.

## License

This repository is intended for research and benchmarking purposes. Adapt or extend as needed for new environments (Maze, MiniBehavior, etc.).
