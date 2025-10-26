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
* [Pillow](https://pypi.org/project/Pillow/) for rendering PNG grids
* [huggingface_hub](https://pypi.org/project/huggingface-hub/) for remote inference (optional)

Install dependencies via:

```bash
pip install pillow huggingface_hub pandas
```

## Dataset Generation

Run the generator to produce the train/test splits:

```bash
./frozenlake_benchmark/scripts/gen_all.sh --output-dir frozenlake_benchmark/data
```

This command creates `train.jsonl` and `test.jsonl`. Default grid sizes are 3×3 to 6×6 with 1,000 train and 250 test layouts per size (note that extremely small grids have a limited number of unique solvable layouts, so you may need to override counts for those sizes). Command-line flags allow you to customize grid sizes, dataset sizes, hole probability, and random seed. Pass `--save-renderings` if you also want ASCII/PNG dumps in `data/render/` (handy for qualitative inspection, but large and therefore ignored by Git).

Use `--train-counts`/`--test-counts` to provide explicit `SIZE:COUNT` overrides when you want the total dataset to follow a specific split. For example, the committed dataset was generated via:

```bash
python -m frozenlake_benchmark.src.generate_dataset \
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
  "optimal_actions": ["DOWN", "DOWN", "DOWN", "RIGHT", "RIGHT"],
  "optimal_action_sequences": [
    ["DOWN", "DOWN", "DOWN", "RIGHT", "RIGHT"],
    ["DOWN", "DOWN", "RIGHT", "DOWN", "RIGHT"],
    ["DOWN", "RIGHT", "DOWN", "DOWN", "RIGHT"],
    ["RIGHT", "DOWN", "DOWN", "DOWN", "RIGHT"]
  ],
  "path_coords": [[0, 1], [1, 1], [2, 1], [3, 1], [3, 2], [3, 3]]
}
```

The start (`S`) and goal (`G`) tiles are sampled uniformly across the grid (subject to being distinct) so the agent must reason over diverse viewpoints. If renderings are saved, the records also include `ascii_path` and `image_path` fields pointing to the corresponding assets.

## Evaluation

Use `run_vlm_eval.py` to compare models across modalities. The script includes a mock backend (which predicts optimal actions for sanity checks) and a Hugging Face backend for real inference against hosted models:

```bash
# Dry run using the built-in oracle
./frozenlake_benchmark/scripts/eval_ascii.sh --backend mock --limit 5

# Remote evaluation with Hugging Face (requires API token)
HUGGING_FACE_HUB_TOKEN=... ./frozenlake_benchmark/scripts/eval_ascii.sh \
  --backend huggingface \
  --model Qwen/Qwen2.5-Instruct \
  --token "$HUGGING_FACE_HUB_TOKEN" \
  --dataset frozenlake_benchmark/data/test.jsonl \
  --limit 10
```

Image-only evaluation requires a local VLM runtime. Export the dataset and run in an environment with GPU access to evaluate `Qwen2.5-VL-Instruct-7B` or similar models.

## Metrics

The following metrics are implemented in `src/metrics.py`:

* **Exact Match (EM)** – fraction of predictions matching the optimal action sequence exactly.
* **Progress Rate (PR)** – degree of overlap between the predicted and optimal paths.
* **Invalid Action Rate (IAR)** – fraction of illegal moves (off-grid or into holes).

## Notebook Report

`report.ipynb` provides a starter notebook to summarize evaluation results and visualize qualitative examples. Populate it with results saved from `run_vlm_eval.py` to compare ASCII and image performance.

## License

This repository is intended for research and benchmarking purposes. Adapt or extend as needed for new environments (Maze, MiniBehavior, etc.).
