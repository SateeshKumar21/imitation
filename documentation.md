# Imitation Learning Pipeline

## Project Structure

```
imitation/
├── algo/                   # Policy implementations (DiffusionPolicy, BC, etc.)
├── config/model/           # Experiment configs (one .py per experiment)
├── data/dataset.py         # SequenceDataset — reads HDF5 for training
├── envs/real_env.py        # TiagoGym — replays HDF5 demos as a Gym env
├── models/                 # Network architectures (ResNet, obs encoders, etc.)
├── scripts/
│   ├── train.py            # Training entry point
│   ├── test_real.py        # Single-demo interactive testing
│   └── evaluate_model.py   # Batch evaluation across demos
└── utils/                  # Logging, tensor ops, normalization
```

## HDF5 Data Format

```
data/
  demo_<N>/
    actions:            (T, 7)          float64   # [x, y, z, rot1, rot2, rot3, gripper]
    obs/
      tiago_head_image: (T, 224, 224, 3) float16
      left:             (T, 8)           float64
      right:            (T, 8)           float64   # optional
      privileged_info:  (T, 2)           float64   # optional
```

Only obs keys listed in the config's `observation_config.obs` are used. Extra keys are ignored.

## Training

### 1. Create a config

Configs live in `imitation/config/model/`. Each defines four blocks:

- **`train_config`** — epochs, batch size, save/log frequency
- **`data_config`** — HDF5 path(s), window size, action horizon
- **`policy_config`** — policy class, architecture, noise scheduler
- **`observation_config`** — obs keys, modalities (`low_dim`/`rgb`/`depth`), encoders, augmentations

### 2. Run training

```bash
conda run -n imitation python imitation/scripts/train.py \
    --config imitation/config/model/<config_name>.py \
    --exp_name <experiment_name>
```

Checkpoints saved to `experiments/<exp_name>/weights/weights_ep<N>.pth`. Logs to W&B (project: `3dmoma`).

## Evaluation

### Batch evaluation (recommended)

```bash
conda run -n imitation python -m imitation.scripts.evaluate_model \
    --config imitation.config.model.<config_name> \
    --num_eval_points 10
```

Auto-discovers the latest checkpoint. Evaluates at uniformly spaced timesteps across each demo. Reports per-demo MSE/MAE and per-dimension breakdown.

Key options: `--ckpt <path>`, `--epoch <N>`, `--demos 11 12 13`, `--quiet`.

Or via Claude Code: `/evaluate-model <config_name> [options]`

### Single-demo test

```bash
conda run -n imitation python imitation/scripts/test_real.py \
    --ckpt <path> --data_dir <path> --demo_index 11 --num_steps 50
```

## Interpreting Results

- **MSE < 0.001** / **MAE < 0.01** — good prediction accuracy
- **Per-dim MSE** — dims 3-5 (rotation) and dim 6 (gripper) typically dominate error
- **Outliers** (MSE > 2x mean) — may indicate distribution gaps or different task variants
