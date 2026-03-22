---
name: evaluate-model
description: Evaluate a trained imitation learning model against ground truth actions. Use when the user wants to test model performance, compare checkpoints, or check prediction accuracy on demo data.
argument-hint: <config_module> [--epoch N] [--demos 11 12 ...] [--num_eval_points 10]
allowed-tools: Bash, Read, Glob, Grep
---

# Evaluate Model Skill

Evaluate a trained model's action predictions against ground truth across demos.

## What this does

Runs `imitation/scripts/evaluate_model.py` which:
1. Loads the config to find data paths, obs keys, and policy class
2. Auto-discovers the matching checkpoint (latest epoch, or a specific one)
3. Feeds observations at uniformly spaced timesteps across each demo
4. Compares predicted actions to ground truth and computes MSE/MAE

## How to run

Parse the user's arguments from `$ARGUMENTS`. The config module path is required.
Map common shorthand: if the user passes just a config filename like `diffusion_policy_ws5_27th_feb_exp3`, expand it to `imitation.config.model.diffusion_policy_ws5_27th_feb_exp3`.

Build and run the command:

```
conda run -n imitation python -m imitation.scripts.evaluate_model \
    --config <config_module_path> \
    [--ckpt <path>] \
    [--epoch <N>] \
    [--experiment_dir <path>] \
    [--demos <idx1> <idx2> ...] \
    [--num_eval_points <N>] \
    [--quiet]
```

Use a timeout of 600000ms (10 min) since evaluation can be slow.

## After running

Summarize results for the user:
- Show the per-demo MSE/MAE table
- Highlight outlier demos (>2x the mean MSE)
- Note which action dimensions have highest error
- Show a few sample predicted vs GT action pairs from both good and bad demos
- If the model has obvious failure modes (e.g. predicting zeros at transitions), call them out
