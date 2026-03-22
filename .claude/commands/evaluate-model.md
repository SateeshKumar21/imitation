Evaluate a trained imitation learning model against ground truth actions.

Parse the user's arguments from $ARGUMENTS. The config module path is required.
Map common shorthand: if the user passes just a config filename like `diffusion_policy_ws5_27th_feb_exp3`, expand it to `imitation.config.model.diffusion_policy_ws5_27th_feb_exp3`.

Build and run the command using the Bash tool:

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

After running, summarize results for the user:
- Show the per-demo MSE/MAE table
- Highlight outlier demos (>2x the mean MSE)
- Note which action dimensions have highest error
- Show a few sample predicted vs GT action pairs from both good and bad demos
- If the model has obvious failure modes (e.g. predicting zeros at transitions), call them out
