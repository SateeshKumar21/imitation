"""
Evaluate a trained model at uniformly spaced steps across demos in an HDF5 dataset.

Reads the config to determine:
  - data path(s)
  - observation keys + modalities
  - policy class

Finds the best (latest epoch) checkpoint in the experiment directory
and evaluates it against ground truth actions.

Usage:
    python -m imitation.scripts.evaluate_model --config <config_module_path> [options]

Example:
    python -m imitation.scripts.evaluate_model \
        --config imitation.config.model.diffusion_policy_ws5_27th_feb_exp3 \
        --num_eval_points 10 \
        --demos 11 12 13
"""
import argparse
import importlib
import os
import re
import sys
from collections import deque, OrderedDict

import h5py
import numpy as np
import torch


def find_checkpoint(experiment_dir, epoch=None):
    """Find a checkpoint in the experiment directory.
    If epoch is given, look for that specific epoch. Otherwise, pick the latest.
    """
    weights_dir = os.path.join(experiment_dir, "weights")
    if not os.path.isdir(weights_dir):
        return None

    pattern = re.compile(r"weights_ep(\d+)\.pth$")
    candidates = []
    for f in os.listdir(weights_dir):
        m = pattern.match(f)
        if m:
            candidates.append((int(m.group(1)), os.path.join(weights_dir, f)))

    if not candidates:
        return None

    if epoch is not None:
        for ep, path in candidates:
            if ep == epoch:
                return path
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_experiment_dir(config_module_name, base_dir="experiments"):
    """Heuristic: derive experiment directory name from config module name."""
    # e.g. "imitation.config.model.diffusion_policy_ws5_27th_feb_exp3"
    #   -> look for experiments/diffpol_ws5_27th_feb_exp3_*
    short_name = config_module_name.split(".")[-1]

    # Try common naming patterns
    prefixes = [
        short_name,
        short_name.replace("diffusion_policy", "diffpol"),
    ]

    if os.path.isdir(base_dir):
        for entry in sorted(os.listdir(base_dir)):
            entry_lower = entry.lower()
            for prefix in prefixes:
                if prefix.lower() in entry_lower:
                    candidate = os.path.join(base_dir, entry)
                    if find_checkpoint(candidate) is not None:
                        return candidate
    return None


def get_obs_keys_and_modalities(observation_config):
    """Extract obs keys to use and their modalities from the observation config."""
    obs_keys = []
    keys_to_modality = OrderedDict()
    for modality in ["low_dim", "rgb", "depth"]:
        for key in observation_config.obs.get(modality, []):
            obs_keys.append(key)
            keys_to_modality[key] = modality
    return obs_keys, keys_to_modality


def list_demos(hdf5_file):
    """List all demo keys in the HDF5 file, sorted by index."""
    if "data" not in hdf5_file:
        return []
    demos = list(hdf5_file["data"].keys())
    demos.sort(key=lambda x: int(x.split("_")[-1]))
    return demos


def evaluate(config, ckpt_path, demo_indices, num_eval_points, verbose=True):
    """Run evaluation and return results dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_class = config.policy_config.policy_class
    model = policy_class.load_weights(ckpt_path)
    model.to(device)

    obs_keys, _ = get_obs_keys_and_modalities(config.observation_config)
    n_obs_steps = getattr(model, "n_obs_steps", 1)

    data_paths = config.data_config.data
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    results = {}

    for data_path in data_paths:
        hdf5 = h5py.File(data_path, "r")
        available_demos = list_demos(hdf5)
        available_indices = [int(d.split("_")[-1]) for d in available_demos]

        if demo_indices is not None:
            run_indices = [i for i in demo_indices if i in available_indices]
        else:
            run_indices = available_indices

        for demo_idx in run_indices:
            demo_key = f"data/demo_{demo_idx}"
            demo = hdf5[demo_key]
            demo_len = demo["actions"].shape[0]
            action_dim = demo["actions"].shape[1]

            eval_indices = np.linspace(0, demo_len - 1, num_eval_points, dtype=int)

            if verbose:
                print(f"\n{'='*70}")
                print(f"Demo {demo_idx} (length={demo_len}), eval at: {eval_indices.tolist()}")
                print(f"{'='*70}")

            predicted = []
            gt_actions = []

            for target_idx in eval_indices:
                # Reset model queues for each eval point
                model.obs_queue = deque(maxlen=n_obs_steps)
                model.action_queue = deque()

                # Feed a window of observations ending at target_idx
                window_start = max(0, target_idx - n_obs_steps + 1)
                for t in range(window_start, target_idx + 1):
                    obs = {}
                    for key in obs_keys:
                        obs[key] = np.array(demo[f"obs/{key}"][t])
                    with torch.no_grad():
                        action = model.get_action(obs, batched=False)

                gt_action = np.array(demo["actions"][target_idx])
                pred_action = action.flatten()

                predicted.append(pred_action)
                gt_actions.append(gt_action)

                if verbose:
                    err = np.abs(pred_action - gt_action)
                    fmt = lambda a: np.array2string(a, precision=4, suppress_small=True, max_line_width=200)
                    print(f"  idx={target_idx:>4d}  pred={fmt(pred_action)}")
                    print(f"          GT  ={fmt(gt_action)}")
                    print(f"          |err|={fmt(err)}")

            predicted = np.array(predicted)
            gt_actions = np.array(gt_actions)
            mse = float(np.mean((predicted - gt_actions) ** 2))
            mae = float(np.mean(np.abs(predicted - gt_actions)))
            per_dim_mse = np.mean((predicted - gt_actions) ** 2, axis=0)

            results[demo_idx] = {
                "demo_len": demo_len,
                "n_eval_points": len(eval_indices),
                "mse": mse,
                "mae": mae,
                "per_dim_mse": per_dim_mse,
            }

            if verbose:
                print(f"  => MSE={mse:.6f}  MAE={mae:.6f}")

        hdf5.close()

    return results


def print_summary(results):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    mses, maes, all_per_dim = [], [], []
    for demo_idx in sorted(results.keys()):
        r = results[demo_idx]
        print(f"  Demo {demo_idx:>3d} (len={r['demo_len']:>3d}): MSE={r['mse']:.6f}  MAE={r['mae']:.6f}")
        mses.append(r["mse"])
        maes.append(r["mae"])
        all_per_dim.append(r["per_dim_mse"])

    if mses:
        print(f"\n  Overall avg MSE: {np.mean(mses):.6f} (+/- {np.std(mses):.6f})")
        print(f"  Overall avg MAE: {np.mean(maes):.6f} (+/- {np.std(maes):.6f})")
        avg_per_dim = np.mean(all_per_dim, axis=0)
        print(f"  Per-dim avg MSE: {np.array2string(avg_per_dim, precision=6)}")
        print(f"  Demos evaluated: {len(mses)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model against GT actions.")
    parser.add_argument("--config", type=str, required=True,
                        help="Config module path, e.g. imitation.config.model.diffusion_policy_ws5_27th_feb_exp3")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint. If not given, auto-discovers from experiment dir.")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch to evaluate. Used with auto-discovery.")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Experiment directory to search for checkpoints.")
    parser.add_argument("--demos", type=int, nargs="+", default=None,
                        help="Demo indices to evaluate. Default: all demos in the dataset.")
    parser.add_argument("--num_eval_points", type=int, default=10,
                        help="Number of uniformly spaced eval points per demo.")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print summary, not per-step details.")
    args = parser.parse_args()

    # Load config
    config_module = importlib.import_module(args.config)
    config = config_module.config

    # Find checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None:
        exp_dir = args.experiment_dir
        if exp_dir is None:
            exp_dir = find_experiment_dir(args.config)
        if exp_dir is None:
            print(f"ERROR: Could not auto-discover experiment dir for '{args.config}'.")
            print("       Pass --ckpt or --experiment_dir explicitly.")
            sys.exit(1)
        ckpt_path = find_checkpoint(exp_dir, epoch=args.epoch)
        if ckpt_path is None:
            print(f"ERROR: No checkpoint found in '{exp_dir}'.")
            sys.exit(1)

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Eval points per demo: {args.num_eval_points}")
    if args.demos:
        print(f"Demos: {args.demos}")
    else:
        print(f"Demos: all")

    results = evaluate(
        config=config,
        ckpt_path=ckpt_path,
        demo_indices=args.demos,
        num_eval_points=args.num_eval_points,
        verbose=not args.quiet,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
