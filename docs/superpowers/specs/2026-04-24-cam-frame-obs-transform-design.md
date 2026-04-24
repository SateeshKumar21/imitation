# Camera-Frame Observation Transform at Rollout and Eval Time

## Problem

Policies trained on `*_cam.h5` datasets expect EEF observations `obs/left` and `obs/right` in the camera optical frame (as produced by `action-transformation/transform_il_data_to_camera_frame.py`). At inference time today:

- **Rollout** (`imitation/scripts/rollout_diffusion_policy.py`): `TiagoGym` emits `obs['left']`/`obs['right']` in `base_footprint`. Actions are inverse-transformed (camera → torso) when `--cam_actions` is set, but the incoming observations are fed to the policy **unchanged in base frame**. This is a train/inference frame mismatch.
- **Eval** (`imitation/scripts/evaluate_model.py`): Works correctly when `config.data_config.data` points at a `*_cam.h5`. If pointed at a base-frame h5 while evaluating a cam-trained checkpoint, obs are fed in the wrong frame.

The policy's cam-frame action output is already handled on both sides; only the obs input side is missing.

## Scope

- `obs/left`, `obs/right` — `(8,) [xyz, qxyzw, grip]` — base → camera when the policy is in camera frame.
- Images are already camera-frame (no change).
- The gripper scalar (column 7) is frame-independent (passes through, as in `transform_eef_pose`).
- Other obs keys (`privileged_info`, head image, etc.) pass through.

Out of scope:

- Changing what rollout writes to HDF5. The existing asymmetry (cam-frame action saved alongside base-frame obs) stays.
- Training / dataset code. Only inference-time code changes.
- Any obs key other than `left` / `right`.

## Design

### 1. Shared helpers (in `action-transformation/`)

`transform_il_data_to_camera_frame.py` already has `transform_eef_pose(eef_base, T_cam_base)` for `(T, 8)` arrays. It's safe to call on `(1, 8)`, so inference sites will reshape `(8,) → (1, 8) → (8,)` rather than duplicating logic.

Add `inverse_transform_eef_pose(eef_cam, T_cam_base)` next to `transform_eef_pose` — the camera → base inverse used only by the roundtrip verification. Symmetric to `inverse_transform_actions` / `transform_actions` in the same repo.

### 2. Rollout (`imitation/scripts/rollout_diffusion_policy.py`)

- Extend `_load_cam_torso_transform` to also return `T_cam_base`. Rename to `_load_cam_transforms` returning `(T_cam_base, T_cam_torso)`.
- In the `cam_actions` branch of the main loop: after computing `raw_obs` and before passing `obs` to `model.get_action`, overwrite `obs['left']` and `obs['right']` with their camera-frame versions (when present). `raw_obs['left']` / `raw_obs['right']` stay in base frame so the saved HDF5 matches raw-collect format.
- One-line stdout at startup confirms obs transform is active, mirroring the existing cam-actions log line.

### 3. Eval (`imitation/scripts/evaluate_model.py`)

Auto-detect from the h5 rather than adding a new flag:

- Open each main data h5 and read `f.attrs.get("frame")`. If `"camera"`, obs are already transformed — no-op.
- If the attr is absent or not `"camera"` **and** `--cam_actions` is set, apply `transform_eef_pose` to the `(8,)` obs vectors for `left` / `right` per step, before `model.get_action`.
- Print one line per h5 at startup: `[cam_obs] transforming obs/left,obs/right base->camera for <path>` or `[cam_obs] <path> already in camera frame (frame attr)`.

Rationale for auto-detect: the data self-identifies via the `frame` root attr that `transform_il_data_to_camera_frame.py` writes. Users can't forget a flag, and the same command works against either layout.

## Verification

### V1 — Numerical roundtrip

Standalone script in `action-transformation/` or a quick inline check. For each demo's `obs/left`, `obs/right` in a base-frame h5:

1. `eef_cam   = transform_eef_pose(eef_base, T_cam_base)`
2. `eef_base' = inverse_transform_eef_pose(eef_cam, T_cam_base)`
3. Assert `max(|eef_base' − eef_base|) < 1e-10` for xyz and gripper columns.
4. Assert quaternion columns match up to ± sign: `min(|q' − q|, |q' + q|)` across each row `< 1e-10`.

### V2 — MSE comparison on a cam-trained policy

Using checkpoint `experiments/milk_vp123_notp_cam/weights/weights_ep950.pth` and config `imitation.config.model.exp_milk_joint_vp123_notp_cam`, on a handful of demos (e.g. `--demos 11 12 13 --num_eval_points 10`):

- **Run A (baseline):** eval against `data_1_to_37_notp_cam.h5` (policy's training frame). Gives the correct per-dim MSE.
- **Run B (broken):** eval against `data_1_to_37_notp.h5` (base frame) with the obs-transform code path disabled (or before this change lands). Expect MSE ≫ A because obs are in the wrong frame.
- **Run C (fix):** eval against `data_1_to_37_notp.h5` with the obs-transform active. Expect per-dim MSE ≈ A.

**Pass criteria:**

- `|mean_MSE_C − mean_MSE_A| / mean_MSE_A < 1%` across the evaluated demos.
- `mean_MSE_B > 2 × mean_MSE_A` (sanity check that we're actually measuring an effect).
- Per-demo MSE agreement C vs A within a small tolerance driven by float16 image / pose storage.

If B fails to show elevated MSE, something else is masking the frame mismatch and the verification is inconclusive — stop and investigate before claiming the fix works.

## File-level change summary

- `action-transformation/transform_il_data_to_camera_frame.py` — add `inverse_transform_eef_pose`.
- `imitation/scripts/rollout_diffusion_policy.py` — return `T_cam_base` from helper; transform `obs['left']` / `obs['right']` before policy call when `cam_actions`.
- `imitation/scripts/evaluate_model.py` — auto-detect `frame` attr per h5; apply obs transform when needed.
- (Verification, not committed to main tree) — short script or ad-hoc run that executes V1 + V2 and reports the three MSE numbers.
