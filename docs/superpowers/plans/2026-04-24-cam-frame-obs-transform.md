# Camera-Frame Observation Transform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `obs/left` and `obs/right` from `base_footprint` to the camera optical frame at rollout and eval time, so cam-trained policies see observations in their training frame.

**Architecture:** Reuse `transform_eef_pose` from `action-transformation/transform_il_data_to_camera_frame.py` on single-step obs. Add a camera→base inverse for the verification roundtrip. In rollout, transform `obs['left']`/`obs['right']` in-place before `model.get_action` when `--cam_actions` is set. In eval, auto-detect via each h5's `frame` root attr — if not `"camera"` and `--cam_actions` is set, transform per-step.

**Tech Stack:** numpy, scipy (`Rotation`), h5py. No pytest — this codebase uses standalone verification scripts with asserts and prints (see `inverse_transform_actions.py:verify`).

**Spec:** [docs/superpowers/specs/2026-04-24-cam-frame-obs-transform-design.md](../specs/2026-04-24-cam-frame-obs-transform-design.md)

---

## File Structure

**Action-transformation repo** (`/home/ec2-user/action-transformation/`):
- Modify `transform_il_data_to_camera_frame.py` — add `inverse_transform_eef_pose`.
- Create `verify_obs_roundtrip.py` — V1 numerical roundtrip verification script.

**Imitation repo** (`/home/ec2-user/imitation/`):
- Modify `imitation/scripts/rollout_diffusion_policy.py` — extend transform helper, transform obs in main loop.
- Modify `imitation/scripts/evaluate_model.py` — detect `frame` attr per h5, transform obs when needed.

No tests directory exists in either repo. Follow the `inverse_transform_actions.py` pattern: standalone `main()` with `--help`, `argparse`, `assert`, and prints summarizing per-demo and aggregate numbers.

---

## Task 1: Add `inverse_transform_eef_pose` to action-transformation

**Files:**
- Modify: `/home/ec2-user/action-transformation/transform_il_data_to_camera_frame.py`

**Context:** `transform_eef_pose` at [transform_il_data_to_camera_frame.py:66](/home/ec2-user/action-transformation/transform_il_data_to_camera_frame.py#L66) converts `(T, 8) [xyz, qxyzw, grip]` base → camera. We need the inverse for the roundtrip verification. The gripper passes through; xyz and quaternion are inverted via `T_cam_base⁻¹ = T_base_cam`.

- [ ] **Step 1: Add `inverse_transform_eef_pose` function**

Insert directly below `transform_eef_pose` (after [line 80](/home/ec2-user/action-transformation/transform_il_data_to_camera_frame.py#L80)):

```python
def inverse_transform_eef_pose(eef_cam, T_cam_base):
    """(T, 8) EEF poses [xyz, qxyzw, grip], camera -> base. Inverse of transform_eef_pose."""
    out = np.asarray(eef_cam, dtype=np.float64).copy()
    xyz = out[:, :3]
    quat = out[:, 3:7]
    grip = out[:, 7:8]

    T_base_cam = invert_transform(T_cam_base)
    R_bc = T_base_cam[:3, :3]
    t_bc = T_base_cam[:3, 3]

    xyz_base = xyz @ R_bc.T + t_bc
    rot_base = R.from_matrix(R_bc) * R.from_quat(quat)
    quat_base = rot_base.as_quat()

    return np.concatenate([xyz_base, quat_base, grip], axis=1)
```

- [ ] **Step 2: Smoke-test the new function**

Run from the `action-transformation/` directory:

```bash
cd /home/ec2-user/action-transformation && python -c "
import numpy as np
from calibration import T_BASE_CAMERA
from transform_il_data_to_camera_frame import transform_eef_pose, inverse_transform_eef_pose, invert_transform
T_cam_base = invert_transform(T_BASE_CAMERA)
rng = np.random.default_rng(0)
xyz = rng.normal(size=(5, 3))
q = rng.normal(size=(5, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
grip = rng.uniform(size=(5, 1))
eef = np.concatenate([xyz, q, grip], axis=1)
eef_cam = transform_eef_pose(eef, T_cam_base)
eef_back = inverse_transform_eef_pose(eef_cam, T_cam_base)
xyz_err = np.max(np.abs(eef_back[:, :3] - eef[:, :3]))
# Quaternion sign ambiguity: take min of |q' - q| and |q' + q| per row.
q_err = np.max(np.minimum(np.abs(eef_back[:, 3:7] - eef[:, 3:7]).max(axis=1),
                           np.abs(eef_back[:, 3:7] + eef[:, 3:7]).max(axis=1)))
g_err = np.max(np.abs(eef_back[:, 7] - eef[:, 7]))
print(f'xyz_err={xyz_err:.2e}  q_err={q_err:.2e}  g_err={g_err:.2e}')
assert xyz_err < 1e-12 and q_err < 1e-12 and g_err < 1e-12
print('OK')
"
```

Expected: `xyz_err=<~1e-15>  q_err=<~1e-15>  g_err=0.00e+00` followed by `OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/ec2-user/action-transformation && git add transform_il_data_to_camera_frame.py && git commit -m "add inverse_transform_eef_pose for cam->base EEF pose"
```

---

## Task 2: Write roundtrip verification script

**Files:**
- Create: `/home/ec2-user/action-transformation/verify_obs_roundtrip.py`

**Context:** Per the spec's V1 — roundtrip obs through `transform_eef_pose` and back, assert max-abs-error below 1e-10. Quaternions have a `q` vs `-q` sign ambiguity so compare `min(|q'-q|, |q'+q|)` elementwise.

- [ ] **Step 1: Create the verification script**

Write the full file:

```python
"""
Roundtrip verification for the base<->camera EEF observation transform.

Reads a base-frame IL h5, applies transform_eef_pose then inverse_transform_eef_pose,
and asserts that the recovered obs matches the original within tight tolerance.

Supports both layouts (flat /obs or packed /data/demo_*/obs).
"""

import argparse

import h5py
import numpy as np

from calibration import T_BASE_CAMERA
from transform_il_data_to_camera_frame import (
    detect_layout,
    invert_transform,
    transform_eef_pose,
    inverse_transform_eef_pose,
)


def _iter_eef(f):
    """Yield (name, arm, (T,8) array) across layouts."""
    layout, eps = detect_layout(f)
    for ep in eps:
        if "obs" not in ep:
            continue
        for arm in ("left", "right"):
            if arm in ep["obs"]:
                data = ep["obs"][arm][...]
                if data.ndim == 2 and data.shape[1] == 8:
                    yield ep.name, arm, data


def _roundtrip_errors(eef_base, T_cam_base):
    eef_cam = transform_eef_pose(eef_base, T_cam_base)
    eef_back = inverse_transform_eef_pose(eef_cam, T_cam_base)
    ref = np.asarray(eef_base, dtype=np.float64)

    xyz_err = np.max(np.abs(eef_back[:, :3] - ref[:, :3]))
    q_plus = np.abs(eef_back[:, 3:7] - ref[:, 3:7]).max(axis=1)
    q_minus = np.abs(eef_back[:, 3:7] + ref[:, 3:7]).max(axis=1)
    q_err = float(np.max(np.minimum(q_plus, q_minus)))
    g_err = float(np.max(np.abs(eef_back[:, 7] - ref[:, 7])))
    return xyz_err, q_err, g_err


def verify(h5_path, calib_npz=None, atol=1e-10):
    if calib_npz is not None:
        d = np.load(calib_npz)
        T_base_camera = np.asarray(d["T_base_camera"], dtype=np.float64)
    else:
        T_base_camera = T_BASE_CAMERA.copy()
    T_cam_base = invert_transform(T_base_camera)

    worst_xyz = worst_q = worst_g = 0.0
    total = 0
    bad = 0
    with h5py.File(h5_path, "r") as f:
        for name, arm, data in _iter_eef(f):
            xyz_err, q_err, g_err = _roundtrip_errors(data, T_cam_base)
            print(f"  {name}/obs/{arm}  T={data.shape[0]}  "
                  f"xyz={xyz_err:.2e}  q={q_err:.2e}  g={g_err:.2e}")
            worst_xyz = max(worst_xyz, xyz_err)
            worst_q = max(worst_q, q_err)
            worst_g = max(worst_g, g_err)
            total += 1
            if max(xyz_err, q_err, g_err) > atol:
                bad += 1

    print()
    print(f"arms checked: {total}   above tol ({atol}): {bad}")
    print(f"worst xyz={worst_xyz:.2e}  worst q={worst_q:.2e}  worst g={worst_g:.2e}")
    assert bad == 0, f"{bad} arm(s) exceeded tolerance {atol}"
    print("OK")
    return worst_xyz, worst_q, worst_g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="h5 with base-frame obs (flat or packed).")
    p.add_argument("--calib", default=None,
                   help=".npz with T_base_camera override (else uses calibration.py defaults).")
    p.add_argument("--atol", type=float, default=1e-10, help="max-abs-error tolerance per arm.")
    args = p.parse_args()

    print(f"[roundtrip] {args.input}")
    verify(args.input, calib_npz=args.calib, atol=args.atol)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the verification against a known base-frame h5**

```bash
cd /home/ec2-user/action-transformation && python verify_obs_roundtrip.py \
  --input /home/ec2-user/milk_bottle_pnp/mar_28/original_viewpoint3_processed_il_data/data_1_to_37_notp.h5
```

Expected: per-demo per-arm lines with `xyz`, `q`, `g` errors all ≤ ~1e-13, final `OK` line.

If any arm exceeds 1e-10: stop. Do not proceed to downstream tasks until the roundtrip is clean — a non-clean roundtrip means the forward/inverse are not actually inverses.

- [ ] **Step 3: Commit**

```bash
cd /home/ec2-user/action-transformation && git add verify_obs_roundtrip.py && git commit -m "add obs roundtrip verification script"
```

---

## Task 3: Rollout — transform obs before policy call

**Files:**
- Modify: `/home/ec2-user/imitation/imitation/scripts/rollout_diffusion_policy.py`

**Context:** `_load_cam_torso_transform` at [rollout_diffusion_policy.py:36-46](/home/ec2-user/imitation/imitation/scripts/rollout_diffusion_policy.py#L36-L46) computes `T_cam_torso` only. We need `T_cam_base` too. The main loop obs transform happens between the `raw_obs` snapshot at [line 245](/home/ec2-user/imitation/imitation/scripts/rollout_diffusion_policy.py#L245) and the `model.get_action` call at [line 250](/home/ec2-user/imitation/imitation/scripts/rollout_diffusion_policy.py#L250).

- [ ] **Step 1: Update the transform-loader to return both matrices**

Replace the function at [lines 36-46](/home/ec2-user/imitation/imitation/scripts/rollout_diffusion_policy.py#L36-L46):

```python
def _load_cam_transforms(calib_npz=None):
    """Return (T_cam_base, T_cam_torso) used for cam-frame obs/action transforms."""
    if calib_npz is not None:
        d = np.load(calib_npz)
        T_base_torso = np.asarray(d["T_base_torso"], dtype=np.float64)
        T_base_camera = np.asarray(d["T_base_camera"], dtype=np.float64)
    else:
        T_base_torso = T_BASE_TORSO.copy()
        T_base_camera = T_BASE_CAMERA.copy()
    T_cam_base = invert_transform(T_base_camera)
    T_cam_torso = T_cam_base @ T_base_torso
    return T_cam_base, T_cam_torso
```

- [ ] **Step 2: Import `transform_eef_pose` and update the call site**

At [line 33](/home/ec2-user/imitation/imitation/scripts/rollout_diffusion_policy.py#L33), change:

```python
from transform_il_data_to_camera_frame import invert_transform
```

to:

```python
from transform_il_data_to_camera_frame import invert_transform, transform_eef_pose
```

At [lines 101-104](/home/ec2-user/imitation/imitation/scripts/rollout_diffusion_policy.py#L101-L104), change:

```python
    T_cam_torso = _load_cam_torso_transform(calib) if cam_actions else None
    if cam_actions:
        print(f"[cam_actions] inverse-transforming camera-frame policy actions to torso frame "
              f"(calib={'builtin' if calib is None else calib})")
```

to:

```python
    T_cam_base = T_cam_torso = None
    if cam_actions:
        T_cam_base, T_cam_torso = _load_cam_transforms(calib)
        print(f"[cam_actions] transforming obs/left,obs/right base->camera and "
              f"inverse-transforming camera-frame policy actions to torso frame "
              f"(calib={'builtin' if calib is None else calib})")
```

- [ ] **Step 3: Transform obs in the main loop before the policy call**

At [lines 244-250](/home/ec2-user/imitation/imitation/scripts/rollout_diffusion_policy.py#L244-L250), change:

```python
            # Keep raw obs for saving before preprocessing
            raw_obs = {k: obs[k] for k in obs_keys_to_save if k in obs}

            img = _crop_and_resize(obs['tiago_head_image']).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs['tiago_head_image'] = img
            policy_action = model.get_action(obs, batched=False, execute_horizon=execute_horizon).reshape(-1,)
```

to:

```python
            # Keep raw obs for saving before preprocessing (stays in base frame).
            raw_obs = {k: obs[k] for k in obs_keys_to_save if k in obs}

            img = _crop_and_resize(obs['tiago_head_image']).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs['tiago_head_image'] = img

            if cam_actions:
                for arm in ('left', 'right'):
                    if arm in obs:
                        arr = np.asarray(obs[arm]).reshape(1, -1)
                        if arr.shape[1] == 8:
                            obs[arm] = transform_eef_pose(arr, T_cam_base)[0].astype(np.asarray(obs[arm]).dtype)

            policy_action = model.get_action(obs, batched=False, execute_horizon=execute_horizon).reshape(-1,)
```

- [ ] **Step 4: Verify imports and no other callers broke**

```bash
cd /home/ec2-user/imitation && grep -n "_load_cam_torso_transform\|_load_cam_transforms" imitation/scripts/rollout_diffusion_policy.py
```

Expected: only the definition and single call site `_load_cam_transforms(calib)` — no stale references to the old name.

- [ ] **Step 5: Syntax check**

```bash
cd /home/ec2-user/imitation && python -c "import ast; ast.parse(open('imitation/scripts/rollout_diffusion_policy.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
cd /home/ec2-user/imitation && git add imitation/scripts/rollout_diffusion_policy.py && git commit -m "rollout: transform obs/left,obs/right base->camera when cam_actions"
```

---

## Task 4: Eval — auto-detect `frame` attr and transform obs when needed

**Files:**
- Modify: `/home/ec2-user/imitation/imitation/scripts/evaluate_model.py`

**Context:** `evaluate_model.py` reads obs straight from the h5 at [line 232](/home/ec2-user/imitation/imitation/scripts/evaluate_model.py#L232): `obs[key] = np.array(demo[f"obs/{key}"][t])`. If the h5 is already cam-frame (root attr `frame == "camera"`), do nothing. Otherwise, if `--cam_actions` was passed, transform `obs['left']` / `obs['right']` on the fly. The per-h5 decision needs to be made once when the h5 is opened.

- [ ] **Step 1: Extend the helper to return `T_cam_base` too**

Replace [lines 54-64](/home/ec2-user/imitation/imitation/scripts/evaluate_model.py#L54-L64):

```python
def _load_cam_torso_transform(calib_npz=None):
    """Return T_cam_torso used to inverse-transform camera-frame policy actions."""
    if calib_npz is not None:
        d = np.load(calib_npz)
        T_base_torso = np.asarray(d["T_base_torso"], dtype=np.float64)
        T_base_camera = np.asarray(d["T_base_camera"], dtype=np.float64)
    else:
        T_base_torso = T_BASE_TORSO.copy()
        T_base_camera = T_BASE_CAMERA.copy()
    T_cam_base = invert_transform(T_base_camera)
    return T_cam_base @ T_base_torso
```

with:

```python
def _load_cam_transforms(calib_npz=None):
    """Return (T_cam_base, T_cam_torso) for cam-frame obs/action transforms."""
    if calib_npz is not None:
        d = np.load(calib_npz)
        T_base_torso = np.asarray(d["T_base_torso"], dtype=np.float64)
        T_base_camera = np.asarray(d["T_base_camera"], dtype=np.float64)
    else:
        T_base_torso = T_BASE_TORSO.copy()
        T_base_camera = T_BASE_CAMERA.copy()
    T_cam_base = invert_transform(T_base_camera)
    T_cam_torso = T_cam_base @ T_base_torso
    return T_cam_base, T_cam_torso
```

- [ ] **Step 2: Update the import of the transform helpers**

At [line 51](/home/ec2-user/imitation/imitation/scripts/evaluate_model.py#L51), change:

```python
from inverse_transform_actions import inverse_transform_actions
```

to:

```python
from inverse_transform_actions import inverse_transform_actions
from transform_il_data_to_camera_frame import transform_eef_pose
```

- [ ] **Step 3: Update the single-line transform load in `evaluate`**

At [lines 168-173](/home/ec2-user/imitation/imitation/scripts/evaluate_model.py#L168-L173), change:

```python
    T_cam_torso = None
    if cam_actions:
        T_cam_torso = _load_cam_torso_transform(calib)
        if verbose:
            print(f"[cam_actions] inverse-transforming camera-frame policy actions to torso frame "
                  f"(calib={'builtin' if calib is None else calib})")
```

to:

```python
    T_cam_base = T_cam_torso = None
    if cam_actions:
        T_cam_base, T_cam_torso = _load_cam_transforms(calib)
        if verbose:
            print(f"[cam_actions] inverse-transforming camera-frame policy actions to torso frame "
                  f"(calib={'builtin' if calib is None else calib})")
```

- [ ] **Step 4: Detect the h5's frame and transform obs when loading**

At [lines 183-186](/home/ec2-user/imitation/imitation/scripts/evaluate_model.py#L183-L186), change:

```python
        for data_path in data_paths:
            hdf5 = h5py.File(data_path, "r")
            available_demos = list_demos(hdf5)
            available_indices = [int(d.split("_")[-1]) for d in available_demos]
```

to:

```python
        for data_path in data_paths:
            hdf5 = h5py.File(data_path, "r")
            available_demos = list_demos(hdf5)
            available_indices = [int(d.split("_")[-1]) for d in available_demos]

            # Decide once per h5 whether obs need on-the-fly base->camera transformation.
            h5_frame = hdf5.attrs.get("frame")
            if isinstance(h5_frame, bytes):
                h5_frame = h5_frame.decode()
            transform_obs = cam_actions and h5_frame != "camera"
            if verbose:
                if transform_obs:
                    print(f"[cam_obs] transforming obs/left,obs/right base->camera for {data_path}")
                elif cam_actions:
                    print(f"[cam_obs] {data_path} already in camera frame (frame attr); obs passthrough")
```

Then, at [lines 229-232](/home/ec2-user/imitation/imitation/scripts/evaluate_model.py#L229-L232), change:

```python
                    for t in range(window_start, target_idx + 1):
                        obs = {}
                        for key in obs_keys:
                            obs[key] = np.array(demo[f"obs/{key}"][t])
```

to:

```python
                    for t in range(window_start, target_idx + 1):
                        obs = {}
                        for key in obs_keys:
                            val = np.array(demo[f"obs/{key}"][t])
                            if transform_obs and key in ("left", "right") and val.shape == (8,):
                                val = transform_eef_pose(val.reshape(1, -1), T_cam_base)[0].astype(val.dtype)
                            obs[key] = val
```

- [ ] **Step 5: Syntax check**

```bash
cd /home/ec2-user/imitation && python -c "import ast; ast.parse(open('imitation/scripts/evaluate_model.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 6: Confirm no stale references to the old helper name**

```bash
cd /home/ec2-user/imitation && grep -n "_load_cam_torso_transform\|_load_cam_transforms" imitation/scripts/evaluate_model.py
```

Expected: only the definition and single call site `_load_cam_transforms(calib)`.

- [ ] **Step 7: Commit**

```bash
cd /home/ec2-user/imitation && git add imitation/scripts/evaluate_model.py && git commit -m "eval: auto-detect h5 frame attr and transform obs base->camera when needed"
```

---

## Task 5: V2 — MSE comparison (baseline vs broken vs fixed)

**Files:**
- No new source files. Runs the eval script three times and captures outputs.
- Create: `/tmp/cam_obs_verify/{run_a.log,run_b.log,run_c.log}` (scratch, not committed).

**Context:** Spec V2. Run the cam-trained VP123 policy against:
- **A**: `data_1_to_37_notp_cam.h5` (training frame, baseline MSE).
- **B**: `data_1_to_37_notp.h5` (base-frame obs, transform **disabled** — simulate pre-fix behavior).
- **C**: `data_1_to_37_notp.h5` (base-frame obs, transform **enabled** — current code).

Run B needs a way to disable the transform without reverting the fix. Simplest approach: stash the `frame` attr check by temporarily forcing `transform_obs = False` via an env var, documented below.

- [ ] **Step 1: Add a temporary debug-only env-var override to disable the eval transform**

Insert at [imitation/scripts/evaluate_model.py](imitation/scripts/evaluate_model.py) immediately after the `transform_obs = cam_actions and h5_frame != "camera"` line added in Task 4:

```python
            if os.environ.get("EVAL_DISABLE_OBS_TRANSFORM") == "1":
                transform_obs = False
                if verbose:
                    print(f"[cam_obs] EVAL_DISABLE_OBS_TRANSFORM=1 -> forcing obs passthrough for {data_path}")
```

This flag exists only for the V2 Run B comparison. It will be removed in Task 6 after verification passes.

- [ ] **Step 2: Syntax check and commit the temporary flag**

```bash
cd /home/ec2-user/imitation && python -c "import ast; ast.parse(open('imitation/scripts/evaluate_model.py').read()); print('OK')" && git add imitation/scripts/evaluate_model.py && git commit -m "eval: temporary EVAL_DISABLE_OBS_TRANSFORM flag for verification (remove after)"
```

Expected: `OK`, then commit success.

- [ ] **Step 3: Run A — baseline, cam-frame h5**

```bash
mkdir -p /tmp/cam_obs_verify && cd /home/ec2-user/imitation && \
python -m imitation.scripts.evaluate_model \
  --config imitation.config.model.exp_milk_joint_vp123_notp_cam \
  --cam_actions \
  --execute_horizon 8 \
  --num_eval_points 10 \
  --demos 11 12 13 \
  --quiet \
  --data /home/ec2-user/milk_bottle_pnp/mar_28/original_viewpoint3_processed_il_data/data_1_to_37_notp_cam.h5 \
  2>&1 | tee /tmp/cam_obs_verify/run_a.log
```

Expected: SUMMARY block with a `cam_MSE` column, finite values. Record `avg MSE` from the `[cam]` line.

- [ ] **Step 4: Run B — base-frame h5 with transform disabled (simulates pre-fix)**

```bash
cd /home/ec2-user/imitation && \
EVAL_DISABLE_OBS_TRANSFORM=1 python -m imitation.scripts.evaluate_model \
  --config imitation.config.model.exp_milk_joint_vp123_notp_cam \
  --cam_actions \
  --execute_horizon 8 \
  --num_eval_points 10 \
  --demos 11 12 13 \
  --quiet \
  --data /home/ec2-user/milk_bottle_pnp/mar_28/original_viewpoint3_processed_il_data/data_1_to_37_notp.h5 \
  2>&1 | tee /tmp/cam_obs_verify/run_b.log
```

Expected: startup log contains both `[cam_actions] inverse-transforming ...` AND `[cam_obs] EVAL_DISABLE_OBS_TRANSFORM=1 -> forcing obs passthrough`. SUMMARY `avg MSE` should be **substantially higher** than Run A.

- [ ] **Step 5: Run C — base-frame h5 with transform enabled (the fix)**

```bash
cd /home/ec2-user/imitation && \
python -m imitation.scripts.evaluate_model \
  --config imitation.config.model.exp_milk_joint_vp123_notp_cam \
  --cam_actions \
  --execute_horizon 8 \
  --num_eval_points 10 \
  --demos 11 12 13 \
  --quiet \
  --data /home/ec2-user/milk_bottle_pnp/mar_28/original_viewpoint3_processed_il_data/data_1_to_37_notp.h5 \
  2>&1 | tee /tmp/cam_obs_verify/run_c.log
```

Expected: startup log contains `[cam_obs] transforming obs/left,obs/right base->camera for ...`. SUMMARY `avg MSE` should match Run A closely.

- [ ] **Step 6: Check pass criteria**

Read the three `avg MSE` values from the logs. Print them and the ratios:

```bash
cd /tmp/cam_obs_verify && for f in run_a.log run_b.log run_c.log; do
  echo "=== $f ==="
  grep "avg MSE" "$f"
done
```

Pass criteria from the spec:
- `|mean_MSE_C − mean_MSE_A| / mean_MSE_A < 1%`  (fix matches baseline)
- `mean_MSE_B > 2 × mean_MSE_A`                     (mismatch is actually detectable)

Compute manually (or inline python) and print PASS/FAIL. If either criterion fails: stop. Do NOT remove the debug flag or claim success. Report the actual numbers and investigate.

---

## Task 6: Remove the temporary verification flag

**Files:**
- Modify: `/home/ec2-user/imitation/imitation/scripts/evaluate_model.py`

**Context:** Only run once Task 5 passes. The `EVAL_DISABLE_OBS_TRANSFORM` env var was for the V2 Run B comparison and is not part of the shipped behavior.

- [ ] **Step 1: Remove the env-var override block**

Delete the four lines added in Task 5 Step 1:

```python
            if os.environ.get("EVAL_DISABLE_OBS_TRANSFORM") == "1":
                transform_obs = False
                if verbose:
                    print(f"[cam_obs] EVAL_DISABLE_OBS_TRANSFORM=1 -> forcing obs passthrough for {data_path}")
```

- [ ] **Step 2: Syntax check and confirm the env var is fully gone**

```bash
cd /home/ec2-user/imitation && python -c "import ast; ast.parse(open('imitation/scripts/evaluate_model.py').read()); print('OK')" && grep -n EVAL_DISABLE_OBS_TRANSFORM imitation/scripts/evaluate_model.py || echo "no references"
```

Expected: `OK` followed by `no references`.

- [ ] **Step 3: Commit**

```bash
cd /home/ec2-user/imitation && git add imitation/scripts/evaluate_model.py && git commit -m "eval: remove temporary EVAL_DISABLE_OBS_TRANSFORM debug flag"
```

---

## Task 7: Final status summary

- [ ] **Step 1: Summarize what was verified**

Write the run_a / run_b / run_c MSE values, the ratios, and PASS/FAIL against the spec's pass criteria back to the user. Include which checkpoint, which config, and which demos were used. No files to commit at this step — the summary is conversational.
