# System Identification Tutorial

This guide explains how to fit MuJoCo simulation parameters to real robot trajectories using the `mujoco-dex-bench` sysid pipeline.

## What is system identification here?

Sim-to-real transfer for dexterous manipulation requires matching simulation dynamics to the physical robot. Small errors in object mass, contact friction, or joint damping compound over a trajectory, producing large pose discrepancies. Our sysid module searches over these parameters to minimise mean-squared error (MSE) between simulated and recorded object poses.

## Collecting a real robot CSV

Record trajectories at your control frequency (default sim: 500 Hz effective with `n_substeps=10`, `timestep=0.002`). Required columns:

| Column | Description |
|--------|-------------|
| `t` | Timestamp or step index |
| `q0..q14` | 15 hand joint positions (rad) |
| `dq0..dq14` | 15 hand joint velocities (rad/s) |
| `obj_x, obj_y, obj_z` | Object position (m) |
| `obj_qw, obj_qx, obj_qy, obj_qz` | Object orientation quaternion |

Example header:

```
t,q0,q1,...,q14,dq0,...,dq14,obj_x,obj_y,obj_z,obj_qw,obj_qx,obj_qy,obj_qz
```

## Testing without a real robot

Generate a synthetic dataset from the simulator:

```python
from mujoco_dex_bench.sysid import generate_synthetic_dataset

generate_synthetic_dataset("DexBallRoll-v0", "synthetic.csv", n_steps=200, seed=0)
```

Or via CLI with auto-generation:

```bash
dex-sysid \
  --real-trajectory-csv synthetic.csv \
  --env-id DexBallRoll-v0 \
  --n-trials 200 \
  --generate-synthetic
```

## Running parameter fitting

```bash
dex-sysid \
  --real-trajectory-csv /path/to/robot_log.csv \
  --env-id DexBallRoll-v0 \
  --n-trials 200 \
  --output-xml fitted_params.xml
```

Python API:

```python
from mujoco_dex_bench.sysid import fit_params

result = fit_params(
    real_trajectory_csv="robot_log.csv",
    env_id="DexBallRoll-v0",
    n_trials=200,
    output_xml="fitted_params.xml",
)
print(f"MSE: {result['mse']:.6f}")
print(f"Mass scale: {result['mass_scale']:.3f}")
print(f"Friction: {result['friction']:.3f}")
```

## Search space

| Parameter | Range |
|-----------|-------|
| Mass scale | 0.5× – 2.0× base |
| Sliding friction | 0.3 – 3.0 |
| Damping scale | 0.5× – 2.0× base |

Optimisation uses `scipy.optimize.differential_evolution`.

## Interpreting output

**`fitted_params.xml`:** Copy of the scene XML with updated `mass`, `friction`, and `damping` attributes on the object geom and joints.

**`fitted_params.json`:** Best parameters and achieved MSE:

```json
{
  "mass_scale": 1.12,
  "mass_kg": 0.0896,
  "friction": 0.85,
  "damping_scale": 1.05,
  "mse": 0.000234,
  "geom_name": "ball_geom",
  "env_id": "DexBallRoll-v0"
}
```

Lower MSE indicates better trajectory alignment. Compare against a held-out validation trajectory to detect overfitting.

## Integrating fitted params into training

1. Replace the object geom attributes in the task scene XML under `mujoco_dex_bench/mjcf/`.
2. Or point `BaseDexEnv` to the fitted XML:

```python
from pathlib import Path
from mujoco_dex_bench.envs import BallRollEnv

env = BallRollEnv(xml_path=Path("fitted_params.xml"))
```

3. Enable domain randomisation around the fitted nominal values for robust policies:

```python
from mujoco_dex_bench.wrappers import DomainRandomisationWrapper

env = DomainRandomisationWrapper(env)
```

## Plotting sensitivity

After fitting, visualise parameter vs. MSE:

```python
from mujoco_dex_bench.eval.plot_results import plot_param_sensitivity

plot_param_sensitivity("fitted_params.json", "param_sensitivity.png")
```

## Tips

- Collect diverse trajectories (slow/fast, different grasp configurations).
- Match sim control frequency to real robot logging rate.
- Start with `DexBallRoll-v0` — smooth dynamics are easiest to fit.
- Use `n_trials=200+` for production; `n_trials=10` for CI smoke tests.
