# MuJoCo Dexterous Hand Benchmark

> 9-task dexterous manipulation suite · MuJoCo 3.x · Gymnasium-Robotics · SAC/PPO baselines

![banner](assets/banner.svg)

## Overview

**mujoco-dex-bench** is an open-source benchmark suite of nine dexterous robotic manipulation tasks built on MuJoCo 3.x, Gymnasium, and Stable-Baselines3. Each task features a physically plausible five-finger anthropomorphic hand modelled with primitive geoms, dense potential-based rewards, and standardised goal-conditioned observations compatible with Hindsight Experience Replay (HER).

Unlike general-purpose robot learning benchmarks that focus on arm reach-and-grasp, this suite targets fine in-hand manipulation: pen spinning, key insertion, peg-in-hole assembly, door opening, and other contact-rich skills that stress tactile sensing, coordinated finger actuation, and precise orientation control. All environments share a common base class, domain-randomisation wrapper, system-identification pipeline, and evaluation tooling so researchers can compare algorithms fairly across tasks.

The repository ships fully runnable SAC and PPO baselines, automatic parameter fitting from real robot trajectories, and CI-tested pytest coverage — clone, install, and train in under ten minutes.

## Task Suite

| Task ID | Description | Object | Success Criterion | Difficulty |
|---------|-------------|--------|-------------------|------------|
| `DexPenSpin-v0` | Rotate pen about long axis | Capsule | ≥15° spin | Hard |
| `DexKeyInsert-v0` | Insert key into lock | Box | 2 mm, 5° | Hard |
| `DexInHandReorient-v0` | Reorient cube in palm | Cube | 5 mm, 10° | Medium |
| `DexChopstickGrasp-v0` | Grasp with chopsticks | 2 capsules | 10 mm, 20° | Hard |
| `DexBallRoll-v0` | Roll ball to target | Sphere | 8 mm | Easy |
| `DexPegInHole-v0` | Peg insertion | Cylinder | 1.5 mm, 3° | Hard |
| `DexDoorOpen-v0` | Pull door open | Hinged panel | >60° angle | Medium |
| `DexScissorsCut-v0` | Cut paper with scissors | Soft sheet | Blade contact | Medium |
| `DexCardFlip-v0` | Flip playing card 180° | Thin box | 5 mm, 180° | Hard |

## Installation

```bash
git clone https://github.com/SameerSphere/mujoco-dex-bench.git
cd mujoco-dex-bench
pip install -e ".[dev]"
```

Conda environment (optional):

```bash
conda create -n dexbench python=3.11 -y
conda activate dexbench
pip install -e ".[dev]"
```

MuJoCo 3.x is bundled with the `mujoco` Python package — no separate license key is required for MuJoCo 3+. For headless servers, set `export MUJOCO_GL=osmesa`.

## Quick Start

**Make an environment:**

```python
import gymnasium as gym
import mujoco_dex_bench  # registers envs

env = gym.make("DexBallRoll-v0")
obs, info = env.reset()
print(obs["observation"].shape)
```

**Train SAC:**

```bash
dex-train --env-id DexBallRoll-v0 --total-timesteps 1000000 --n-envs 4
```

**Evaluate:**

```bash
dex-eval --env-id DexBallRoll-v0 --model-path checkpoints/best_model --n-episodes 100
```

**Run system identification:**

```bash
dex-sysid --real-trajectory-csv synthetic.csv --env-id DexBallRoll-v0 --n-trials 200
```

## Environment Details

### Observation space (Dict)

| Key | Shape | Description |
|-----|-------|-------------|
| `observation` | `(obs_dim,)` | sin/cos joint pos, joint vel, touch (5), object pose, object vel, palm→object |
| `achieved_goal` | `(7,)` | Current object pose `[x,y,z,qw,qx,qy,qz]` |
| `desired_goal` | `(7,)` | Target object pose |

### Action space

Continuous `Box(15,)` in `[-1, 1]` — position-controlled finger joints (15 actuators).

### Reward structure

```
r = r_dense + r_bonus − 0.01·‖a‖² + r_contact
```

- `r_dense`: negative weighted pose distance (position + orientation)
- `r_bonus`: +10.0 on task success
- `r_contact`: +0.5 for relevant geom contact (peg, key tasks)

## Domain Randomisation

The `DomainRandomisationWrapper` randomises at each episode reset:

- Object mass ±20%
- Friction ±35%
- Actuator gear ±10%
- Joint damping ±25%
- Observation noise (configurable σ)
- Action latency 0–2 steps
- Gravity tilt ±0.02

```python
from mujoco_dex_bench.wrappers import DomainRandomisationWrapper, DomainRandomisationConfig

env = gym.make("DexBallRoll-v0")
env = DomainRandomisationWrapper(env, config=DomainRandomisationConfig())
obs, info = env.reset()
print(env.get_randomisation_log())
```

## System Identification

Fit simulation parameters (mass, friction, damping) to real robot trajectories using differential evolution. See [docs/sysid.md](docs/sysid.md) for the full tutorial.

CSV format: `t, q0..qN, dq0..dqN, obj_x, obj_y, obj_z, obj_qw, obj_qx, obj_qy, obj_qz`

```python
from mujoco_dex_bench.sysid import fit_params, generate_synthetic_dataset

generate_synthetic_dataset("DexBallRoll-v0", "synthetic.csv")
result = fit_params("synthetic.csv", "DexBallRoll-v0", n_trials=200)
print(result["mse"], result["mass_scale"])
```

## Baselines

| Task | SAC mean reward | PPO mean reward | SAC success | PPO success |
|------|-----------------|-----------------|-------------|-------------|
| DexBallRoll-v0 | — | — | — | — |
| DexPenSpin-v0 | — | — | — | — |
| *(placeholder — run training to populate)* | | | | |

Train PPO: `python -m mujoco_dex_bench.baselines.train_ppo train --env-id DexBallRoll-v0`

## Repo Structure

```
mujoco_dex_bench/
├── envs/           # Base env + 9 task implementations
├── mjcf/           # Hand model + scene XML files
├── wrappers/       # Domain randomisation, obs normalisation
├── baselines/      # SAC and PPO training scripts
├── sysid/          # Parameter fitting from real trajectories
└── eval/           # Evaluation and plotting utilities
```

## Citing

```bibtex
@software{mujoco_dex_bench2025,
  author  = {Sameer Chaulagain},
  title   = {MuJoCo Dexterous Hand Benchmark},
  year    = {2025},
  url     = {https://github.com/SameerSphere/mujoco-dex-bench}
}
```

## License

MIT
