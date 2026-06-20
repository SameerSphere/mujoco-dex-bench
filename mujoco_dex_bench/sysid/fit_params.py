"""System identification via trajectory matching with differential evolution."""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import mujoco_dex_bench  # noqa: F401
import numpy as np
import pandas as pd
import typer
from scipy.optimize import differential_evolution

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

OBJECT_GEOM_MAP: dict[str, str] = {
    "DexPenSpin-v0": "pen_geom",
    "DexKeyInsert-v0": "key_geom",
    "DexInHandReorient-v0": "cube_geom",
    "DexChopstickGrasp-v0": "stick_b_geom",
    "DexBallRoll-v0": "ball_geom",
    "DexPegInHole-v0": "peg_geom",
    "DexDoorOpen-v0": "door_panel",
    "DexScissorsCut-v0": "blade_a_geom",
    "DexCardFlip-v0": "card_geom",
}


def generate_synthetic_dataset(
    env_id: str,
    output_csv: str | Path,
    n_steps: int = 200,
    seed: int = 0,
) -> Path:
    """Roll out a random policy and save trajectory CSV for sysid testing.

    Args:
        env_id: Registered Gymnasium environment ID.
        output_csv: Destination CSV path.
        n_steps: Number of simulation steps.
        seed: Random seed.

    Returns:
        Path to the written CSV file.
    """
    output_path = Path(output_csv)
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    base_env = env.unwrapped

    rows: list[dict[str, float]] = []
    qpos = base_env._get_joint_positions()
    qvel = base_env._get_joint_velocities()
    obj_pose = base_env._get_object_pose()

    for t in range(n_steps):
        row: dict[str, float] = {"t": float(t)}
        for i, val in enumerate(qpos):
            row[f"q{i}"] = float(val)
        for i, val in enumerate(qvel):
            row[f"dq{i}"] = float(val)
        row.update(
            {
                "obj_x": float(obj_pose[0]),
                "obj_y": float(obj_pose[1]),
                "obj_z": float(obj_pose[2]),
                "obj_qw": float(obj_pose[3]),
                "obj_qx": float(obj_pose[4]),
                "obj_qy": float(obj_pose[5]),
                "obj_qz": float(obj_pose[6]),
            }
        )
        rows.append(row)

        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        _ = obs
        qpos = base_env._get_joint_positions()
        qvel = base_env._get_joint_velocities()
        obj_pose = base_env._get_object_pose()
        if terminated or truncated:
            obs, _ = env.reset()

    pd.DataFrame(rows).to_csv(output_path, index=False)
    env.close()
    logger.info("Saved synthetic dataset to %s", output_path)
    return output_path


def _load_trajectory(csv_path: str | Path) -> pd.DataFrame:
    """Load trajectory CSV with required columns."""
    df = pd.read_csv(csv_path)
    required = {"t", "obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def _pose_mse(sim_poses: np.ndarray, real_poses: np.ndarray) -> float:
    """Compute mean squared error between pose trajectories."""
    pos_err = np.mean((sim_poses[:, :3] - real_poses[:, :3]) ** 2)
    quat_err = np.mean((sim_poses[:, 3:7] - real_poses[:, 3:7]) ** 2)
    return float(pos_err + quat_err)


def fit_params(
    real_trajectory_csv: str,
    env_id: str,
    n_trials: int = 200,
    output_xml: str = "fitted_params.xml",
) -> dict[str, Any]:
    """Fit simulation parameters to a real robot trajectory.

    Uses differential evolution over mass, friction, and damping scales to
    minimise object-pose MSE between simulation replay and recorded data.

    Args:
        real_trajectory_csv: Path to real trajectory CSV.
        env_id: Gymnasium environment ID.
        n_trials: Maximum DE iterations (passed as ``maxiter``).
        output_xml: Output XML path for fitted parameters.

    Returns:
        Dictionary with best parameters and achieved MSE.
    """
    df = _load_trajectory(real_trajectory_csv)
    real_poses = df[
        ["obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz"]
    ].to_numpy(dtype=np.float64)

    env = gym.make(env_id)
    base_env = env.unwrapped
    model = base_env.model
    data = base_env.data
    geom_name = OBJECT_GEOM_MAP.get(env_id, "ball_geom")
    geom_id = model.geom(geom_name).id
    body_id = model.geom_bodyid[geom_id]
    base_mass = float(model.body_mass[body_id])
    base_friction = model.geom_friction[geom_id].copy()
    base_damping = model.dof_damping.copy()

    n_steps = min(len(df), 200)

    def objective(x: np.ndarray) -> float:
        """Evaluate parameter candidate."""
        mass_scale, friction_val, damp_scale = x
        model.body_mass[body_id] = base_mass * mass_scale
        model.geom_friction[geom_id] = np.array(
            [friction_val, base_friction[1], base_friction[2]], dtype=np.float64
        )
        model.dof_damping[:] = base_damping * damp_scale

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        sim_poses = []
        for t in range(n_steps):
            pose = base_env._get_object_pose()
            sim_poses.append(pose)
            action = env.action_space.sample()
            ctrl = base_env._scale_action(action)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

        return _pose_mse(np.array(sim_poses), real_poses[:n_steps])

    bounds = [(0.5, 2.0), (0.3, 3.0), (0.5, 2.0)]
    result = differential_evolution(
        objective,
        bounds,
        maxiter=max(1, n_trials // 10),
        seed=42,
        polish=True,
        workers=1,
    )

    best_mass_scale, best_friction, best_damp_scale = result.x
    best_mse = float(result.fun)

    best_params = {
        "mass_scale": float(best_mass_scale),
        "mass_kg": float(base_mass * best_mass_scale),
        "friction": float(best_friction),
        "damping_scale": float(best_damp_scale),
        "mse": best_mse,
        "geom_name": geom_name,
        "env_id": env_id,
    }

    _write_fitted_xml(
        base_env.xml_path,
        Path(output_xml),
        geom_name,
        best_params,
    )

    log_path = Path(output_xml).with_suffix(".json")
    log_path.write_text(json.dumps(best_params, indent=2))
    env.close()
    logger.info("SysID complete: MSE=%.6f, params=%s", best_mse, best_params)
    return best_params


def _write_fitted_xml(
    source_xml: Path,
    output_xml: Path,
    geom_name: str,
    params: dict[str, Any],
) -> None:
    """Write fitted parameters into a copy of the scene XML."""
    tree = ET.parse(source_xml)
    root = tree.getroot()

    for geom in root.iter("geom"):
        if geom.get("name") == geom_name:
            geom.set("mass", f"{params['mass_kg']:.6f}")
            geom.set("friction", f"{params['friction']:.4f} 0.005 0.0001")

    for joint in root.iter("joint"):
        if joint.get("damping"):
            base = float(joint.get("damping", "0.05"))
            joint.set("damping", f"{base * params['damping_scale']:.6f}")

    tree.write(output_xml, encoding="unicode", xml_declaration=True)


@app.command()
def main(
    real_trajectory_csv: str = typer.Option(..., help="Real robot trajectory CSV"),
    env_id: str = typer.Option("DexBallRoll-v0", help="Environment ID"),
    n_trials: int = typer.Option(200, help="Optimisation trials"),
    output_xml: str = typer.Option("fitted_params.xml", help="Output XML path"),
    generate_synthetic: bool = typer.Option(
        False, help="Generate synthetic CSV before fitting (for testing)"
    ),
) -> None:
    """CLI entry point for system identification."""
    logging.basicConfig(level=logging.INFO)
    csv_path = Path(real_trajectory_csv)
    if generate_synthetic or not csv_path.exists():
        generate_synthetic_dataset(env_id, csv_path, n_steps=100, seed=0)
    result = fit_params(str(csv_path), env_id, n_trials=n_trials, output_xml=output_xml)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
