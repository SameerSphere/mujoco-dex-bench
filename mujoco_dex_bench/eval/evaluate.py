"""Evaluate trained SAC/PPO policies on dexterous manipulation tasks."""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco_dex_bench  # noqa: F401
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_dex_bench.baselines.train_sac import OBJECT_GEOM_MAP
from mujoco_dex_bench.wrappers.domain_randomisation import (
    DomainRandomisationConfig,
    DomainRandomisationWrapper,
)

app = typer.Typer(add_completion=False)
console = Console()
logger = logging.getLogger(__name__)


def _make_eval_env(env_id: str, domain_rand: bool) -> gym.Env:
    """Create a single evaluation environment."""
    env = gym.make(env_id)
    if domain_rand:
        geoms = OBJECT_GEOM_MAP.get(env_id, ("pen_geom",))
        env = DomainRandomisationWrapper(
            env, config=DomainRandomisationConfig(object_geom_names=geoms)
        )
    return env


def _load_model(model_path: str, env: gym.Env):
    """Load SAC or PPO model from path."""
    path = Path(model_path)
    sac_path = path if path.suffix == ".zip" else Path(str(path) + ".zip")
    try:
        return SAC.load(str(sac_path), env=env)
    except Exception:
        return PPO.load(str(sac_path), env=env)


def _mean_contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Compute mean contact force magnitude across all contacts."""
    if data.ncon == 0:
        return 0.0
    forces = []
    force_array = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, force_array)
        magnitude = float(np.linalg.norm(force_array[:3]))
        if np.isfinite(magnitude):
            forces.append(min(magnitude, 1e4))
    return float(np.mean(forces)) if forces else 0.0


@app.command()
def evaluate(
    env_id: str = typer.Option(..., help="Environment ID"),
    model_path: str = typer.Option(..., help="Path to saved model"),
    n_episodes: int = typer.Option(100, help="Number of evaluation episodes"),
    render: bool = typer.Option(False, help="Open MuJoCo viewer"),
    results_csv: str = typer.Option("eval_results.csv", help="Output CSV path"),
    domain_rand: bool = typer.Option(False, help="Enable domain randomisation"),
) -> None:
    """Evaluate a trained policy and save metrics to CSV."""
    logging.basicConfig(level=logging.INFO)
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    if domain_rand:
        geoms = OBJECT_GEOM_MAP.get(env_id, ("pen_geom",))
        env = DomainRandomisationWrapper(
            env, config=DomainRandomisationConfig(object_geom_names=geoms)
        )

    vec_path = Path(model_path).parent / "vecnormalize.pkl"
    wrapped = DummyVecEnv([lambda: env])
    if vec_path.exists():
        wrapped = VecNormalize.load(str(vec_path), wrapped)
        wrapped.training = False
        wrapped.norm_reward = False

    model = _load_model(model_path, wrapped)

    records: list[dict[str, float | bool | int]] = []
    for ep in range(n_episodes):
        obs = wrapped.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        contact_forces: list[float] = []
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = wrapped.step(action)
            ep_reward += float(reward[0])
            ep_len += 1
            success = bool(infos[0].get("is_success", False))
            done = bool(dones[0])

            base = env.unwrapped if hasattr(env, "unwrapped") else env
            while hasattr(base, "env"):
                if hasattr(base, "model"):
                    break
                base = base.env
            if hasattr(base, "model"):
                contact_forces.append(_mean_contact_force(base.model, base.data))

            if render:
                env.render()

        records.append(
            {
                "episode": ep,
                "reward": ep_reward,
                "success": success,
                "length": ep_len,
                "mean_contact_force": float(np.mean(contact_forces)) if contact_forces else 0.0,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(results_csv, index=False)

    table = Table(title=f"Evaluation: {env_id}")
    table.add_column("Metric")
    table.add_column("Mean ± Std")
    for col in ["reward", "length", "mean_contact_force"]:
        table.add_row(col, f"{df[col].mean():.3f} ± {df[col].std():.3f}")
    table.add_row("success_rate", f"{df['success'].mean():.1%}")
    console.print(table)

    env.close()


if __name__ == "__main__":
    app()
