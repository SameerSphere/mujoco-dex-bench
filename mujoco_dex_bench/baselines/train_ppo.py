"""PPO baseline training for goal-conditioned dexterous tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import gymnasium as gym
import mujoco_dex_bench  # noqa: F401
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from mujoco_dex_bench.baselines.train_sac import OBJECT_GEOM_MAP, _evaluate_policy
from mujoco_dex_bench.wrappers.domain_randomisation import (
    DomainRandomisationConfig,
    DomainRandomisationWrapper,
)

app = typer.Typer(add_completion=False)
console = Console()
logger = logging.getLogger(__name__)


def _make_env(env_id: str, domain_rand: bool, seed: int, rank: int) -> Callable[[], gym.Env]:
    """Build environment factory for vectorised PPO training."""

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        if domain_rand:
            geoms = OBJECT_GEOM_MAP.get(env_id, ("pen_geom",))
            config = DomainRandomisationConfig(object_geom_names=geoms)
            env = DomainRandomisationWrapper(env, config=config)
        return Monitor(env)

    return _init


@app.command()
def train(
    env_id: str = typer.Option("DexPenSpin-v0", help="Environment ID"),
    total_timesteps: int = typer.Option(1_000_000, help="Total training timesteps"),
    n_envs: int = typer.Option(4, help="Number of parallel environments"),
    seed: int = typer.Option(42, help="Random seed"),
    log_dir: str = typer.Option("./logs_ppo", help="TensorBoard log directory"),
    save_path: str = typer.Option("./checkpoints_ppo", help="Checkpoint directory"),
    domain_rand: bool = typer.Option(False, help="Enable domain randomisation wrapper"),
    eval_freq: int = typer.Option(10_000, help="Evaluation frequency in timesteps"),
) -> None:
    """Train a PPO agent with MultiInputPolicy on a dexterous manipulation task."""
    logging.basicConfig(level=logging.INFO)
    log_path = Path(log_dir)
    ckpt_path = Path(save_path)
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(
        _make_env(env_id, domain_rand, seed, 0),
        n_envs=n_envs,
        seed=seed,
    )
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, norm_obs_keys=["observation"]
    )

    eval_env = make_vec_env(_make_env(env_id, domain_rand, seed + 1000, 0), n_envs=1)
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        norm_obs_keys=["observation"],
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_path),
        log_path=str(log_path / "eval"),
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq // n_envs, 1),
        save_path=str(ckpt_path / "intermediate"),
        name_prefix="ppo",
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(log_path),
        seed=seed,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    final_path = ckpt_path / "final_ppo"
    model.save(str(final_path))
    vec_env.save(str(ckpt_path / "vecnormalize.pkl"))

    mean_reward, success_rate = _evaluate_policy_ppo(model, eval_env, n_episodes=20)

    table = Table(title="PPO Training Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Environment", env_id)
    table.add_row("Timesteps", str(total_timesteps))
    table.add_row("Mean Eval Reward", f"{mean_reward:.3f}")
    table.add_row("Success Rate", f"{success_rate:.1%}")
    console.print(table)


def _evaluate_policy_ppo(model: PPO, env: VecNormalize, n_episodes: int) -> tuple[float, float]:
    """Run a short PPO evaluation rollout."""
    return _evaluate_policy(model, env, n_episodes)


if __name__ == "__main__":
    app()
