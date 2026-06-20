"""SAC baseline training with HER for goal-conditioned dexterous tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import gymnasium as gym
import mujoco_dex_bench  # noqa: F401 — registers environments
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from mujoco_dex_bench.wrappers.domain_randomisation import (
    DomainRandomisationConfig,
    DomainRandomisationWrapper,
)

app = typer.Typer(add_completion=False)
console = Console()
logger = logging.getLogger(__name__)

OBJECT_GEOM_MAP: dict[str, tuple[str, ...]] = {
    "DexPenSpin-v0": ("pen_geom",),
    "DexKeyInsert-v0": ("key_geom",),
    "DexInHandReorient-v0": ("cube_geom",),
    "DexChopstickGrasp-v0": ("stick_a_geom", "stick_b_geom"),
    "DexBallRoll-v0": ("ball_geom",),
    "DexPegInHole-v0": ("peg_geom",),
    "DexDoorOpen-v0": ("door_panel",),
    "DexScissorsCut-v0": ("blade_a_geom", "blade_b_geom"),
    "DexCardFlip-v0": ("card_geom",),
}


def _make_env(env_id: str, domain_rand: bool, seed: int, rank: int) -> Callable[[], gym.Env]:
    """Build environment factory for vectorised training."""

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
    log_dir: str = typer.Option("./logs", help="TensorBoard log directory"),
    save_path: str = typer.Option("./checkpoints", help="Checkpoint directory"),
    domain_rand: bool = typer.Option(False, help="Enable domain randomisation wrapper"),
    eval_freq: int = typer.Option(10_000, help="Evaluation frequency in timesteps"),
) -> None:
    """Train a SAC agent with HER on a dexterous manipulation task."""
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
        name_prefix="sac",
    )

    learning_starts = min(max(1000, 201), max(total_timesteps // 2, 201))

    model = SAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            handle_timeout_termination=True,
        ),
        verbose=1,
        tensorboard_log=str(log_path),
        seed=seed,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        learning_starts=learning_starts,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    final_path = ckpt_path / "final_sac"
    model.save(str(final_path))
    vec_env.save(str(ckpt_path / "vecnormalize.pkl"))

    mean_reward, success_rate = _evaluate_policy(model, eval_env, n_episodes=20)

    table = Table(title="SAC Training Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Environment", env_id)
    table.add_row("Timesteps", str(total_timesteps))
    table.add_row("Mean Eval Reward", f"{mean_reward:.3f}")
    table.add_row("Success Rate", f"{success_rate:.1%}")
    console.print(table)


def _evaluate_policy(model: SAC, env: VecNormalize, n_episodes: int) -> tuple[float, float]:
    """Run a short evaluation rollout."""
    rewards: list[float] = []
    successes: list[float] = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            ep_reward += float(reward[0])
            success = bool(infos[0].get("is_success", False))
            done = bool(dones[0])
        rewards.append(ep_reward)
        successes.append(float(success))
    return float(np.mean(rewards)), float(np.mean(successes))


if __name__ == "__main__":
    app()
