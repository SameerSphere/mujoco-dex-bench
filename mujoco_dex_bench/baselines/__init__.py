"""Stable-Baselines3 baseline training scripts."""

from mujoco_dex_bench.baselines.train_ppo import app as ppo_app
from mujoco_dex_bench.baselines.train_sac import app as sac_app

__all__ = ["sac_app", "ppo_app"]
