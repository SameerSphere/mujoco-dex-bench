"""Environment smoke and interface tests."""

from __future__ import annotations

import numpy as np
import pytest

import mujoco_dex_bench  # noqa: F401 — registers envs
import gymnasium as gym

ENV_IDS = [
    "DexPenSpin-v0",
    "DexKeyInsert-v0",
    "DexInHandReorient-v0",
    "DexChopstickGrasp-v0",
    "DexBallRoll-v0",
    "DexPegInHole-v0",
    "DexDoorOpen-v0",
    "DexScissorsCut-v0",
    "DexCardFlip-v0",
]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_reset_and_step(env_id: str) -> None:
    """Each env resets, steps, and returns valid observations."""
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert obs["observation"].dtype == np.float32
    assert obs["achieved_goal"].shape == (7,)
    assert obs["desired_goal"].shape == (7,)
    assert isinstance(info, dict)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(info, dict)
        assert "is_success" in info
        assert obs["observation"].shape == env.observation_space["observation"].shape
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_compute_reward_batched(env_id: str) -> None:
    """compute_reward supports batched goal arrays."""
    env = gym.make(env_id)
    env.reset(seed=0)
    achieved = np.tile(env.unwrapped._get_object_pose(), (3, 1)).astype(np.float32)
    desired = achieved.copy()
    reward = env.unwrapped.compute_reward(achieved, desired, {})
    assert isinstance(reward, (float, np.floating)) or (
        isinstance(reward, np.ndarray) and reward.shape == (3,)
    )
    env.close()
