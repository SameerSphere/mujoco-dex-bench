"""Domain randomisation wrapper tests."""

from __future__ import annotations

import gymnasium as gym
import mujoco_dex_bench  # noqa: F401
import pytest

from mujoco_dex_bench.wrappers.domain_randomisation import (
    DomainRandomisationConfig,
    DomainRandomisationWrapper,
)


def test_domain_randomisation_changes_between_episodes() -> None:
    """Randomisation log differs across consecutive resets."""
    env = gym.make("DexBallRoll-v0")
    config = DomainRandomisationConfig(object_geom_names=("ball_geom",))
    wrapped = DomainRandomisationWrapper(env, config=config)

    wrapped.reset(seed=0)
    log_a = wrapped.get_randomisation_log()

    wrapped.reset(seed=1)
    log_b = wrapped.get_randomisation_log()

    assert log_a != log_b
    wrapped.close()


def test_domain_randomisation_preserves_obs_shape() -> None:
    """Observation shape is unchanged after wrapping."""
    env = gym.make("DexBallRoll-v0")
    base_shape = env.observation_space["observation"].shape
    wrapped = DomainRandomisationWrapper(
        env, config=DomainRandomisationConfig(object_geom_names=("ball_geom",))
    )
    obs, _ = wrapped.reset(seed=0)
    assert obs["observation"].shape == base_shape
    wrapped.close()
