"""Running observation normalisation wrapper."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ObservationNormaliserWrapper(gym.Wrapper):
    """Online normalisation of the ``observation`` vector using running statistics."""

    def __init__(self, env: gym.Env, epsilon: float = 1e-8) -> None:
        """Create the normaliser wrapper.

        Args:
            env: Dict-observation environment.
            epsilon: Small constant for numerical stability.
        """
        super().__init__(env)
        obs_space = env.observation_space
        dim = obs_space["observation"].shape[0]
        self.epsilon = epsilon
        self._count = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._var = np.ones(dim, dtype=np.float64)

    def _update(self, obs: np.ndarray) -> None:
        """Update running mean and variance."""
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._var += delta * delta2

    def _normalise(self, obs: np.ndarray) -> np.ndarray:
        """Normalise observation vector."""
        if self._count < 2:
            return obs.astype(np.float32)
        std = np.sqrt(self._var / (self._count - 1) + self.epsilon)
        return ((obs - self._mean) / std).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset and normalise initial observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        vec = obs["observation"].astype(np.float64)
        self._update(vec)
        obs = dict(obs)
        obs["observation"] = self._normalise(vec)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Step and normalise observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        vec = obs["observation"].astype(np.float64)
        self._update(vec)
        obs = dict(obs)
        obs["observation"] = self._normalise(vec)
        return obs, reward, terminated, truncated, info

    def get_stats(self) -> dict[str, np.ndarray]:
        """Return running mean and std for serialisation."""
        std = np.sqrt(self._var / max(self._count - 1, 1) + self.epsilon)
        return {"mean": self._mean.copy(), "std": std.copy(), "count": np.array([self._count])}
