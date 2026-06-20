"""Domain randomisation wrapper for dexterous manipulation environments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


@dataclass
class DomainRandomisationConfig:
    """Configurable ranges for physics and observation domain randomisation."""

    mass_scale: tuple[float, float] = (0.8, 1.2)
    friction_scale: tuple[float, float] = (0.65, 1.35)
    gear_scale: tuple[float, float] = (0.9, 1.1)
    damping_scale: tuple[float, float] = (0.75, 1.25)
    obs_noise_sigma: float = 0.01
    joint_vel_noise_sigma: float = 0.05
    touch_noise_sigma: float = 0.02
    max_action_latency: int = 2
    gravity_tilt_range: tuple[float, float] = (-0.02, 0.02)
    object_geom_names: tuple[str, ...] = field(default_factory=lambda: ("pen_geom",))


class DomainRandomisationWrapper(gym.Wrapper):
    """Randomises physical parameters at each episode reset.

    Randomises object mass, friction, actuator gear, joint damping, observation
    noise, action latency, and gravity tilt. Exposes sampled parameters via
    ``get_randomisation_log()`` for system identification.
    """

    def __init__(
        self,
        env: gym.Env,
        config: DomainRandomisationConfig | None = None,
    ) -> None:
        """Wrap an environment with domain randomisation.

        Args:
            env: Base Gymnasium environment with an underlying MuJoCo model.
            config: Randomisation configuration dataclass.
        """
        super().__init__(env)
        self.config = config or DomainRandomisationConfig()
        self._randomisation_log: dict[str, Any] = {}
        self._action_buffer: deque[np.ndarray] = deque()
        self._latency = 0

    def get_randomisation_log(self) -> dict[str, Any]:
        """Return parameters sampled during the most recent reset."""
        return dict(self._randomisation_log)

    def _unwrapped_env(self) -> Any:
        """Return the base dex environment."""
        env = self.env
        while hasattr(env, "env"):
            if hasattr(env, "model") and hasattr(env, "data"):
                return env
            env = env.env
        return env

    def _apply_physics_randomisation(self) -> None:
        """Sample and apply physics parameter perturbations."""
        base_env = self._unwrapped_env()
        model: mujoco.MjModel = base_env.model
        rng = self.np_random
        log: dict[str, Any] = {}

        for geom_name in self.config.object_geom_names:
            try:
                geom_id = model.geom(geom_name).id
            except KeyError:
                continue
            base_mass = float(model.body_mass[model.geom_bodyid[geom_id]])
            mass = float(rng.uniform(base_mass * self.config.mass_scale[0], base_mass * self.config.mass_scale[1]))
            model.body_mass[model.geom_bodyid[geom_id]] = mass
            log[f"mass_{geom_name}"] = mass

            base_friction = model.geom_friction[geom_id].copy()
            scale = rng.uniform(self.config.friction_scale[0], self.config.friction_scale[1])
            model.geom_friction[geom_id] = base_friction * scale
            log[f"friction_{geom_name}"] = model.geom_friction[geom_id].tolist()

        for i in range(model.nu):
            base_gear = float(model.actuator_gear[i, 0])
            model.actuator_gear[i, 0] = float(
                rng.uniform(base_gear * self.config.gear_scale[0], base_gear * self.config.gear_scale[1])
            )
            log[f"gear_actuator_{i}"] = float(model.actuator_gear[i, 0])

        for j in range(model.njnt):
            base_damp = float(model.dof_damping[model.jnt_dofadr[j]])
            if base_damp > 0:
                model.dof_damping[model.jnt_dofadr[j]] = float(
                    rng.uniform(
                        base_damp * self.config.damping_scale[0],
                        base_damp * self.config.damping_scale[1],
                    )
                )
                log[f"damping_joint_{j}"] = float(model.dof_damping[model.jnt_dofadr[j]])

        tilt_x = float(rng.uniform(*self.config.gravity_tilt_range))
        tilt_y = float(rng.uniform(*self.config.gravity_tilt_range))
        model.opt.gravity[:] = np.array([-9.81 * tilt_x, -9.81 * tilt_y, -9.81])
        log["gravity"] = model.opt.gravity.tolist()

        self._latency = int(rng.integers(0, self.config.max_action_latency + 1))
        log["action_latency"] = self._latency
        self._action_buffer.clear()

        self._randomisation_log = log

    def _add_obs_noise(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply Gaussian noise to observation components."""
        noisy = dict(obs)
        vec = obs["observation"].copy()
        n_joints = 15
        vec[: 2 * n_joints] += self.np_random.normal(
            0, self.config.obs_noise_sigma, size=2 * n_joints
        ).astype(np.float32)
        vec[2 * n_joints : 3 * n_joints] += self.np_random.normal(
            0, self.config.joint_vel_noise_sigma, size=n_joints
        ).astype(np.float32)
        vec[3 * n_joints : 3 * n_joints + 5] += self.np_random.normal(
            0, self.config.touch_noise_sigma, size=5
        ).astype(np.float32)
        noisy["observation"] = vec.astype(np.float32)
        return noisy

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment and apply domain randomisation."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._apply_physics_randomisation()
        obs = self._add_obs_noise(obs)
        info["randomisation"] = self.get_randomisation_log()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Step with optional action latency."""
        action = np.asarray(action, dtype=np.float32)
        self._action_buffer.append(action)
        if len(self._action_buffer) <= self._latency:
            delayed = self._action_buffer[0]
        else:
            delayed = self._action_buffer[-1 - self._latency]

        obs, reward, terminated, truncated, info = self.env.step(delayed)
        obs = self._add_obs_noise(obs)
        return obs, reward, terminated, truncated, info
