"""Pen spinning dexterous manipulation task."""

from __future__ import annotations

import numpy as np

from mujoco_dex_bench.envs.base_dex_env import BaseDexEnv, DexEnvConfig
from mujoco_dex_bench.envs.utils import normalize_quaternion, random_quaternion, rotation_about_axis_angle


class PenSpinEnv(BaseDexEnv):
    """Spin a pen about its long axis while maintaining grasp."""

    def __init__(self, **kwargs) -> None:
        """Create the pen-spin environment."""
        config = DexEnvConfig(
            xml_file="pen_spin_scene.xml",
            object_body_name="pen",
            rot_weight=1.5,
        )
        super().__init__(config=config, **kwargs)
        self._initial_spin_angle = 0.0

    def _sample_goal(self) -> np.ndarray:
        """Sample target orientation: 180 deg spin about pen long axis."""
        pos, quat = self._get_object_pose()[:3], self._get_object_pose()[3:7]
        axis = np.array([0.0, 0.0, 1.0])
        delta = random_quaternion(self.np_random)
        goal_quat = normalize_quaternion(
            np.array(
                [
                    np.cos(np.pi / 2),
                    axis[0] * np.sin(np.pi / 2),
                    axis[1] * np.sin(np.pi / 2),
                    axis[2] * np.sin(np.pi / 2),
                ]
            )
        )
        _ = delta  # keep RNG consumption stable across tasks
        return np.concatenate([pos, goal_quat])

    def _set_initial_state(self) -> None:
        super()._set_initial_state()
        self._initial_spin_angle = rotation_about_axis_angle(
            self._get_object_pose()[3:7], np.array([0.0, 0.0, 1.0])
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Reset and record initial spin angle after simulation forward."""
        obs, info = super().reset(seed=seed, options=options)
        self._initial_spin_angle = rotation_about_axis_angle(
            self._get_object_pose()[3:7], np.array([0.0, 0.0, 1.0])
        )
        return obs, info

    def _position_tolerance(self) -> float:
        return np.inf

    def _orientation_tolerance(self) -> float:
        return np.deg2rad(15.0)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Success when pen has rotated >= 15 deg about long axis from start."""
        achieved = np.atleast_2d(achieved_goal)[0]
        angle = abs(
            rotation_about_axis_angle(achieved[3:7], np.array([0.0, 0.0, 1.0]))
            - self._initial_spin_angle
        )
        return bool(angle >= self._orientation_tolerance())

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float | np.ndarray:
        achieved = np.atleast_2d(achieved_goal)
        rewards = []
        for ach in achieved:
            angle = abs(
                rotation_about_axis_angle(ach[3:7], np.array([0.0, 0.0, 1.0]))
                - self._initial_spin_angle
            )
            target = np.pi / 2
            rewards.append(-abs(angle - target))
        if achieved_goal.ndim == 1:
            return float(rewards[0])
        return np.asarray(rewards, dtype=np.float32)
