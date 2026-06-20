"""Ball rolling dexterous manipulation task."""

from __future__ import annotations

import numpy as np

from mujoco_dex_bench.envs.base_dex_env import BaseDexEnv, DexEnvConfig
from mujoco_dex_bench.envs.utils import normalize_quaternion


class BallRollEnv(BaseDexEnv):
    """Roll a sphere to a target location on the table."""

    def __init__(self, **kwargs) -> None:
        """Create the ball-roll environment."""
        config = DexEnvConfig(
            xml_file="ball_roll_scene.xml",
            object_body_name="ball",
            rot_weight=0.0,
        )
        super().__init__(config=config, **kwargs)

    def _sample_goal(self) -> np.ndarray:
        pos = np.array([0.15, 0.05, 0.025], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.concatenate([pos, normalize_quaternion(quat)])

    def _position_tolerance(self) -> float:
        return 0.008

    def _orientation_tolerance(self) -> float:
        return np.inf

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        achieved = np.atleast_2d(achieved_goal)[0]
        desired = np.atleast_2d(desired_goal)[0]
        return bool(np.linalg.norm(achieved[:3] - desired[:3]) <= self._position_tolerance())
