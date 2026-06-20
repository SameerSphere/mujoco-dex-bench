"""Chopstick grasping dexterous manipulation task."""

from __future__ import annotations

import numpy as np

from mujoco_dex_bench.envs.base_dex_env import BaseDexEnv, DexEnvConfig
from mujoco_dex_bench.envs.utils import normalize_quaternion


class ChopstickGraspEnv(BaseDexEnv):
    """Grasp and lift an object using two chopsticks."""

    def __init__(self, **kwargs) -> None:
        """Create the chopstick-grasp environment."""
        config = DexEnvConfig(
            xml_file="chopstick_grasp_scene.xml",
            object_body_name="chopstick_b",
            pos_weight=1.5,
        )
        super().__init__(config=config, **kwargs)

    def _sample_goal(self) -> np.ndarray:
        pos = np.array([0.06, 0.05, 0.25], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.concatenate([pos, normalize_quaternion(quat)])

    def _position_tolerance(self) -> float:
        return 0.010

    def _orientation_tolerance(self) -> float:
        return np.deg2rad(20.0)
