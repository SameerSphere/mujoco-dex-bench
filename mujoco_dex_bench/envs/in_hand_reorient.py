"""In-hand cube reorientation task."""

from __future__ import annotations

import numpy as np

from mujoco_dex_bench.envs.base_dex_env import BaseDexEnv, DexEnvConfig
from mujoco_dex_bench.envs.utils import normalize_quaternion, random_quaternion


class InHandReorientEnv(BaseDexEnv):
    """Reorient a cube held in the palm without dropping it."""

    def __init__(self, **kwargs) -> None:
        """Create the in-hand reorientation environment."""
        config = DexEnvConfig(
            xml_file="in_hand_reorient_scene.xml",
            object_body_name="cube",
        )
        super().__init__(config=config, **kwargs)

    def _sample_goal(self) -> np.ndarray:
        pos = self._get_object_pose()[:3]
        quat = random_quaternion(self.np_random)
        return np.concatenate([pos, quat])

    def _position_tolerance(self) -> float:
        return 0.005

    def _orientation_tolerance(self) -> float:
        return np.deg2rad(10.0)
