"""Peg-in-hole dexterous manipulation task."""

from __future__ import annotations

import numpy as np

from mujoco_dex_bench.envs.base_dex_env import BaseDexEnv, DexEnvConfig
from mujoco_dex_bench.envs.utils import geom_pair_in_contact, normalize_quaternion


class PegInHoleEnv(BaseDexEnv):
    """Insert a cylindrical peg into a tight-clearance hole."""

    def __init__(self, **kwargs) -> None:
        """Create the peg-in-hole environment."""
        config = DexEnvConfig(
            xml_file="peg_in_hole_scene.xml",
            object_body_name="peg",
            pos_weight=2.5,
            rot_weight=1.5,
        )
        super().__init__(config=config, **kwargs)

    def _sample_goal(self) -> np.ndarray:
        pos = np.array([0.12, 0.0, 0.10], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.concatenate([pos, normalize_quaternion(quat)])

    def _position_tolerance(self) -> float:
        return 0.0015

    def _orientation_tolerance(self) -> float:
        return np.deg2rad(3.0)

    def _contact_reward(self) -> float:
        if geom_pair_in_contact(self.model, self.data, "peg_geom", "hole_plate_geom"):
            return self.config.contact_reward_scale
        return 0.0
