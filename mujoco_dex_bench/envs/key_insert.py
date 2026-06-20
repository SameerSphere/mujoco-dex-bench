"""Key insertion dexterous manipulation task."""

from __future__ import annotations

import numpy as np

from mujoco_dex_bench.envs.base_dex_env import BaseDexEnv, DexEnvConfig
from mujoco_dex_bench.envs.utils import geom_pair_in_contact, normalize_quaternion


class KeyInsertEnv(BaseDexEnv):
    """Insert a key into a lock keyhole."""

    def __init__(self, **kwargs) -> None:
        """Create the key-insert environment."""
        config = DexEnvConfig(
            xml_file="key_insert_scene.xml",
            object_body_name="key",
            pos_weight=2.0,
            rot_weight=1.0,
        )
        super().__init__(config=config, **kwargs)

    def _sample_goal(self) -> np.ndarray:
        target_pos = np.array([0.15, 0.0, 0.12], dtype=np.float64)
        target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.concatenate([target_pos, normalize_quaternion(target_quat)])

    def _position_tolerance(self) -> float:
        return 0.002

    def _orientation_tolerance(self) -> float:
        return np.deg2rad(5.0)

    def _contact_reward(self) -> float:
        if geom_pair_in_contact(self.model, self.data, "key_geom", "lock_body"):
            return self.config.contact_reward_scale
        return 0.0
