"""Scissors cutting dexterous manipulation task."""

from __future__ import annotations

import numpy as np

from mujoco_dex_bench.envs.base_dex_env import BaseDexEnv, DexEnvConfig
from mujoco_dex_bench.envs.utils import geom_pair_in_contact


class ScissorsCutEnv(BaseDexEnv):
    """Cut a soft paper sheet using scissor blades."""

    def __init__(self, **kwargs) -> None:
        """Create the scissors-cut environment."""
        config = DexEnvConfig(
            xml_file="scissors_cut_scene.xml",
            object_body_name="scissors",
        )
        super().__init__(config=config, **kwargs)

    def _sample_goal(self) -> np.ndarray:
        return self._get_object_pose()

    def _position_tolerance(self) -> float:
        return np.inf

    def _orientation_tolerance(self) -> float:
        return np.inf

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        _ = achieved_goal, desired_goal
        blade_a = geom_pair_in_contact(self.model, self.data, "blade_a_geom", "paper_geom")
        blade_b = geom_pair_in_contact(self.model, self.data, "blade_b_geom", "paper_geom")
        return bool(blade_a and blade_b)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float | np.ndarray:
        _ = achieved_goal, desired_goal, info
        reward = float(self._contact_reward())
        if np.ndim(achieved_goal) > 1:
            return np.full(achieved_goal.shape[0], reward, dtype=np.float32)
        return reward

    def _contact_reward(self) -> float:
        if self.is_success(np.zeros(7), np.zeros(7)):
            return self.config.contact_reward_scale
        return 0.0
