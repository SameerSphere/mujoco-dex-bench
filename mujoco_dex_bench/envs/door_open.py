"""Door opening dexterous manipulation task."""

from __future__ import annotations

import numpy as np

from mujoco_dex_bench.envs.base_dex_env import BaseDexEnv, DexEnvConfig


class DoorOpenEnv(BaseDexEnv):
    """Pull a hinged door open past 60 degrees."""

    def __init__(self, **kwargs) -> None:
        """Create the door-open environment."""
        config = DexEnvConfig(
            xml_file="door_open_scene.xml",
            object_body_name="door",
            rot_weight=0.0,
        )
        super().__init__(config=config, **kwargs)

    def _sample_goal(self) -> np.ndarray:
        pos, quat = self._get_object_pose()[:3], self._get_object_pose()[3:7]
        return np.concatenate([pos, quat])

    def _door_angle(self) -> float:
        joint_id = self.model.joint("door_hinge").id
        adr = self.model.jnt_qposadr[joint_id]
        return float(np.deg2rad(self.data.qpos[adr]))

    def _position_tolerance(self) -> float:
        return np.inf

    def _orientation_tolerance(self) -> float:
        return np.deg2rad(60.0)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        _ = achieved_goal, desired_goal
        return bool(self._door_angle() >= self._orientation_tolerance())

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float | np.ndarray:
        _ = achieved_goal, desired_goal, info
        angle = self._door_angle()
        target = self._orientation_tolerance()
        reward = -abs(angle - target)
        if np.ndim(achieved_goal) > 1:
            return np.full(achieved_goal.shape[0], reward, dtype=np.float32)
        return float(reward)
