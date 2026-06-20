"""Playing-card flip dexterous manipulation task."""

from __future__ import annotations

import numpy as np

from mujoco_dex_bench.envs.base_dex_env import BaseDexEnv, DexEnvConfig
from mujoco_dex_bench.envs.utils import normalize_quaternion, quaternion_angle_diff


class CardFlipEnv(BaseDexEnv):
    """Flip a thin playing card 180 degrees in hand."""

    def __init__(self, **kwargs) -> None:
        """Create the card-flip environment."""
        config = DexEnvConfig(
            xml_file="card_flip_scene.xml",
            object_body_name="card",
            rot_weight=2.0,
        )
        super().__init__(config=config, **kwargs)
        self._start_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _set_initial_state(self) -> None:
        super()._set_initial_state()
        self._start_quat = self._get_object_pose()[3:7].copy()

    def _sample_goal(self) -> np.ndarray:
        pos = self._get_object_pose()[:3]
        flip = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        goal_quat = normalize_quaternion(
            np.array(
                [
                    self._start_quat[0] * flip[0] - self._start_quat[1] * flip[1],
                    self._start_quat[0] * flip[1] + self._start_quat[1] * flip[0],
                    self._start_quat[0] * flip[2] + self._start_quat[3] * flip[1],
                    self._start_quat[0] * flip[3] - self._start_quat[2] * flip[1],
                ]
            )
        )
        return np.concatenate([pos, goal_quat])

    def _position_tolerance(self) -> float:
        return 0.005

    def _orientation_tolerance(self) -> float:
        return np.deg2rad(180.0)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        achieved = np.atleast_2d(achieved_goal)[0]
        flip_angle = quaternion_angle_diff(achieved[3:7], self._start_quat)
        pos_ok = np.linalg.norm(achieved[:3] - self._get_object_pose()[:3]) <= self._position_tolerance()
        return bool(pos_ok and flip_angle >= np.deg2rad(170.0))

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float | np.ndarray:
        achieved = np.atleast_2d(achieved_goal)
        desired = np.atleast_2d(desired_goal)
        rewards = []
        for ach, des in zip(achieved, desired, strict=True):
            flip_angle = quaternion_angle_diff(ach[3:7], self._start_quat)
            target = np.pi
            pos_dist = float(np.linalg.norm(ach[:3] - des[:3]))
            rewards.append(-pos_dist - abs(flip_angle - target))
        if achieved_goal.ndim == 1:
            return float(rewards[0])
        return np.asarray(rewards, dtype=np.float32)
