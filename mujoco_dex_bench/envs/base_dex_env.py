"""Base Gymnasium environment for dexterous MuJoCo manipulation tasks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_dex_bench.envs.utils import (
    body_pose,
    body_velocity,
    mjcf_path,
    normalize_quaternion,
    quaternion_angle_diff,
    random_quaternion,
)

logger = logging.getLogger(__name__)


@dataclass
class DexEnvConfig:
    """Configuration shared by dexterous manipulation environments."""

    xml_file: str
    object_body_name: str
    palm_body_name: str = "palm"
    max_episode_steps: int = 200
    n_substeps: int = 10
    control_penalty: float = 0.01
    success_bonus: float = 10.0
    pos_weight: float = 1.0
    rot_weight: float = 0.5
    contact_reward_scale: float = 0.5


class BaseDexEnv(gym.Env, ABC):
    """Goal-conditioned dexterous hand environment built on MuJoCo 3.x.

    Implements the Gymnasium GoalEnv interface with dict observations suitable
    for hindsight experience replay and Stable-Baselines3 MultiInputPolicy.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        config: DexEnvConfig | None = None,
        n_substeps: int = 10,
        render_mode: str | None = None,
    ) -> None:
        """Initialise the dexterous manipulation environment.

        Args:
            xml_path: Optional explicit path to scene XML. If ``None``, uses
                ``config.xml_file`` resolved via package resources.
            config: Task configuration dataclass.
            n_substeps: Number of MuJoCo substeps per Gymnasium step.
            render_mode: Rendering mode (``human``, ``rgb_array``, or ``None``).
        """
        super().__init__()
        if config is None:
            raise ValueError("DexEnvConfig must be provided")

        self.config = config
        self.n_substeps = n_substeps
        self.render_mode = render_mode

        if xml_path is None:
            xml_path = mjcf_path(config.xml_file)
        self.xml_path = Path(xml_path)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self._hand_joint_ids = [
            self.model.joint(name).id
            for name in [
                "index_mcp",
                "index_pip",
                "index_dip",
                "middle_mcp",
                "middle_pip",
                "middle_dip",
                "ring_mcp",
                "ring_pip",
                "ring_dip",
                "pinky_mcp",
                "pinky_pip",
                "pinky_dip",
                "thumb_cmc",
                "thumb_mcp",
                "thumb_ip",
            ]
        ]
        self._touch_sensor_names = [
            "touch_index",
            "touch_middle",
            "touch_ring",
            "touch_pinky",
            "touch_thumb",
        ]
        self._touch_sensor_ids = [
            self.model.sensor(name).id for name in self._touch_sensor_names
        ]

        self.n_actuators = self.model.nu
        self.n_hand_joints = len(self._hand_joint_ids)
        self.goal_dim = 7
        self.obs_dim = (
            2 * self.n_hand_joints
            + self.n_hand_joints
            + len(self._touch_sensor_ids)
            + 7
            + 6
            + 3
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.goal_dim,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.goal_dim,), dtype=np.float32
                ),
            }
        )

        self._desired_goal = np.zeros(self.goal_dim, dtype=np.float32)
        self._episode_steps = 0
        self._viewer: Any | None = None
        self._renderer: mujoco.Renderer | None = None

        low = self.model.actuator_ctrlrange[:, 0].copy()
        high = self.model.actuator_ctrlrange[:, 1].copy()
        self._ctrl_low = low.astype(np.float64)
        self._ctrl_high = high.astype(np.float64)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalised actions in ``[-1, 1]`` to actuator control ranges."""
        action = np.clip(action, -1.0, 1.0)
        return self._ctrl_low + 0.5 * (action + 1.0) * (self._ctrl_high - self._ctrl_low)

    def _get_joint_positions(self) -> np.ndarray:
        """Return hand joint positions in radians."""
        qpos_indices = [self.model.jnt_qposadr[j] for j in self._hand_joint_ids]
        return np.array([self.data.qpos[i] for i in qpos_indices], dtype=np.float64)

    def _get_joint_velocities(self) -> np.ndarray:
        """Return hand joint velocities."""
        qvel_indices = [self.model.jnt_dofadr[j] for j in self._hand_joint_ids]
        return np.array([self.data.qvel[i] for i in qvel_indices], dtype=np.float64)

    def _get_touch_readings(self) -> np.ndarray:
        """Return fingertip touch sensor values."""
        readings = []
        for sensor_id in self._touch_sensor_ids:
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            readings.append(float(self.data.sensordata[adr : adr + dim].sum()))
        return np.array(readings, dtype=np.float64)

    def _get_object_pose(self) -> np.ndarray:
        """Return object pose as ``[x, y, z, qw, qx, qy, qz]``."""
        pos, quat = body_pose(self.model, self.data, self.config.object_body_name)
        return np.concatenate([pos, quat]).astype(np.float64)

    def _get_object_velocity(self) -> np.ndarray:
        """Return object linear and angular velocity concatenated."""
        linvel, angvel = body_velocity(self.model, self.data, self.config.object_body_name)
        return np.concatenate([linvel, angvel]).astype(np.float64)

    def _get_palm_to_object(self) -> np.ndarray:
        """Return vector from palm centre to object."""
        palm_pos, _ = body_pose(self.model, self.data, self.config.palm_body_name)
        obj_pos = self._get_object_pose()[:3]
        return (obj_pos - palm_pos).astype(np.float64)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Construct the full dict observation."""
        qpos = self._get_joint_positions()
        qvel = self._get_joint_velocities()
        touch = self._get_touch_readings()
        obj_pose = self._get_object_pose()
        obj_vel = self._get_object_velocity()
        rel = self._get_palm_to_object()

        obs_vec = np.concatenate(
            [
                np.sin(qpos),
                np.cos(qpos),
                qvel,
                touch,
                obj_pose,
                obj_vel,
                rel,
            ]
        ).astype(np.float32)

        achieved_goal = obj_pose.astype(np.float32)
        return {
            "observation": obs_vec,
            "achieved_goal": achieved_goal,
            "desired_goal": self._desired_goal.copy(),
        }

    @abstractmethod
    def _sample_goal(self) -> np.ndarray:
        """Sample a task-specific desired goal."""

    def _reset_sim(self) -> None:
        """Reset simulation state and forward kinematics."""
        mujoco.mj_resetData(self.model, self.data)
        self._set_initial_state()
        mujoco.mj_forward(self.model, self.data)

    def _set_initial_state(self) -> None:
        """Set default joint configuration at episode start."""
        for joint_id in self._hand_joint_ids:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = 0.2

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        self._reset_sim()
        self._desired_goal = self._sample_goal().astype(np.float32)
        self._episode_steps = 0
        obs = self._get_obs()
        info: dict[str, Any] = {"is_success": False}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply action and advance simulation."""
        action = np.asarray(action, dtype=np.float64)
        ctrl = self._scale_action(action)
        self.data.ctrl[:] = ctrl

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._episode_steps += 1
        obs = self._get_obs()
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        reward = self.compute_reward(achieved, desired, {})
        reward -= self.config.control_penalty * float(np.sum(action**2))
        reward += self._contact_reward()

        success = self.is_success(achieved, desired)
        if success:
            reward += self.config.success_bonus

        terminated = success
        truncated = self._episode_steps >= self.config.max_episode_steps
        info: dict[str, Any] = {"is_success": bool(success)}
        return obs, float(reward), terminated, truncated, info

    def _contact_reward(self) -> float:
        """Optional contact-based shaping; override in contact-rich tasks."""
        return 0.0

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> float | np.ndarray:
        """Dense potential-based reward from pose distance to goal.

        Args:
            achieved_goal: Current object pose(s) ``(7,)`` or ``(N, 7)``.
            desired_goal: Target pose(s) with matching shape.
            info: Unused info dict for GoalEnv compatibility.

        Returns:
            Scalar reward for unbatched input, else array of shape ``(N,)``.
        """
        achieved = np.atleast_2d(achieved_goal)
        desired = np.atleast_2d(desired_goal)
        rewards = []
        for ach, des in zip(achieved, desired, strict=True):
            pos_dist = float(np.linalg.norm(ach[:3] - des[:3]))
            rot_dist = quaternion_angle_diff(ach[3:7], des[3:7])
            dense = -(
                self.config.pos_weight * pos_dist + self.config.rot_weight * rot_dist
            )
            rewards.append(dense)
        if achieved_goal.ndim == 1:
            return float(rewards[0])
        return np.asarray(rewards, dtype=np.float32)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Default success check using task tolerances."""
        achieved = np.atleast_2d(achieved_goal)[0]
        desired = np.atleast_2d(desired_goal)[0]
        pos_ok = np.linalg.norm(achieved[:3] - desired[:3]) <= self._position_tolerance()
        rot_ok = quaternion_angle_diff(achieved[3:7], desired[3:7]) <= self._orientation_tolerance()
        return bool(pos_ok and rot_ok)

    @abstractmethod
    def _position_tolerance(self) -> float:
        """Position success tolerance in metres."""

    @abstractmethod
    def _orientation_tolerance(self) -> float:
        """Orientation success tolerance in radians."""

    def render(self) -> np.ndarray | None:
        """Render the current state."""
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self._viewer.sync()
            return None
        raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self) -> None:
        """Release rendering resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer = None

    def sample_random_goal(self) -> np.ndarray:
        """Public helper to sample and normalise a random orientation goal."""
        goal = self._sample_goal()
        goal[3:7] = normalize_quaternion(goal[3:7])
        return goal.astype(np.float32)
