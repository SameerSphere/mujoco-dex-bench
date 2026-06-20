"""MuJoCo Dexterous Hand Benchmark — environment registration."""

from __future__ import annotations

import re

import gymnasium as gym

TASKS: list[tuple[str, str, dict]] = [
    ("DexPenSpin-v0", "pen_spin", {}),
    ("DexKeyInsert-v0", "key_insert", {}),
    ("DexInHandReorient-v0", "in_hand_reorient", {}),
    ("DexChopstickGrasp-v0", "chopstick_grasp", {}),
    ("DexBallRoll-v0", "ball_roll", {}),
    ("DexPegInHole-v0", "peg_in_hole", {}),
    ("DexDoorOpen-v0", "door_open", {}),
    ("DexScissorsCut-v0", "scissors_cut", {}),
    ("DexCardFlip-v0", "card_flip", {}),
]


def _to_class(module_name: str) -> str:
    """Convert snake_case module name to PascalCase Env class name."""
    parts = module_name.split("_")
    if parts[-1] in {"spin", "insert", "reorient", "grasp", "roll", "hole", "open", "cut", "flip"}:
        return "".join(p.capitalize() for p in parts) + "Env"
    return "".join(p.capitalize() for p in parts) + "Env"


def _camel_env_class(module_name: str) -> str:
    """Map module names to explicit env class names."""
    mapping = {
        "pen_spin": "PenSpinEnv",
        "key_insert": "KeyInsertEnv",
        "in_hand_reorient": "InHandReorientEnv",
        "chopstick_grasp": "ChopstickGraspEnv",
        "ball_roll": "BallRollEnv",
        "peg_in_hole": "PegInHoleEnv",
        "door_open": "DoorOpenEnv",
        "scissors_cut": "ScissorsCutEnv",
        "card_flip": "CardFlipEnv",
    }
    return mapping.get(module_name, _to_class(module_name))


for gym_id, module_name, kwargs in TASKS:
    class_name = _camel_env_class(module_name)
    gym.register(
        id=gym_id,
        entry_point=f"mujoco_dex_bench.envs.{module_name}:{class_name}",
        max_episode_steps=200,
        kwargs=kwargs,
    )

__all__ = ["TASKS"]
