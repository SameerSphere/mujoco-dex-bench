"""Shared math and path utilities for dexterous manipulation environments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MJCF_DIR = Path(__file__).resolve().parent.parent / "mjcf"


def mjcf_path(filename: str) -> Path:
    """Return absolute path to an MJCF scene file bundled with the package.

    Args:
        filename: Scene XML filename (e.g. ``pen_spin_scene.xml``).

    Returns:
        Resolved path to the MJCF file.
    """
    return MJCF_DIR / filename


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length.

    Args:
        quat: Quaternion in ``(w, x, y, z)`` order.

    Returns:
        Unit quaternion with the same shape as the input.
    """
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return quat / norm


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random unit quaternion.

    Args:
        rng: NumPy random generator.

    Returns:
        Quaternion array of shape ``(4,)`` in ``(w, x, y, z)`` order.
    """
    u1, u2, u3 = rng.random(3)
    return normalize_quaternion(
        np.array(
            [
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),
                np.sqrt(u1) * np.cos(2 * np.pi * u3),
            ],
            dtype=np.float64,
        )
    )


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in ``(w, x, y, z)`` order.

    Args:
        q1: First quaternion.
        q2: Second quaternion.

    Returns:
        Product quaternion ``q1 * q2``.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return normalize_quaternion(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def quaternion_conjugate(quat: np.ndarray) -> np.ndarray:
    """Return the conjugate of a quaternion.

    Args:
        quat: Quaternion in ``(w, x, y, z)`` order.

    Returns:
        Conjugate quaternion.
    """
    w, x, y, z = quat
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quaternion_angle_diff(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute the geodesic angle between two quaternions in radians.

    Args:
        q1: First quaternion ``(w, x, y, z)``.
        q2: Second quaternion ``(w, x, y, z)``.

    Returns:
        Angular distance in radians.
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    rel = quaternion_multiply(quaternion_conjugate(q1), q2)
    w = float(np.clip(abs(rel[0]), -1.0, 1.0))
    return 2.0 * np.arccos(w)


def rotation_about_axis_angle(quat: np.ndarray, axis: np.ndarray) -> float:
    """Extract rotation angle about a given axis from a quaternion.

    Args:
        quat: Quaternion ``(w, x, y, z)``.
        axis: Unit axis vector.

    Returns:
        Signed rotation angle in radians about ``axis``.
    """
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    q = normalize_quaternion(quat)
    vec = q[1:]
    proj = np.dot(vec, axis) * axis
    angle = 2.0 * np.arctan2(np.linalg.norm(proj), q[0])
    return float(angle)


def body_pose(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return body position and quaternion from MuJoCo data.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_name: Name of the body.

    Returns:
        Tuple of position ``(3,)`` and quaternion ``(w, x, y, z)`` arrays.
    """
    body_id = model.body(body_name).id
    pos = data.xpos[body_id].copy()
    quat = data.xquat[body_id].copy()
    return pos.astype(np.float64), quat.astype(np.float64)


def body_velocity(model, data, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return body linear and angular velocity.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_name: Name of the body.

    Returns:
        Tuple of linear velocity ``(3,)`` and angular velocity ``(3,)``.
    """
    body_id = model.body(body_name).id
    linvel = data.cvel[body_id, 3:6].copy()
    angvel = data.cvel[body_id, :3].copy()
    return linvel.astype(np.float64), angvel.astype(np.float64)


def geom_pair_in_contact(model, data, geom_a: str, geom_b: str) -> bool:
    """Check whether two named geoms are in contact.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        geom_a: First geom name.
        geom_b: Second geom name.

    Returns:
        True if the geoms are in contact.
    """
    id_a = model.geom(geom_a).id
    id_b = model.geom(geom_b).id
    for i in range(data.ncon):
        con = data.contact[i]
        pair = (con.geom1, con.geom2)
        if pair == (id_a, id_b) or pair == (id_b, id_a):
            return True
    return False
