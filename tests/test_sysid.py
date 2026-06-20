"""System identification pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mujoco_dex_bench.sysid.fit_params import fit_params, generate_synthetic_dataset


def test_sysid_pipeline(tmp_path: Path) -> None:
    """Synthetic dataset generation and fast parameter fitting."""
    csv_path = tmp_path / "synthetic.csv"
    generate_synthetic_dataset("DexBallRoll-v0", csv_path, n_steps=50, seed=0)
    assert csv_path.exists()

    result = fit_params(
        str(csv_path),
        "DexBallRoll-v0",
        n_trials=10,
        output_xml=str(tmp_path / "fitted_params.xml"),
    )

    assert "mse" in result
    assert "mass_scale" in result
    assert "friction" in result
    assert "damping_scale" in result
    assert result["mse"] >= 0.0
    assert (tmp_path / "fitted_params.xml").exists()
