"""Shared fixtures for LDC tests."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

N_GB_PARAMS = 8


@pytest.fixture
def sample_config() -> dict:
    return {
        "data_set": "Mojito",
        "dt": 2.5,
        "snr_threshold": 9.0,
        "tdi_generation": 2,
        "max_signals_per_window": 10,
        "max_signals_per_window_first_run": 3,
        "channel_combination": "AET",
        "frequency_range": [0.0003, 0.05],
        "seed": 1,
        "batch_size": 10,
        "data_path": "/tmp/test_data.h5",
        "save_path": "/tmp/test_output",
        "catalog_path": "/tmp/catalogues",
        "match_criteria": "overlap",
        "overlap_threshold": 0.9,
        "scaled_error_threshold": 0.3,
    }


@pytest.fixture
def config_file(tmp_path: Path, sample_config: dict) -> Path:
    config_dir = tmp_path / "globalGB"
    config_dir.mkdir()
    config_path = config_dir / "GB_search_config.json"
    config_path.write_text(json.dumps(sample_config), encoding="utf-8")
    return config_path


@pytest.fixture
def sample_boundaries() -> dict[str, list[float]]:
    return {
        "Frequency": [1e-4, 1e-2],
        "FrequencyDerivative": [-1e-18, 1e-18],
        "Amplitude": [-23.0, -20.0],
        "RightAscension": [0.0, 2 * np.pi],
        "Declination": [-1.0, 1.0],
        "Polarization": [0.0, 2 * np.pi],
        "Inclination": [-1.0, 1.0],
        "InitialPhase": [0.0, 2 * np.pi],
    }


@pytest.fixture
def sample_source_vector(sample_boundaries: dict[str, list[float]]) -> np.ndarray:
    from globalGB.search_utils_GB import PARAM_INDICES

    source = np.zeros(N_GB_PARAMS)
    source[PARAM_INDICES["Frequency"]] = 5e-4
    source[PARAM_INDICES["FrequencyDerivative"]] = 0.0
    source[PARAM_INDICES["Amplitude"]] = 1e-22
    source[PARAM_INDICES["RightAscension"]] = 1.0
    source[PARAM_INDICES["Declination"]] = 0.2
    source[PARAM_INDICES["Polarization"]] = 0.5
    source[PARAM_INDICES["Inclination"]] = 0.3
    source[PARAM_INDICES["InitialPhase"]] = 1.2
    return source


def write_batch_h5(
    path: Path,
    sources: np.ndarray,
    wall_times: np.ndarray | None = None,
    number_of_evaluations: np.ndarray | None = None,
) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("recovered_sources", data=sources)
        f.create_dataset("wall_times", data=wall_times if wall_times is not None else np.array([1.0]))
        if number_of_evaluations is not None:
            f.create_dataset("number_of_evaluations", data=number_of_evaluations)


@pytest.fixture
def batch_h5_files(tmp_path: Path) -> list[Path]:
    batch1 = np.array(
        [
            [3e-4, 0.0, 1e-22, 1.0, 0.1, 0.5, 0.2, 1.0],
            [1e-3, 0.0, 1e-22, 1.1, 0.2, 0.6, 0.3, 1.1],
        ]
    )
    batch2 = np.array([[5e-4, 0.0, 1e-22, 1.2, 0.3, 0.7, 0.4, 1.2]])

    paths = []
    for idx, sources in enumerate([batch1, batch2], start=1):
        path = tmp_path / f"batch_{idx}.h5"
        write_batch_h5(path, sources, wall_times=np.array([2.0, 3.0][: sources.shape[0]]))
        paths.append(path)
    return paths
