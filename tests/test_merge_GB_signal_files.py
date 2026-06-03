"""Tests for merge_GB_signal_files helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from globalGB.search_utils_GB import PARAM_INDICES
from merge_GB_signal_files import (
    flatten_found_sources,
    load_sources_from_file,
    sort_by_frequency,
)
from tests.conftest import write_batch_h5


def test_load_sources_from_file(batch_h5_files: list[Path]) -> None:
    sources, wall_times, number_of_evaluations = load_sources_from_file(str(batch_h5_files[0]))

    assert sources.shape == (2, 8)
    assert wall_times.shape == (2,)
    assert number_of_evaluations.size == 0


def test_load_sources_from_file_minimal(tmp_path: Path) -> None:
    path = tmp_path / "minimal.h5"
    sources = np.array([[1e-3, 0.0, 1e-22, 1.0, 0.0, 0.0, 0.0, 0.0]])
    with h5py.File(path, "w") as f:
        f.create_dataset("recovered_sources", data=sources)

    loaded_sources, wall_times, number_of_evaluations = load_sources_from_file(str(path))
    assert np.array_equal(loaded_sources, sources)
    assert wall_times.size == 0
    assert number_of_evaluations.size == 0


def test_flatten_found_sources(batch_h5_files: list[Path]) -> None:
    raw = [load_sources_from_file(str(path)) for path in batch_h5_files]
    flat, total_time, total_evaluations = flatten_found_sources(raw)

    assert flat.shape == (3, 8)
    assert total_time == 7.0
    assert total_evaluations == 0


def test_flatten_found_sources_empty() -> None:
    flat, total_time, total_evaluations = flatten_found_sources([])
    assert flat.shape == (0, len(PARAM_INDICES))
    assert total_time == 0.0
    assert total_evaluations == 0


def test_sort_by_frequency() -> None:
    sources = np.array(
        [
            [1e-3, 0.0, 1e-22, 1.0, 0.0, 0.0, 0.0, 0.0],
            [3e-4, 0.0, 1e-22, 1.0, 0.0, 0.0, 0.0, 0.0],
            [5e-4, 0.0, 1e-22, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    sorted_sources = sort_by_frequency(sources)
    freqs = sorted_sources[:, PARAM_INDICES["Frequency"]]
    assert np.all(np.diff(freqs) >= 0)


def test_sort_by_frequency_empty() -> None:
    empty = np.empty((0, len(PARAM_INDICES)))
    assert sort_by_frequency(empty).shape == empty.shape


def test_merge_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip.h5"
    sources = np.array([[5e-4, 0.0, 1e-22, 1.0, 0.0, 0.0, 0.0, 0.0]])
    write_batch_h5(path, sources)

    raw = [load_sources_from_file(str(path))]
    flat, _, _ = flatten_found_sources(raw)
    sorted_sources = sort_by_frequency(flat)

    assert sorted_sources.shape == (1, 8)
    assert sorted_sources[0, PARAM_INDICES["Frequency"]] == 5e-4
