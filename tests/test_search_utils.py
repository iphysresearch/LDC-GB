"""Tests for lightweight globalGB.search_utils_GB helpers."""

from __future__ import annotations

import numpy as np

from globalGB.search_utils_GB import (
    GBConfig,
    PARAM_NAMES,
    create_frequency_windows,
    frequency_derivative,
    max_signal_bandwidth,
    scaleto01,
    scaletooriginal,
    transform_parameters_from_01,
    transform_parameters_to_01,
)


def test_gbconfig_sets_attributes(sample_config: dict) -> None:
    config = GBConfig(sample_config)
    assert config.data_set == "Mojito"
    assert config.frequency_range == [0.0003, 0.05]
    assert config.batch_size == 10


def test_frequency_derivative_is_positive() -> None:
    fdot = frequency_derivative(1e-3, Mc=1.0)
    assert fdot > 0


def test_max_signal_bandwidth_increases_with_frequency() -> None:
    t_obs = 31536000.0
    bw_low = max_signal_bandwidth(1e-4, t_obs)
    bw_high = max_signal_bandwidth(1e-3, t_obs)
    assert bw_high > bw_low


def test_create_frequency_windows_covers_search_range() -> None:
    search_range = [1e-4, 2e-3]
    windows = create_frequency_windows(search_range, Tobs=31536000.0)

    assert len(windows) > 0
    assert windows[0][0] == search_range[0]
    assert windows[-1][1] >= search_range[0]
    for lower, upper in windows:
        assert upper > lower


def test_scaleto01_and_scaletooriginal_roundtrip(
    sample_boundaries: dict[str, list[float]],
    sample_source_vector: np.ndarray,
) -> None:
    scaled = scaleto01(sample_source_vector, sample_boundaries)
    restored = scaletooriginal(scaled, sample_boundaries)

    assert scaled.shape == (len(PARAM_NAMES),)
    assert np.all(scaled >= 0.0)
    assert np.all(scaled <= 1.0)
    np.testing.assert_allclose(restored, sample_source_vector, rtol=1e-10, atol=1e-12)


def test_transform_parameters_unit_cube_roundtrip() -> None:
    boundaries = np.array([[0.0, 1.0], [0.0, 2.0], [-1.0, 1.0]])
    params = np.array([0.25, 1.0, 0.0])

    params01 = transform_parameters_to_01(params, boundaries)
    restored = transform_parameters_from_01(params01, boundaries)

    np.testing.assert_allclose(params01, [0.25, 0.5, 0.5])
    np.testing.assert_allclose(restored, params)


def test_param_names_length() -> None:
    assert len(PARAM_NAMES) == 8
