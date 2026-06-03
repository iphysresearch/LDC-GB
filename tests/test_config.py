"""Tests for globalGB.config."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from globalGB.config import get_config_path, load_config


def test_get_config_path_from_env(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LDC_CONFIG", str(config_file))
    assert get_config_path() == str(config_file)


def test_get_config_path_from_cwd(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("LDC_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_config_path() == str(config_file)


def test_get_config_path_missing_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LDC_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("LDC_CONFIG", str(missing))

    with pytest.raises(FileNotFoundError, match="LDC_CONFIG"):
        get_config_path()


def test_get_config_path_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LDC_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="GB search config not found"):
        get_config_path()


def test_load_config(config_file: Path, sample_config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LDC_CONFIG", str(config_file))
    loaded = load_config()
    assert loaded == sample_config


def test_example_config_is_valid_json() -> None:
    from importlib.resources import files

    example = files("globalGB").joinpath("GB_search_config.json.example")
    config = json.loads(example.read_text(encoding="utf-8"))
    assert config["data_set"] == "Mojito"
    assert "frequency_range" in config
