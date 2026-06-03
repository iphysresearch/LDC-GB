"""Tests for CLI argument parsing."""

from __future__ import annotations

import pytest

import GB_search
import merge_GB_signal_files


def test_gb_search_parse_args() -> None:
    args = GB_search.parse_args(["even1st", "42"])
    assert args.which_run == "even1st"
    assert args.batch_index == 42


def test_gb_search_parse_args_invalid_run() -> None:
    with pytest.raises(SystemExit):
        GB_search.parse_args(["invalid", "0"])


def test_merge_script_requires_which_run() -> None:
    with pytest.raises(SystemExit):
        merge_GB_signal_files.main([])
