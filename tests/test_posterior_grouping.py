"""Tests for source grouping helpers in GB_posterior."""

from __future__ import annotations

import numpy as np
import pytest

from globalGB.grouping import merge_ranges, ranges_overlap


@pytest.mark.parametrize(
    ("range1", "range2", "expected"),
    [
        ((0.0, 1.0), (0.5, 1.5), True),
        ((0.0, 1.0), (1.0, 2.0), True),
        ((0.0, 1.0), (1.1, 2.0), False),
        ((2.0, 3.0), (0.0, 1.0), False),
    ],
)
def test_ranges_overlap(range1, range2, expected) -> None:
    assert ranges_overlap(range1, range2) is expected


def test_merge_ranges() -> None:
    assert merge_ranges((0.2, 1.0), (0.5, 1.5)) == (0.2, 1.5)


def test_group_sources_by_overlap() -> None:
    """Mirror the grouping loop used in GB_posterior.main."""
    source_ranges = [(0.0, 1.0), (0.8, 1.5), (3.0, 4.0), (3.5, 4.5)]
    sources = [np.array([i]) for i in range(len(source_ranges))]

    groups: list[list] = []
    for source, source_range in zip(sources, source_ranges):
        merged = False
        for group in groups[-10:]:
            group_range, group_sources = group
            if ranges_overlap(group_range, source_range):
                group[0] = merge_ranges(group_range, source_range)
                group_sources.append(source)
                merged = True
                break
        if not merged:
            groups.append([source_range, [source]])

    assert len(groups) == 2
    assert groups[0][0] == (0.0, 1.5)
    assert len(groups[0][1]) == 2
    assert groups[1][0] == (3.0, 4.5)
    assert len(groups[1][1]) == 2
