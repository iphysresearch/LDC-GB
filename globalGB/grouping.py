"""Helpers for grouping recovered sources by overlapping frequency ranges."""


def ranges_overlap(range1, range2):
    """Check if two frequency ranges overlap."""
    return range1[0] <= range2[1] and range1[1] >= range2[0]


def merge_ranges(range1, range2):
    """Merge two overlapping ranges."""
    return (min(range1[0], range2[0]), max(range1[1], range2[1]))
