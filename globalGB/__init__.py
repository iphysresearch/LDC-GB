from .config import get_config_path, load_config

__all__ = [
    "boundaries_dict",
    "GBSearchRunner",
    "get_config_path",
    "load_config",
    "PARAM_INDICES",
    "PARAM_NAMES",
    "GB_pe",
    "Segment_GB_Searcher",
    "create_frequency_windows",
    "tdi_subtraction",
    "GB_Searcher",
]


def __getattr__(name: str):
    if name == "boundaries_dict":
        from .GB_boundaries import boundaries_dict

        return boundaries_dict
    if name == "GBSearchRunner":
        from .GB_runner import GBSearchRunner

        return GBSearchRunner
    if name in {
        "PARAM_INDICES",
        "PARAM_NAMES",
        "GB_pe",
        "Segment_GB_Searcher",
        "create_frequency_windows",
        "tdi_subtraction",
        "GB_Searcher",
    }:
        from . import search_utils_GB

        return getattr(search_utils_GB, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
