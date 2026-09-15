"""Seven-Turn Rescue T4 Combined v1: corrected Kaggle entrypoint."""
import importlib.util
from pathlib import Path
import sys


def _source_directory():
    filename = globals().get("__file__") or _source_directory.__code__.co_filename
    return Path(filename).resolve().parent


def _load_runtime():
    path = _source_directory() / "combined_runtime.py"
    spec = importlib.util.spec_from_file_location("_seven_turn_t4_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_RUNTIME = _load_runtime()
_AGENT = _RUNTIME.build_agent(sale_mode="adaptive_lead_two", crop_guard=True)


def agent(observation, configuration=None):
    return _AGENT(observation, configuration)


def kaggle_agent_t4_combined_v1(observation, configuration=None):
    """Unique final callable for Kaggle's raw Python source loader."""
    return agent(observation, configuration)
