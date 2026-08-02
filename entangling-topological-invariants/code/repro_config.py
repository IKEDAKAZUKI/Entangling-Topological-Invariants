from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "seeds.json"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if "seeds" not in config:
        raise RuntimeError(f"Missing seed configuration: {CONFIG_PATH}")
    return config


def seed(name: str) -> int:
    try:
        return int(load_config()["seeds"][name])
    except KeyError as exc:
        raise KeyError(f"Unknown deterministic seed {name!r}") from exc
