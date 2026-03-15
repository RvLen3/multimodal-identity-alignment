import csv
from pathlib import Path
from threading import Lock
from typing import Dict, List

_LOCK = Lock()
_CACHE: Dict[str, List[str]] = {}


def normalize_platform(platform: str) -> str:
    mapping = {
        "bilibili": "bili",
        "bili": "bili",
        "douyin": "douyin",
        "weibo": "weibo",
    }
    return mapping.get((platform or "").lower(), (platform or "").lower())


def _vectors_csv_path() -> Path:
    # backend/app/user_index.py -> backend/app -> backend -> project root
    return Path(__file__).resolve().parents[2] / "user_vectors.csv"


def _load_cache_if_needed() -> None:
    if _CACHE:
        return
    with _LOCK:
        if _CACHE:
            return

        by_platform = {"bili": set(), "douyin": set(), "weibo": set()}
        csv_path = _vectors_csv_path()
        if not csv_path.exists():
            _CACHE.update({k: [] for k in by_platform})
            return

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                platform = normalize_platform(row.get("platform", ""))
                uid = str(row.get("uid", "")).strip()
                if platform in by_platform and uid:
                    by_platform[platform].add(uid)

        _CACHE.update({k: sorted(v) for k, v in by_platform.items()})


def suggest_user_ids(platform: str, query: str, limit: int = 8) -> List[str]:
    _load_cache_if_needed()
    normalized_platform = normalize_platform(platform)
    if normalized_platform not in _CACHE:
        return []

    q = (query or "").strip()
    if not q:
        return _CACHE[normalized_platform][:limit]

    matched = [uid for uid in _CACHE[normalized_platform] if uid.startswith(q)]
    return matched[:limit]
