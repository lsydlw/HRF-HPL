"""数据集路径与规模配置（支持从 pickle 自动推断）。"""
from __future__ import annotations

import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT.parent / "ICASSP2024_ASTHL-main" / "datasets"

# 仅作无法读取文件时的回退
FALLBACK_STATS = {
    "NYC": (834, 3835),
    "TKY": (2173, 7038),
    "Gowalla": (3467, 22527),
}


def get_data_root(dataset: str) -> Path:
    return DATASETS_ROOT / dataset


def _poi_pickle_path(data_root: Path, dataset: str) -> Path:
    return data_root / f"{dataset}_pois_coos_poi_zero.pkl"


def _train_pickle_path(data_root: Path) -> Path:
    return data_root / "train_poi_zero.txt"


def load_dataset_stats(dataset: str, data_root: Path | None = None) -> tuple[int, int, int]:
    """
    返回 (num_users, num_pois, padding_idx)。
    优先从 datasets/{dataset}/ 下 pickle 自动统计。
    """
    data_root = data_root or get_data_root(dataset)
    poi_path = _poi_pickle_path(data_root, dataset)
    train_path = _train_pickle_path(data_root)

    if poi_path.exists() and train_path.exists():
        with open(poi_path, "rb") as f:
            poi_coos = pickle.load(f)
        with open(train_path, "rb") as f:
            train_trajs, _ = pickle.load(f)

        if isinstance(poi_coos, dict):
            num_pois = len(poi_coos)
        else:
            num_pois = len(poi_coos)
        num_users = len(train_trajs)
        return num_users, num_pois, num_pois

    if dataset in FALLBACK_STATS:
        num_users, num_pois = FALLBACK_STATS[dataset]
        return num_users, num_pois, num_pois

    raise FileNotFoundError(
        f"找不到数据集 {dataset}，请确认目录存在: {data_root}\n"
        f"  需要: train_poi_zero.txt, test_poi_zero.txt, {dataset}_pois_coos_poi_zero.pkl"
    )


def dataset_file_paths(dataset: str, data_root: Path | None = None) -> dict[str, Path]:
    data_root = data_root or get_data_root(dataset)
    return {
        "data_root": data_root,
        "train": data_root / "train_poi_zero.txt",
        "test": data_root / "test_poi_zero.txt",
        "poi_coos": _poi_pickle_path(data_root, dataset),
    }
