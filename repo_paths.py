"""Repository path helpers.

Paths resolve in this order:
1. Environment variable (uppercase key, e.g. DATA_ROOT)
2. Local data directory under ``data/``
3. Repository-relative defaults (splits, outputs)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def data_root() -> Path:
    return _env_path("DATA_ROOT") or (REPO_ROOT / "data")


def output_root() -> Path:
    return _env_path("OUTPUT_ROOT") or (REPO_ROOT / "outputs")


def ssl_pretrain_data() -> Path:
    return _env_path("SSL_DATA_PATH") or (data_root() / "ssl_pretrain" / "chips_512_4ch")


def ssl_output_dir() -> Path:
    return _env_path("SSL_OUTPUT_DIR") or (output_root() / "ssl")


def ssl_moco_swin_checkpoint() -> Path | None:
    return _env_path("SSL_MOCO_SWIN_CHECKPOINT")


def ssl_fastsiam_checkpoint() -> Path | None:
    return _env_path("SSL_FASTSIAM_CHECKPOINT")


def ssl_mae_vit_checkpoint() -> Path | None:
    return _env_path("SSL_MAE_VIT_CHECKPOINT")


def ssl_mae_vit_base_checkpoint() -> Path | None:
    return _env_path("SSL_MAE_VIT_BASE_CHECKPOINT")


def sequoia_splits_root() -> Path:
    """Return Sequoia splits root for downstream experiments.
    
    Expected structure: Sequoia_train_loop/{5p,10p,25p,50p,75p,100p}/
    Each split contains: train_files.csv, validation_files.csv, 
                        Normalize_Bands.csv, Summary_train_val.csv
    """
    env = _env_path("SEQUOIA_SPLITS_ROOT")
    if env is not None:
        return env

    # Primary location: Sequoia_train_loop
    primary = REPO_ROOT / "Sequoia_train_loop"
    if primary.is_dir():
        return primary

    # Fallback to data root
    data_splits = data_root() / "sequoia" / "train_loop"
    if data_splits.is_dir():
        return data_splits

    # Legacy fallback
    legacy = REPO_ROOT / "splits" / "sequoia" / "label_train_val_loop"
    if legacy.is_dir():
        return legacy
    
    return primary


def rededge_splits_root() -> Path:
    """Return RedEdge splits root for downstream experiments.
    
    Expected structure: Rededge_train_loop/{5p,10p,25p,50p,75p,100p}/
    Each split contains: train_files.csv, validation_files.csv,
                        Normalize_Bands.csv, Summary_train_val.csv
    """
    env = _env_path("REDEDGE_SPLITS_ROOT")
    if env is not None:
        return env

    # Primary location: Rededge_train_loop
    primary = REPO_ROOT / "Rededge_train_loop"
    if primary.is_dir():
        return primary

    # Fallback to data root
    data_splits = data_root() / "rededge" / "train_loop"
    if data_splits.is_dir():
        return data_splits

    # Legacy fallback
    legacy = REPO_ROOT / "splits" / "rededge" / "test_data_512x512_20min_percent"
    if legacy.is_dir():
        return legacy
    
    return primary


def downstream_output_dir(dataset: str = "sequoia") -> Path:
    env = _env_path("DOWNSTREAM_OUTPUT_DIR")
    if env is not None:
        return env
    return output_root() / "downstream" / dataset
