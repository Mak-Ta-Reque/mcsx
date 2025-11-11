import os
import configparser
from typing import Tuple, Optional

_CONFIG_CACHE: Optional[configparser.ConfigParser] = None


def _load_config(path: str) -> configparser.ConfigParser:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    cfg = configparser.ConfigParser()
    try:
        with open(path, 'r') as f:
            cfg.read_file(f)
    except Exception:
        cfg.read_dict({})
    _CONFIG_CACHE = cfg
    return cfg


def get_train_config(
    default_epochs: int,
    default_lr: float,
    default_batch_size: int,
    specific_prefix: Optional[str] = None,
) -> Tuple[int, float, int]:
    """
    Unified training configuration for auto-training routines.

    Precedence (highest to lowest):
      1. Specific env vars: f"{specific_prefix}_EPOCHS/LR/BS" when prefix is provided
      2. Generic env vars: TRAIN_EPOCHS, TRAIN_LR, TRAIN_BS
      3. config.conf [Training] section: epochs, lr, batch_size
      4. Provided defaults
    """
    # 1) Specific env vars
    if specific_prefix:
        e = os.getenv(f"{specific_prefix}_EPOCHS")
        l = os.getenv(f"{specific_prefix}_LR")
        b = os.getenv(f"{specific_prefix}_BS")
        if e is not None or l is not None or b is not None:
            epochs = int(e) if e is not None else default_epochs
            lr = float(l) if l is not None else default_lr
            batch_size = int(b) if b is not None else default_batch_size
            return epochs, lr, batch_size

    # 2) Generic env vars
    ge = os.getenv("TRAIN_EPOCHS")
    gl = os.getenv("TRAIN_LR")
    gb = os.getenv("TRAIN_BS")
    if ge is not None or gl is not None or gb is not None:
        epochs = int(ge) if ge is not None else default_epochs
        lr = float(gl) if gl is not None else default_lr
        batch_size = int(gb) if gb is not None else default_batch_size
        return epochs, lr, batch_size

    # 3) Config file
    cfg_path = os.getenv('TRAIN_CONFIG_PATH', os.path.join(os.getcwd(), 'config.conf'))
    cfg = _load_config(cfg_path)
    if cfg.has_section('Training'):
        try:
            epochs = cfg.getint('Training', 'epochs', fallback=default_epochs)
            lr = cfg.getfloat('Training', 'lr', fallback=default_lr)
            batch_size = cfg.getint('Training', 'batch_size', fallback=default_batch_size)
            return epochs, lr, batch_size
        except Exception:
            pass

    # 4) Defaults
    return default_epochs, default_lr, default_batch_size
