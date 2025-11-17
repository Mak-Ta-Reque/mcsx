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


def _parse_bool(value: str) -> Optional[bool]:
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    lowered = value.strip().lower()
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    return None


def is_auto_train_enabled(default: bool = True) -> bool:
    """Check whether auto-training missing checkpoints is allowed.

    Precedence (highest to lowest):
      1. Env var ``ALLOW_AUTO_TRAIN`` (accepts 1/0, true/false, yes/no, on/off)
      2. ``config.conf`` [Training] ``train_on_requested_dataset`` flag
      3. Provided default value
    """
    env_value = os.getenv("ALLOW_AUTO_TRAIN")
    if env_value is not None:
        parsed = _parse_bool(env_value)
        if parsed is not None:
            return parsed

    cfg_path = os.getenv('TRAIN_CONFIG_PATH', os.path.join(os.getcwd(), 'config.conf'))
    cfg = _load_config(cfg_path)
    if cfg.has_section('Training') and cfg.has_option('Training', 'train_on_requested_dataset'):
        try:
            return cfg.getboolean('Training', 'train_on_requested_dataset', fallback=default)
        except ValueError:
            pass

    return default


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


def is_warmup_enabled(default: bool = False) -> bool:
    """Determine whether post-load warm-up training should run."""

    env_value = os.getenv("ENABLE_WARM_UP")
    if env_value is None:
        env_value = os.getenv("WARM_UP")
    if env_value is not None:
        parsed = _parse_bool(env_value)
        if parsed is not None:
            return parsed

    cfg_path = os.getenv('TRAIN_CONFIG_PATH', os.path.join(os.getcwd(), 'config.conf'))
    cfg = _load_config(cfg_path)
    if cfg.has_section('Training') and cfg.has_option('Training', 'warm_up'):
        try:
            return cfg.getboolean('Training', 'warm_up', fallback=default)
        except ValueError:
            pass

    return default


def get_warmup_config(
    default_epochs: int = 10,
    default_lr: float = 5e-4,
    default_batch_size: int = 128,
) -> Tuple[int, float, int]:
    """Retrieve warm-up training hyperparameters (epochs, lr, batch size)."""

    env_epochs = os.getenv("WARM_UP_EPOCHS")
    env_lr = os.getenv("WARM_UP_LR")
    env_bs = os.getenv("WARM_UP_BS") or os.getenv("WARM_UP_BATCH_SIZE")
    if env_epochs is not None or env_lr is not None or env_bs is not None:
        epochs = int(env_epochs) if env_epochs is not None else default_epochs
        lr = float(env_lr) if env_lr is not None else default_lr
        batch_size = int(env_bs) if env_bs is not None else default_batch_size
        return epochs, lr, batch_size

    cfg_path = os.getenv('TRAIN_CONFIG_PATH', os.path.join(os.getcwd(), 'config.conf'))
    cfg = _load_config(cfg_path)
    if cfg.has_section('Training'):
        try:
            epochs = cfg.getint('Training', 'warm_up_epochs', fallback=default_epochs)
            lr = cfg.getfloat('Training', 'warm_up_lr', fallback=default_lr)
            batch_size = cfg.getint('Training', 'warm_up_batch_size', fallback=default_batch_size)
            return epochs, lr, batch_size
        except Exception:
            pass

    return default_epochs, default_lr, default_batch_size
