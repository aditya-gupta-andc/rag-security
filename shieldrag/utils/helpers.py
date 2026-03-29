"""Utility helpers — config loading, device detection, seeding, HMAC."""

import os
import sys
import yaml
import random
import hashlib
import hmac as hmac_mod
import logging

import numpy as np
import torch

logger = logging.getLogger("shieldrag")


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU: {name} ({vram:.1f} GB VRAM)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
            logger.info("Using Apple MPS")
        else:
            dev = torch.device("cpu")
            logger.info("Using CPU")
    else:
        dev = torch.device(preference)
    return dev


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_hmac(content: str, secret: str, algo: str = "sha256") -> str:
    return hmac_mod.new(
        secret.encode(), content.encode(), getattr(hashlib, algo)
    ).hexdigest()


def setup_logging(level: str = "INFO"):
    os.makedirs("results", exist_ok=True)
    fmt = "[%(asctime)s] %(levelname)-7s %(name)s — %(message)s"
    logging.basicConfig(
        level=getattr(logging, level), format=fmt, datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("results/shieldrag.log", mode="a"),
        ],
    )
