"""
Common utilities for nanocode.
"""

import os
import re
import logging
import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    if os.environ.get("NANOCODE_BASE_DIR"):
        nanocode_dir = os.environ.get("NANOCODE_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanocode_dir = os.path.join(cache_dir, "nanocode")
    os.makedirs(nanocode_dir, exist_ok=True)
    return nanocode_dir

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, flush=True, **kwargs)

def print_banner():
    banner = """
                                                       █████
                                                      ░░███
 ████████    ██████   ████████    ██████    ██████   ███████   ██████
░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ███░░███  ███░░███
 ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░ ░███ ░███ ░███████
 ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███░███ ░███ ░███░░░
 ████ █████░░████████ ████ █████░░██████ ░░██████ ░░████████░░██████
░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░   ░░░░░░░░  ░░░░░░
"""
    print0(banner)

def is_ddp():
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def compute_init():
    """Basic initialization for compute."""
    assert torch.cuda.is_available(), "CUDA is needed for a distributed run atm"
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp:
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda")
    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def get_gpu_flops():
    """Return peak bf16 FLOPS for the current GPU."""
    gpu_name = torch.cuda.get_device_name()
    # Peak bf16 TFLOPS (tensor cores) for common GPUs
    flops_map = {
        "H100 SXM": 989e12,
        "H100 PCIe": 756e12,
        "H100 NVL": 835e12,
        "A100 SXM": 312e12,
        "A100 PCIe": 312e12,
        "A100-SXM": 312e12,
        "A100-PCIE": 312e12,
        "L40S": 362e12,
        "A10G": 70e12,
        "RTX 4090": 330e12,
        "RTX 3090": 71e12,
    }
    for key, flops in flops_map.items():
        if key in gpu_name:
            return flops
    logger.warning(f"Unknown GPU '{gpu_name}', assuming H100 SXM FLOPS for MFU calculation")
    return 989e12

def compute_cleanup():
    """Companion function to compute_init."""
    if is_ddp():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures."""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
