import re
import shutil
from pathlib import Path


def copy_latest_checkpoint(
    checkpoint_dir: str = "../cdiff_logs/run/cdiff_best/checkpoints/",
    target_dir: str = "./",
    target_name: str = "weights_cdiff.ckpt"
) -> Path:
    """
    Copy the checkpoint with the highest epoch number from checkpoint_dir
    to target_dir and rename it to weights_cdiff.ckpt.
    """
    checkpoint_dir = Path(checkpoint_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"epoch=(\d+)")
    latest_epoch = -1
    latest_ckpt = None

    for ckpt in checkpoint_dir.glob("*.ckpt"):
        match = pattern.search(ckpt.name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt = ckpt

    if latest_ckpt is None:
        raise RuntimeError(
            f"No checkpoint matching 'epoch=...' found in {checkpoint_dir}"
        )

    target_path = target_dir / target_name
    shutil.copy2(latest_ckpt, target_path)

    return target_path



if __name__ == "__main__":
    copy_latest_checkpoint()