"""Smoke-test the local zoom tool on one real image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from tools.zoom import ZoomTool  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    right = max(1, image.width // 2)
    bottom = max(1, image.height // 2)
    tool = ZoomTool(config={"work_dir": "/tmp/zoom", "output_timeout": 30})
    result = tool.call(
        {"code": f"display(original_image.crop((0, 0, {right}, {bottom})))"},
        image_map={"original_image": image},
    )
    if "observation_" not in str(result):
        raise RuntimeError(f"Zoom smoke test produced no image: {result}")
    print("Local zoom smoke test passed.")


if __name__ == "__main__":
    main()
