import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D projection)


def load_extrinsics(txt_file: Path) -> torch.Tensor:
    """Load a trajectory text file into a (N, 3, 4) tensor.

    Each line must contain 12 floating-point numbers (row-major flattened [R|t])."""
    data: List[List[float]] = []
    with open(txt_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            nums = [float(x) for x in stripped.split()]
            if len(nums) != 12:
                raise ValueError(f"Line has {len(nums)} values, expected 12: '{stripped[:50]}...'")
            data.append(nums)
    if not data:
        raise ValueError("No extrinsics found in the file.")
    arr = torch.tensor(data, dtype=torch.float32).view(-1, 3, 4)
    return arr


def camera_center(extrinsic: torch.Tensor) -> torch.Tensor:
    """Compute camera center (-R^T t) from a single 3×4 extrinsic."""
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return -(R.T @ t)


def plot_trajectory(
    positions: np.ndarray,
    extrinsics: torch.Tensor,
    show_orientations: bool = False,
    axis_scale: float = 0.1,
    out_path: Path | None = None,
):
    fig = plt.figure(figsize=(12, 10))
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    # Plot path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "-o", color="blue", label="Camera path")

    # Highlight start (first) and end (last) positions
    ax.scatter(*positions[0], color="green", marker="^", s=150, label="Start")
    ax.scatter(*positions[-1], color="red", marker="s", s=150, label="End")

    # Annotate start and end
    ax.text(*positions[0], "  Start", color="green", fontsize=9)
    ax.text(*positions[-1], "  End", color="red", fontsize=9)

    # Optionally plot orientations
    if show_orientations:
        for i, extr in enumerate(extrinsics):
            R = extr[:3, :3].numpy()
            pos = positions[i]
            # X (red)
            ax.quiver(*pos, *R[:, 0], length=axis_scale, color="r")
            # Y (green)
            ax.quiver(*pos, *R[:, 1], length=axis_scale, color="g")
            # Z (blue)
            ax.quiver(*pos, *R[:, 2], length=axis_scale, color="b")

    # Labels and equal aspect ratio
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        print(f"Figure saved to {out_path}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a camera trajectory in 3D")
    parser.add_argument("trajectory", type=str, help="Path to trajectory txt file (one 3x4 extrinsic per line)")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure instead of showing")
    parser.add_argument("--orient", action="store_true", help="Plot camera orientation axes as well")
    parser.add_argument("--axis_scale", type=float, default=0.1, help="Length of orientation axes")
    return parser.parse_args()


def main():
    args = parse_args()
    extrinsics = load_extrinsics(Path(args.trajectory))

    # Compute positions
    centers = torch.stack([camera_center(E) for E in extrinsics])  # (N, 3)
    positions = centers.numpy()

    plot_trajectory(
        positions,
        extrinsics,
        show_orientations=args.orient,
        axis_scale=args.axis_scale,
        out_path=Path(args.save) if args.save else None,
    )


if __name__ == "__main__":
    main()
