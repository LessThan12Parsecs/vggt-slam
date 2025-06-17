import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch


def load_extrinsics(txt_file: Path) -> torch.Tensor:
    data: List[List[float]] = []
    with open(txt_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split() if x.strip()]
            if not vals:
                continue
            if len(vals) != 12:
                raise ValueError("Each line must have 12 numbers (flattened 3x4 extrinsic)")
            data.append(vals)
    if not data:
        raise ValueError("No extrinsics found in file")
    return torch.tensor(data, dtype=torch.float32).view(-1, 3, 4)


def camera_centers(extrinsics: torch.Tensor) -> torch.Tensor:
    R = extrinsics[:, :, :3]
    t = extrinsics[:, :, 3]
    # C = -R^T t
    centers = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
    return centers  # (N, 3)


def fit_plane_svd(points: torch.Tensor):
    # points: (N,3)
    centroid = points.mean(dim=0)
    centered = points - centroid
    U, S, Vt = torch.linalg.svd(centered)
    normal = Vt[-1]  # smallest singular value
    return centroid, normal / torch.linalg.norm(normal)


def project_points_to_plane(points: torch.Tensor, centroid: torch.Tensor, normal: torch.Tensor):
    # signed distance along normal for each point
    dists = torch.sum((points - centroid) * normal, dim=1, keepdim=True)
    proj = points - dists * normal  # (N,3)
    return proj


def update_extrinsics_with_centers(extrinsics: torch.Tensor, new_centers: torch.Tensor):
    updated = extrinsics.clone()
    for i in range(extrinsics.shape[0]):
        R = extrinsics[i, :, :3]
        t_new = -torch.matmul(R, new_centers[i])
        updated[i, :, 3] = t_new
    return updated


def save_extrinsics(extrinsics: torch.Tensor, path: Path):
    with open(path, "w") as f:
        for E in extrinsics:
            vals = E.reshape(-1).tolist()
            f.write(" ".join(f"{v:.6f}" for v in vals) + "\n")


def parse_args():
    p = argparse.ArgumentParser("Flatten a camera trajectory onto best-fit plane")
    p.add_argument("input", type=str, help="Input trajectory file (txt)")
    p.add_argument("--output", type=str, default="flattened_trajectory.txt", help="Output txt file")
    p.add_argument("--rotate_to_xy", action="store_true", help="Rotate so that plane aligns with Z=0 (XY plane)")
    return p.parse_args()


def main():
    args = parse_args()
    extr = load_extrinsics(Path(args.input))
    centers = camera_centers(extr)

    centroid, normal = fit_plane_svd(centers)
    projected = project_points_to_plane(centers, centroid, normal)

    # Replace centers with projected ones
    extr_flat = update_extrinsics_with_centers(extr, projected)

    if args.rotate_to_xy:
        # Build rotation that maps plane normal to Z axis
        z_axis = torch.tensor([0.0, 0.0, 1.0])
        v = torch.cross(normal, z_axis)
        if torch.linalg.norm(v) < 1e-6:
            R_align = torch.eye(3)
        else:
            s = torch.linalg.norm(v)
            c = torch.dot(normal, z_axis)
            vx = torch.tensor(
                [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
            )  # skew-symmetric
            R_align = torch.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        # Rotate centers and update extrinsics accordingly
        for i in range(extr_flat.shape[0]):
            R = extr_flat[i, :, :3]
            t = extr_flat[i, :, 3]
            R_new = R @ R_align.T
            t_new = R_align @ t
            extr_flat[i, :, :3] = R_new
            extr_flat[i, :, 3] = t_new
        centers_rot = camera_centers(extr_flat)
        projected = centers_rot  # for residual print after rotation
        normal = torch.tensor([0.0, 0.0, 1.0])  # aligned

    save_extrinsics(extr_flat, Path(args.output))
    print(f"Flattened trajectory saved to {args.output}")
    # Print residual mean abs distance from plane
    residual = torch.abs(torch.sum((projected - centroid) * normal, dim=1)).mean()
    print(f"Average abs deviation from plane: {residual:.4f} units")


if __name__ == "__main__":
    main() 