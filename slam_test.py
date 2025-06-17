import argparse
import glob
import os
from typing import List, Tuple, Optional

import torch
import rerun as rr  # NEW: Rerun for live 3-D visualization
import numpy as np

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Optional heavy-duty utilities are imported lazily to avoid cost when not used
try:
    from vggt.utils.geometry import unproject_depth_map_to_point_map  # heavy
except Exception:
    unproject_depth_map_to_point_map = None  # type: ignore


def make_homogeneous(extrinsic: torch.Tensor) -> torch.Tensor:
    """Convert 3x4 extrinsic matrix to 4x4 homogeneous form."""
    last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 4)
    return torch.cat([extrinsic, last_row], dim=0)


def dehomogenize(mat: torch.Tensor) -> torch.Tensor:
    """Convert 4x4 homogeneous matrix back to 3x4 extrinsic."""
    return mat[:3, :]


def camera_center_from_extrinsic(extrinsic: torch.Tensor) -> torch.Tensor:
    """Compute camera center in world coordinates from a single 3x4 extrinsic.

    In OpenCV convention: X_cam = R * X_world + t  =>  Center_world = -R^T t
    """
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return -(R.T @ t)


def align_window(
    prev_global_extr: torch.Tensor,
    curr_local_extr: torch.Tensor,
) -> torch.Tensor:
    """Compute transformation that maps current window's local world frame to the global one.

    Args:
        prev_global_extr: 3x4 extrinsic of the overlapping frame expressed in *global* coords.
        curr_local_extr: 3x4 extrinsic of the same frame expressed in *current local* coords.

    Returns:
        4x4 transform T such that  X_global = T @ X_local.
    """
    Eg = make_homogeneous(prev_global_extr)  # 4x4
    El = make_homogeneous(curr_local_extr)
    # T = Eg^{-1} * El   (derived in analysis section)
    return torch.linalg.inv(Eg) @ El


def run_vggt_on_chunk(
    model: VGGT,
    image_paths: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run VGGT on a list of images and return extrinsics tensor of shape (S, 3, 4)."""
    images = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, _ = model.aggregator(images[None])
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        else:
            aggregated_tokens_list, _ = model.aggregator(images[None])
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]

        extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    # Remove batch dimension (1, S, 3, 4) -> (S, 3, 4)
    return extrinsic.squeeze(0).float().cpu()


# --------------------------------------------------------------------------------------
# Incremental SLAM with optional Rerun live visualisation
# --------------------------------------------------------------------------------------

def incremental_slam(
    image_paths: List[str],
    chunk_size: int = 5,
    overlap: int = 1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_rerun: bool = False,
    show_points: bool = False,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Run a simple incremental SLAM over the sequence of images.

    Returns:
        global_extrinsics: list of 3x4 tensors in trajectory order (one per image, deduped).
        img_order:          list of image paths corresponding to the extrinsics above.
    """
    assert 0 < overlap < chunk_size, "overlap must be in (0, chunk_size)"

    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    print("Loading VGGT model ...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    global_extrinsics: List[torch.Tensor] = []
    img_order: List[str] = []

    step = chunk_size - overlap
    start = 0
    chunk_idx = 0

    # ------------------------------------------------------------------
    # Helper for Rerun
    # ------------------------------------------------------------------
    if use_rerun:
        # Log the world coordinate frame once.
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    def log_to_rerun_global(points_global: Optional[torch.Tensor] = None, colors: Optional[torch.Tensor] = None) -> None:
        """Send the accumulated global trajectory and camera poses to Rerun."""
        if not use_rerun or len(global_extrinsics) == 0:
            return
        # Stack camera centres
        ctrs = torch.stack([camera_center_from_extrinsic(E) for E in global_extrinsics]).cpu().numpy()
        # Draw the growing line and points (blue)
        rr.log("world/trajectory", rr.LineStrips3D([ctrs], colors=[[0, 0, 255]]))
        rr.log("world/centers", rr.Points3D(ctrs, colors=[[0, 0, 255]] * len(ctrs), radii=0.02))
        # Log each camera transform (cam -> world)
        for i, E in enumerate(global_extrinsics[-step:]):  # only newest chunk for efficiency
            T = torch.linalg.inv(make_homogeneous(E)).cpu().numpy()
            # Split into rotation and translation for Transform3D
            R = T[:3, :3]
            t = T[:3, 3]
            rr.log(
                f"world/cam_{i}",
                rr.Transform3D(translation=t, mat3x3=R),
            )

        # Optionally log point cloud
        if show_points and points_global is not None and colors is not None:
            if isinstance(points_global, torch.Tensor):
                pts_np = points_global.cpu().numpy()
            else:
                pts_np = points_global

            if isinstance(colors, torch.Tensor):
                cols_np = (colors.cpu().numpy() * 255).astype("uint8")
            else:
                cols_np = (colors * 255).astype("uint8")
            rr.log(
                "world/points",
                rr.Points3D(pts_np, colors=cols_np, radii=0.002),
            )

    while start < len(image_paths):
        end = min(start + chunk_size, len(image_paths))
        chunk_paths = image_paths[start:end]

        if len(chunk_paths) < overlap and start != 0:
            # Not enough new images to form a chunk; stop processing.
            break

        chunk_idx += 1
        print(f"\nProcessing chunk {chunk_idx} ... Images {start} to {end - 1}")

        # ------------------------------------------------------------
        # VGGT forward pass – extrinsics (+ depth if needed)
        # ------------------------------------------------------------
        images_tensor = load_and_preprocess_images(chunk_paths).to(device)

        with torch.no_grad():
            if device.type == "cuda":
                autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
            else:
                # CPU autocast is no-op
                class _Dummy:
                    def __enter__(self):
                        return None
                    def __exit__(self, *args):
                        return False

                autocast_ctx = _Dummy()

            with autocast_ctx:
                aggregated_tokens_list, ps_idx = model.aggregator(images_tensor[None])
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]

                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc, images_tensor.shape[-2:]
                )

                # Remove batch dim -> (S,3,4)
                local_extrs = extrinsic.squeeze(0).float().cpu()

                # If we want points, run depth head
                if show_points:
                    depth_map, depth_conf = model.depth_head(
                        aggregated_tokens_list, images_tensor[None], ps_idx
                    )
                    depth_map = depth_map.squeeze(0)  # (S,H,W,1)
                    world_pts_map = unproject_depth_map_to_point_map(
                        depth_map, local_extrs, intrinsic.squeeze(0)
                    )  # (S,H,W,3)
                    # Convert to torch for downstream handling
                    if isinstance(world_pts_map, np.ndarray):
                        world_pts_map = torch.from_numpy(world_pts_map).float()
                else:
                    world_pts_map = None

        if start == 0:
            # First window sets the global reference frame
            for extr in local_extrs:
                global_extrinsics.append(extr)
            img_order.extend(chunk_paths)
        else:
            # Align current window to global using the first frame (which overlaps with last global frame)
            prev_global_extr = global_extrinsics[-overlap]
            curr_local_extr = local_extrs[0]
            T = align_window(prev_global_extr, curr_local_extr)
            T_inv = torch.linalg.inv(T)

            # Transform all local extrinsics to global frame
            for i, extr_local in enumerate(local_extrs):
                if i < overlap:
                    # Skip overlapped frames to avoid duplicates
                    continue
                Eg = dehomogenize(make_homogeneous(extr_local) @ T_inv)
                global_extrinsics.append(Eg)
                img_order.append(chunk_paths[i])

        # ------------------------------------------------------------------
        # Rerun: visualise the local window before registration (red)
        # ------------------------------------------------------------------
        if use_rerun:
            centers_local = torch.stack(
                [camera_center_from_extrinsic(E) for E in local_extrs]
            ).cpu().numpy()
            rr.log(
                f"chunk_{chunk_idx}/local_raw",
                rr.Points3D(
                    centers_local,
                    colors=[[255, 0, 0]] * len(centers_local),
                    radii=0.025,
                ),
            )

        # ------------------------------------------------------------
        # Prepare global point map (optional)
        # ------------------------------------------------------------
        if show_points and world_pts_map is not None:
            # Apply T to local points if not first chunk
            if start == 0:
                pts_global = world_pts_map.reshape(-1, 3)
            else:
                # Homogenize & apply T to every point
                S, H, W, _ = world_pts_map.shape
                pts_h = torch.cat(
                    [world_pts_map.reshape(-1, 3), torch.ones((S * H * W, 1))], dim=1
                )  # (N,4)
                pts_glob = (T @ pts_h.T).T[:, :3]  # (N,3)
                pts_global = pts_glob

            # Downsample to keep viewer responsive
            if pts_global.shape[0] > 200_000:
                idx = torch.randperm(pts_global.shape[0])[:200_000]
                pts_global = pts_global[idx]

            # Corresponding colours
            cols = (
                images_tensor.cpu().permute(0, 2, 3, 1).contiguous().view(-1, 3)
            )  # (N,3)
        else:
            pts_global = None
            cols = None

        # After each chunk we update the viewer
        log_to_rerun_global(pts_global, cols)

        # Advance to next chunk
        start += step

    return global_extrinsics, img_order


def save_trajectory(extrinsics: List[torch.Tensor], out_path: str):
    """Save 3x4 extrinsics to a text file (one line per camera, flattened)."""
    with open(out_path, "w") as f:
        for E in extrinsics:
            vals = E.reshape(-1).tolist()
            f.write(" ".join(f"{v:.6f}" for v in vals) + "\n")
    print(f"Trajectory saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple incremental SLAM on top of VGGT")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing input images")
    parser.add_argument("--pattern", type=str, default="*.jpg", help="Glob pattern to match images")
    parser.add_argument("--chunk_size", type=int, default=2, help="Number of images per window")
    parser.add_argument("--overlap", type=int, default=1, help="Number of overlapping images between consecutive windows (must be < chunk_size)")
    parser.add_argument("--output", type=str, default="trajectory.txt", help="Where to save resulting extrinsics")
    parser.add_argument("--rerun", action="store_true", help="Enable live 3-D visualisation using Rerun viewer")
    parser.add_argument("--show_points", action="store_true", help="Log point clouds (depth unprojection) to the Rerun viewer — slower")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "vulkan", "gl", "dx12", "metal"],
        default="auto",
        help="Graphics backend for Rerun viewer (maps to env var RERUN_WGPU_BACKEND)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Collect images
    image_paths = sorted(glob.glob(os.path.join(args.image_folder, args.pattern)))
    if len(image_paths) < args.chunk_size:
        raise ValueError("Not enough images to form a single chunk")

    print(f"Found {len(image_paths)} images. Running incremental SLAM...")

    # ------------------------------------------------------------------
    # Optionally start the Rerun viewer
    # ------------------------------------------------------------------
    if args.rerun:
        # Select wgpu backend if requested
        if args.backend != "auto":
            os.environ["RERUN_WGPU_BACKEND"] = args.backend
            print(f"Using Rerun graphics backend: {args.backend}")
        rr.init("VGGT_SLAM", spawn=True)  # launch external viewer

    global_extrinsics, img_order = incremental_slam(
        image_paths,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        use_rerun=args.rerun,
        show_points=args.show_points,
    )

    # Save trajectory
    save_trajectory(global_extrinsics, args.output)

    # Print summary
    print(f"Processed {len(img_order)} images. Trajectory length: {len(global_extrinsics)}")


if __name__ == "__main__":
    main()
