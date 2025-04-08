import random
from typing import BinaryIO, Dict, List, Optional, Union

import numpy as np
import torch

from  .camera import extract_camera_params


def get_point_clouds(cameras, depths, alphas, rgbs=None):
    """
    Generates a 3D point cloud from camera parameters, depth maps, and optional RGB colors.

    Args:
        cameras: Camera intrinsics and extrinsics.
        depths: Depth maps of shape (N, H, W), where N is the number of images.A
        alphas: Binary mask indicating valid depth points.
        rgbs: Optional RGB color values corresponding to depth points.

    Returns:
        PointCloud: A structured point cloud representation with 3D coordinates and color information.
    """
    Hs, Ws, intrinsics, c2ws = extract_camera_params(cameras)
    W, H = int(Ws[0].item()), int(Hs[0].item())
    assert (depths.shape == alphas.shape)
    coords = []
    rgbas = []

    # TODO: Compute ray origins and directions for each pixel
    for i in range(cameras.shape[0]):
        # Get current image's camera parameters
        intrinsic = intrinsics[i]  # (4, 4)
        c2w = c2ws[i]              # (4, 4)
        # Create meshgrid for pixel coordinates (using pixel centers)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=depths.device, dtype=torch.float32),
            torch.arange(W, device=depths.device, dtype=torch.float32),
            indexing="ij"
        )
        # Extract intrinsic parameters (upper-left 3x3) and compute normalized image coordinates
        K = intrinsic[:3, :3]
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        x_cam = (grid_x - cx) / fx
        y_cam = (grid_y - cy) / fy
        ones = torch.ones_like(x_cam)
        # Form ray directions in camera space and normalize
        ray_dirs_cam = torch.stack([x_cam, y_cam, ones], dim=-1)  # (H, W, 3)
        ray_dirs_cam = ray_dirs_cam / torch.norm(ray_dirs_cam, dim=-1, keepdim=True)
        # Transform ray directions to world space using the rotation part of c2w
        R = c2w[:3, :3]
        ray_dirs_world = torch.einsum('ij,hwj->hwi', R, ray_dirs_cam)
        ray_dirs_world = ray_dirs_world / torch.norm(ray_dirs_world, dim=-1, keepdim=True)
        # The ray origin is the camera center (translation part of c2w)
        ray_origin = c2w[:3, 3]
        rays_o = ray_origin.expand(H, W, 3)  # (H, W, 3)
        rays_d = ray_dirs_world           # (H, W, 3)

        # TODO: Compute 3D world coordinates using depth values
        # Use the ray equation: P = O + D * depth
        depth_img = depths[i]  # (H, W)
        pts = rays_o + rays_d * depth_img.unsqueeze(-1)  # (H, W, 3)

        # TODO: Apply the alpha mask to filter valid points
        # Mask valid points (and RGB if provided) using a threshold on alphas
        mask = alphas[i] > 0.5  # (H, W)
        valid_pts = pts[mask]   # (N_valid, 3)
        coords.append(valid_pts)
        if rgbs is not None:
            rgb_img = rgbs[i]  # (H, W, 3)
            valid_rgba = torch.cat([rgb_img[mask], alphas[i][mask].unsqueeze(-1)], dim=-1)  # (N_valid, 4)
            rgbas.append(valid_rgba)

    if rgbs is not None and len(rgbas) > 0:
        coords = torch.cat(coords, dim=0).cpu().numpy()
        rgbas = torch.cat(rgbas, dim=0)
    else:
        coords = torch.cat(coords, dim=0).cpu().numpy()

    if rgbs is not None:
        channels = dict(
            R=rgbas[..., 0].cpu().numpy(),
            G=rgbas[..., 1].cpu().numpy(),
            B=rgbas[..., 2].cpu().numpy(),
            A=rgbas[..., 3].cpu().numpy(),
        )
    else:
        channels = {}

    point_cloud = PointCloud(coords, channels)
    return point_cloud




def preprocess(data, channel):
    if channel in ["R", "G", "B", "A"]:
        return np.round(data * 255.0)
    return data


class PointCloud:
    def __init__(self, coords: np.ndarray, channels: Dict[str, np.ndarray]) -> None:
        self.coords = coords
        self.channels = channels

    def __repr__(self) -> str:
        str = f"coords:{len(self.coords)} \t channels:{list(self.channels.keys())}"
        return str

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return PointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"},
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with open(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx
            cur_dists = np.minimum(cur_dists, compute_dists(idx))
        return self.subsample(indices, **subsample_kwargs)

    def subsample(self, indices: np.ndarray, average_neighbors: bool = False) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
            )

        new_coords = self.coords[indices]
        neighbor_indices = PointCloud(coords=new_coords, channels={}).nearest_points(self.coords)

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(coords=new_coords, channels=new_channels)

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack([preprocess(self.channels[name], name) for name in channel_names], axis=-1)
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            dists = norms + np.sum(batch**2, axis=-1)[:, None] - 2 * (batch @ self.coords.T)
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0) for k, v in self.channels.items()
            },
        )
