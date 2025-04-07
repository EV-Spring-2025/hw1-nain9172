import torch


def compute_mean_knn_dist(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Computes the mean k-nearest neighbor (k-NN) distance for each point.
    This distance is used to initialize the scale of the Gaussian,
    as suggested in the paper (using the mean distance of the 3 nearest neighbors).

    Args:
        points (torch.Tensor): Tensor of shape (N, D).
        k (int): Number of nearest neighbors.

    Returns:
        torch.Tensor: Mean distance to the k nearest neighbors for each point.
    """
    # Compute pairwise squared distances.
    diffs = points[:, None, :] - points[None, :, :]
    dists_sq = torch.sum(diffs ** 2, dim=-1)

    # Set self-distances to infinity so they are ignored.
    dists_sq.fill_diagonal_(float('inf'))

    # Find the k smallest distances for each point.
    knn_vals, _ = torch.topk(dists_sq, k, largest=False)
    mean_knn_dist = knn_vals.mean(dim=1)
    return mean_knn_dist