import torch


def compute_pca_basis(targets: torch.Tensor) -> torch.Tensor:
    """Compute an orthonormal PCA basis over the time dimension.

    Args:
        targets: Tensor shaped [B, T, D].

    Returns:
        Tensor shaped [T, T] representing the orthonormal basis (rows).
    """
    # Flatten batch and channel dimensions: [B * D, T]
    targets_flat = targets.reshape(-1, targets.shape[1])
    targets_mean = targets_flat.mean(dim=0, keepdim=True)
    targets_centered = targets_flat - targets_mean

    covariance = (targets_centered.transpose(0, 1) @ targets_centered) / targets_centered.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    idx = eigvals.argsort(descending=True)
    eigvecs = eigvecs[:, idx]

    return eigvecs.transpose(0, 1)


def project_onto_basis(data: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project sequences onto an orthonormal basis.

    Args:
        data: Tensor shaped [B, T, D].
        basis: Tensor shaped [T, T].

    Returns:
        Tensor shaped [B, D, T] containing projected sequences.
    """
    B, T, D = data.shape
    data_flat = data.reshape(-1, T)
    projected = torch.matmul(basis, data_flat.permute(1, 0)).reshape(T, B, D)
    return projected.permute(1, 2, 0)
