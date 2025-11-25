import torch

from utils.frequency import compute_pca_basis, project_onto_basis


def _pca_auxiliary_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    basis = compute_pca_basis(targets)
    outputs_proj = project_onto_basis(outputs, basis)
    targets_proj = project_onto_basis(targets, basis)
    diff = outputs_proj - targets_proj
    return (diff.abs() ** 2).mean()


def _build_identity_covariance_targets():
    # Six samples reshaped to [B, T, D] -> [2, 3, 3]
    # Covariance across the time dimension is diagonal with decreasing eigenvalues,
    # so the PCA basis should be the identity matrix ordered by variance magnitude.
    flatten_data = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
        ]
    )
    return flatten_data.reshape(2, 3, 3)


def test_compute_pca_basis_is_orthonormal_and_sorted():
    targets = _build_identity_covariance_targets()

    basis = compute_pca_basis(targets)

    # Orthonormal rows
    assert torch.allclose(basis @ basis.T, torch.eye(targets.shape[1]), atol=1e-5)
    # Largest-variance directions stay aligned to each time position (identity basis)
    assert torch.allclose(basis, torch.eye(targets.shape[1]), atol=1e-4)


def test_pca_auxiliary_loss_matches_mse_for_identity_basis():
    targets = _build_identity_covariance_targets()
    outputs = targets + 0.25

    aux_loss = _pca_auxiliary_loss(outputs, targets)
    baseline_loss = ((outputs - targets) ** 2).mean()

    assert torch.allclose(aux_loss, baseline_loss, atol=1e-6)


def test_pca_auxiliary_loss_preserves_gradient_flow_and_magnitude():
    targets = _build_identity_covariance_targets()

    outputs_pca = (targets + torch.tensor(0.1)).clone().detach().requires_grad_(True)
    outputs_mse = (targets + torch.tensor(0.1)).clone().detach().requires_grad_(True)

    pca_loss = _pca_auxiliary_loss(outputs_pca, targets)
    mse_loss = ((outputs_mse - targets) ** 2).mean()

    pca_loss.backward()
    mse_loss.backward()

    assert outputs_pca.grad is not None
    assert torch.allclose(outputs_pca.grad, outputs_mse.grad, atol=1e-6)
