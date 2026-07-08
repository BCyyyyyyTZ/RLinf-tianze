import torch

from rlinf.models.embodiment.dreamzero.ppo_utils import (
    dreamzero_action_gaussian_entropy,
    dreamzero_action_get_logprob_norm,
    normalize_action_payload,
)


def test_dreamzero_action_get_logprob_norm_matches_openpi_zero_sigma_semantics():
    sample = torch.tensor([[[0.0, 1.0, 9.0]]])
    mean = torch.tensor([[[0.0, 0.0, 0.0]]])
    sigma = torch.tensor([[[1.0, 1.0, 0.0]]])
    mask = torch.tensor([[[True, True, False]]])

    logprob = dreamzero_action_get_logprob_norm(sample, mean, sigma, mask=mask)

    expected = -0.5 * (
        sample[..., :2].pow(2) + torch.log(torch.tensor(2.0 * torch.pi))
    )
    assert logprob.shape == sample.shape
    assert torch.allclose(logprob[..., :2], expected)
    assert logprob[..., 2].item() == 0.0


def test_dreamzero_action_gaussian_entropy_keeps_per_dim_shape_and_masks_invalid_dims():
    sigma = torch.ones(2, 4, 3) * 0.5
    sigma[..., -1] = 0.0
    mask = torch.ones_like(sigma, dtype=torch.bool)
    mask[..., -1] = False

    entropy = dreamzero_action_gaussian_entropy(sigma, mask=mask)

    assert entropy.shape == sigma.shape
    assert torch.isfinite(entropy).all()
    assert torch.all(entropy[..., -1] == 0.0)


def test_normalize_action_payload_clamps_and_keeps_batch_time_shape():
    action = torch.tensor([[2.0, -2.0, 0.25], [0.0, 0.5, -0.5]])

    normalized = normalize_action_payload(action, action_dim=3)

    assert normalized.shape == (2, 1, 3)
    assert normalized.max() <= 1.0
    assert normalized.min() >= -1.0
