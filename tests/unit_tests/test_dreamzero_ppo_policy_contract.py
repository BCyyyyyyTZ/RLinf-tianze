import torch
from torch import nn

from rlinf.models.embodiment.dreamzero.ppo_policy import DreamZeroPPOPolicyMixin


class FakeDreamZeroPPOPolicy(DreamZeroPPOPolicyMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "action_dim": 3,
                "env_action_dim": 2,
                "action_horizon": 2,
                "num_steps": 3,
                "noise_method": "flow_sde",
                "safe_get_logprob": False,
                "joint_logprob": False,
                "noise_level": 0.5,
            },
        )()
        self._setup_ppo_heads(value_input_dim=3)

    def _predict_action_velocity(self, action_obs, x_t, timestep):
        return torch.ones_like(x_t) * 0.25, torch.ones(
            x_t.shape[0], x_t.shape[-1], device=x_t.device
        )

    def _build_action_mask(self, action):
        mask = torch.ones_like(action, dtype=torch.bool)
        mask[..., -1] = False
        return mask


def test_sample_mean_var_val_returns_openpi_style_tensors():
    model = FakeDreamZeroPPOPolicy()
    x_t = torch.zeros(4, 2, 3)

    mean, std, value, velocity = model.dreamzero_action_sample_mean_var_val(
        x_t=x_t,
        idx=torch.zeros(4, dtype=torch.long),
        action_obs={"state": torch.zeros(4, 5)},
        sample_method="flow_sde",
        denoise_steps=3,
        compute_values=True,
    )

    assert mean.shape == x_t.shape
    assert std.shape == x_t.shape
    assert value.shape == (4,)
    assert velocity.shape == x_t.shape


def test_sample_mean_var_val_matches_bfloat16_value_head_dtype():
    model = FakeDreamZeroPPOPolicy().to(dtype=torch.bfloat16)
    x_t = torch.zeros(4, 2, 3, dtype=torch.float32)

    _, _, value, _ = model.dreamzero_action_sample_mean_var_val(
        x_t=x_t,
        idx=torch.zeros(4, dtype=torch.long),
        action_obs={"state": torch.zeros(4, 5)},
        sample_method="flow_sde",
        denoise_steps=3,
        compute_values=True,
    )

    assert value.shape == (4,)
    assert value.dtype == torch.float32


def test_ppo_forward_replays_chains_and_denoise_indices():
    model = FakeDreamZeroPPOPolicy()
    action_obs = {"state": torch.zeros(4, 5)}
    chains, denoise_inds, prev_logprobs, prev_values = model._sample_action_chain(
        action_obs=action_obs,
        initial_noise=torch.zeros(4, 2, 3),
        mode="train",
        compute_values=True,
    )
    forward_inputs = model._build_ppo_forward_inputs(
        action_obs=action_obs,
        chains=chains,
        denoise_inds=denoise_inds,
        action_mask=model._build_action_mask(chains[:, -1]),
        model_action=chains[:, -1],
        env_action=chains[:, -1, :, :2],
    )

    out = model.ppo_forward(
        forward_inputs=forward_inputs,
        compute_logprobs=True,
        compute_values=True,
        compute_entropy=True,
    )

    assert out["logprobs"].shape == (4, 2, 3)
    assert out["values"].shape == (4, 1)
    assert out["entropy"].shape == (4, 2, 3)
    assert prev_logprobs.shape == (4, 2, 3)
    assert prev_values.shape == (4, 1)
    assert out["logprobs"].dtype == torch.float32
    assert out["values"].dtype == torch.float32


def test_async_proximal_forward_contract_returns_logprobs_only():
    model = FakeDreamZeroPPOPolicy()
    action_obs = {"state": torch.zeros(5, 5)}
    chains, denoise_inds, _, _ = model._sample_action_chain(
        action_obs=action_obs,
        initial_noise=torch.zeros(5, 2, 3),
        mode="train",
        compute_values=False,
    )
    forward_inputs = model._build_ppo_forward_inputs(
        action_obs=action_obs,
        chains=chains,
        denoise_inds=denoise_inds,
        action_mask=torch.ones(5, 2, 3, dtype=torch.bool),
        model_action=chains[:, -1],
        env_action=chains[:, -1, :, :2],
    )

    out = model.ppo_forward(
        forward_inputs=forward_inputs,
        compute_logprobs=True,
        compute_values=False,
        compute_entropy=False,
    )

    assert set(out.keys()) == {"logprobs"}
    assert out["logprobs"].shape == (5, 2, 3)
