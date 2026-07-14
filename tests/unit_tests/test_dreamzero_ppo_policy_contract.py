import torch
from torch import nn

from rlinf.models.embodiment.dreamzero.ppo_policy import DreamZeroPPOPolicyMixin
from rlinf.workers.actor.fsdp_actor_worker import (
    align_dreamzero_logprob_pair,
    get_dreamzero_loss_action_dim,
)


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
        self.velocity_calls = []
        self.sample_method_calls = []
        self.training_states = []

    def _predict_action_velocity(self, action_obs, x_t, timestep, *, use_velocity_only=False):
        self.velocity_calls.append((bool(use_velocity_only), timestep.detach().clone()))
        self.training_states.append(self.training)
        return torch.ones_like(x_t) * 0.25, torch.ones(
            x_t.shape[0], x_t.shape[-1], device=x_t.device
        )

    def dreamzero_action_sample_mean_var_val(self, **kwargs):
        self.sample_method_calls.append(kwargs["sample_method"])
        return super().dreamzero_action_sample_mean_var_val(**kwargs)

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

    assert out["logprobs"].shape == (4, 2, 2)
    assert out["values"].shape == (4, 1)
    assert out["entropy"].shape == (4, 2, 2)
    assert prev_logprobs.shape == (4, 2, 2)
    assert prev_values.shape == (4, 1)
    assert out["logprobs"].dtype == torch.float32
    assert out["values"].dtype == torch.float32
    assert not any(use_velocity_only for use_velocity_only, _ in model.velocity_calls)
    assert not any(model.training_states)


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


def test_dreamzero_loss_action_dim_matches_compacted_policy_dims():
    model_cfg = {
        "model_type": "dreamzero",
        "action_dim": 32,
        "env_action_dim": 7,
    }

    assert get_dreamzero_loss_action_dim(model_cfg, "actor_critic") == 7


def test_dreamzero_logprob_alignment_drops_padded_action_dims():
    logprobs = torch.ones(2, 16, 7)
    old_logprobs = torch.arange(2 * 16 * 32, dtype=torch.float32).reshape(2, 16, 32)

    aligned_logprobs, aligned_old_logprobs = align_dreamzero_logprob_pair(
        logprobs, old_logprobs
    )

    assert aligned_logprobs.shape == (2, 16, 7)
    assert aligned_old_logprobs.shape == (2, 16, 7)
    assert torch.equal(aligned_old_logprobs, old_logprobs[..., :7])


def test_rollout_and_replay_use_same_default_transition_kernel():
    torch.manual_seed(0)
    model = FakeDreamZeroPPOPolicy()
    action_obs = {"state": torch.zeros(2, 5)}
    chains, denoise_inds, prev_logprobs, _ = model._sample_action_chain(
        action_obs=action_obs,
        initial_noise=torch.zeros(2, 2, 3),
        mode="train",
        compute_values=True,
    )

    rollout_calls = list(model.velocity_calls)
    assert len(rollout_calls) == model.config.num_steps
    assert not any(use_velocity_only for use_velocity_only, _ in rollout_calls)
    selected_idx = int(denoise_inds[0, 0].item())
    assert torch.equal(denoise_inds, torch.full_like(denoise_inds, selected_idx))
    assert model.sample_method_calls == [
        "flow_sde" if idx == selected_idx else "flow_ode"
        for idx in range(model.config.num_steps)
    ]

    model.velocity_calls.clear()
    model.sample_method_calls.clear()
    out = model.ppo_forward(
        forward_inputs=model._build_ppo_forward_inputs(
            action_obs=action_obs,
            chains=chains,
            denoise_inds=denoise_inds,
            action_mask=model._build_action_mask(chains[:, -1]),
            model_action=chains[:, -1],
            env_action=chains[:, -1, :, :2],
        ),
        compute_logprobs=True,
        compute_values=True,
    )

    assert len(model.velocity_calls) == 1
    assert model.velocity_calls[0][0] is False
    expected_timestep = model._get_dreamzero_timesteps(
        model.config.num_steps, chains.device
    )[denoise_inds[:, 0]]
    assert torch.equal(model.velocity_calls[0][1], expected_timestep)
    assert model.sample_method_calls == ["flow_sde"]
    assert torch.allclose(out["logprobs"], prev_logprobs)
    assert model.training is True
    assert not any(model.training_states)


def test_rollout_and_replay_can_use_configured_velocity_only_kernel():
    torch.manual_seed(0)
    model = FakeDreamZeroPPOPolicy()
    model.config.ppo_use_velocity_only = True
    action_obs = {"state": torch.zeros(2, 5)}
    chains, denoise_inds, prev_logprobs, _ = model._sample_action_chain(
        action_obs=action_obs,
        initial_noise=torch.zeros(2, 2, 3),
        mode="train",
        compute_values=True,
    )

    assert all(use_velocity_only for use_velocity_only, _ in model.velocity_calls)

    model.velocity_calls.clear()
    out = model.ppo_forward(
        forward_inputs=model._build_ppo_forward_inputs(
            action_obs=action_obs,
            chains=chains,
            denoise_inds=denoise_inds,
            action_mask=model._build_action_mask(chains[:, -1]),
            model_action=chains[:, -1],
            env_action=chains[:, -1, :, :2],
        ),
        compute_logprobs=True,
        compute_values=True,
    )

    assert len(model.velocity_calls) == 1
    assert model.velocity_calls[0][0] is True
    assert torch.allclose(out["logprobs"], prev_logprobs)


def test_ppo_forward_restores_training_mode_after_deterministic_replay():
    model = FakeDreamZeroPPOPolicy()
    model.train()
    action_obs = {"state": torch.zeros(2, 5)}
    chains, denoise_inds, _, _ = model._sample_action_chain(
        action_obs=action_obs,
        initial_noise=torch.zeros(2, 2, 3),
        mode="train",
        compute_values=True,
    )
    assert model.training is True

    model.training_states.clear()
    model.ppo_forward(
        forward_inputs=model._build_ppo_forward_inputs(
            action_obs=action_obs,
            chains=chains,
            denoise_inds=denoise_inds,
            action_mask=model._build_action_mask(chains[:, -1]),
            model_action=chains[:, -1],
            env_action=chains[:, -1, :, :2],
        ),
        compute_logprobs=True,
        compute_values=True,
    )

    assert model.training is True
    assert model.training_states == [False]


def test_eval_rollout_uses_ode_chain_like_openpi_inference():
    model = FakeDreamZeroPPOPolicy()
    action_obs = {"state": torch.zeros(2, 5)}
    model._sample_action_chain(
        action_obs=action_obs,
        initial_noise=torch.zeros(2, 2, 3),
        mode="eval",
        compute_values=False,
    )

    assert model.sample_method_calls == ["flow_ode"] * model.config.num_steps


def test_predict_action_velocity_prefers_velocity_only_interface():
    class FakeActionHead:
        def __init__(self):
            self.calls = 0

        def predict_action_velocity_only(self, action_obs, x_t, timestep):
            self.calls += 1
            assert action_obs["state"].device == x_t.device
            assert timestep.dtype == torch.float32
            return {
                "velocity": torch.full_like(x_t, 0.5),
                "value_feature": torch.ones(x_t.shape[0], x_t.shape[-1], device=x_t.device),
            }

    class FakePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.action_head = FakeActionHead()

        def _action_obs_to_device(self, action_obs):
            device = next(self.parameters()).device
            return {
                key: value.to(device=device) if torch.is_tensor(value) else value
                for key, value in action_obs.items()
            }

        def _predict_action_velocity(self, action_obs, x_t, timestep):
            action_obs = self._action_obs_to_device(action_obs)
            x_t = x_t.to(device=next(self.parameters()).device, dtype=torch.float32)
            timestep = timestep.to(device=x_t.device, dtype=torch.float32)
            output = self.action_head.predict_action_velocity_only(
                action_obs, x_t, timestep
            )
            velocity = output["velocity"]
            value_feature = output.get("value_feature", velocity.float().mean(dim=1))
            return velocity.float(), value_feature.float()

        def lazy_joint_video_action_causal(self, obs):
            raise AssertionError("velocity-only path must not call lazy_joint_video_action_causal")

    model = FakePolicy()
    x_t = torch.zeros(2, 3, 4)
    velocity, value_feature = model._predict_action_velocity(
        {"state": torch.zeros(2, 1, 4)},
        x_t,
        torch.tensor([1.0, 0.5]),
    )

    assert model.action_head.calls == 1
    assert torch.allclose(velocity, torch.full_like(x_t, 0.5))
    assert value_feature.shape == (2, 4)
