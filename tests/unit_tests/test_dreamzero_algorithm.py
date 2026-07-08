# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for DreamZero algorithm registry integration."""

import asyncio
import importlib
import types

import pytest
import torch
from omegaconf import OmegaConf

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.models.embodiment.dreamzero.world_model import DreamZeroWorldModel
from rlinf.workers.actor.fsdp_actor_worker import (
    build_dreamzero_forward_inputs,
    get_dreamzero_train_rollout_size,
    get_dreamzero_loss_action_dim,
    process_nested_dict_for_train,
    should_log_actor_training_progress,
)


def test_dreamzero_action_head_rl_loss_type_is_removed():
    with pytest.raises(ValueError, match="not registered"):
        policy_loss(
            task_type="embodied",
            loss_type="dreamzero_action_head_rl",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            single_action_dim=3,
            logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
            old_logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
            advantages=torch.ones(1, 1, dtype=torch.float32),
        )


def test_dreamzero_world_model_proxy_loss_type_is_removed():
    with pytest.raises(ValueError, match="not registered"):
        policy_loss(
            task_type="embodied",
            loss_type="dreamzero_world_model_proxy",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            single_action_dim=3,
            logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
            old_logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
            advantages=torch.ones(1, 1, dtype=torch.float32),
            dreamzero_losses={
                "model_loss": torch.tensor(1.0),
                "actor_loss": torch.tensor(1.0),
                "value_loss": torch.tensor(1.0),
            },
        )


def test_dreamzero_standard_ppo_uses_action_dim_for_loss_reshape():
    model_cfg = {
        "model_type": "dreamzero",
        "action_dim": 32,
        "env_action_dim": 7,
    }

    assert get_dreamzero_loss_action_dim(model_cfg, loss_type="actor_critic") == 32


def test_standard_dreamzero_ppo_uses_action_dim_for_actor_critic():
    model_cfg = OmegaConf.create(
        {
            "model_type": "dreamzero",
            "action_dim": 32,
            "env_action_dim": 7,
        }
    )

    assert get_dreamzero_loss_action_dim(model_cfg, "actor_critic") == 32
    assert get_dreamzero_loss_action_dim(model_cfg, "decoupled_actor_critic") == 32


def test_dreamzero_legacy_loss_type_is_not_a_world_model_proxy_alias():
    with pytest.raises(ValueError, match="not registered"):
        policy_loss(
            task_type="embodied",
            loss_type="dreamzero",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            single_action_dim=3,
            logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
            old_logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
            advantages=torch.ones(1, 1, dtype=torch.float32),
            dreamzero_losses={
                "model_loss": torch.tensor(1.0),
                "actor_loss": torch.tensor(0.5),
                "value_loss": torch.tensor(0.25),
            },
        )


def test_dreamzero_world_model_forward_uses_batch_time_layout():
    torch.manual_seed(0)
    model = DreamZeroWorldModel(
        obs_dim=6,
        action_dim=4,
        stochastic_dim=5,
        deterministic_dim=7,
        hidden_dim=16,
        imagination_horizon=3,
    )
    batch_size = 2
    time_steps = 4
    curr_obs = {"states": torch.randn(batch_size, time_steps, 6)}
    next_obs = {"states": torch.randn(batch_size, time_steps, 6)}
    actions = torch.randn(batch_size, time_steps, 4)
    rewards = torch.randn(batch_size, time_steps, 1)
    dones = torch.zeros(batch_size, time_steps, 1, dtype=torch.bool)

    outputs = model(
        curr_obs=curr_obs,
        next_obs=next_obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )

    assert set(outputs["losses"]) == {"model_loss", "actor_loss", "value_loss"}
    assert outputs["posterior_features"].shape == (batch_size, time_steps, 12)
    assert outputs["reconstructions"].shape == (batch_size, time_steps, 6)
    assert outputs["imagined_features"].shape == (batch_size, 3, 12)

    total_loss = sum(outputs["losses"].values())
    total_loss.backward()
    assert model.encoder.net[0].weight.grad is not None
    assert model.actor.net[0].weight.grad is not None


def test_dreamzero_world_model_forward_accepts_single_step_vectors():
    torch.manual_seed(0)
    model = DreamZeroWorldModel(
        obs_dim=112,
        action_dim=16,
        stochastic_dim=5,
        deterministic_dim=7,
        hidden_dim=16,
        imagination_horizon=3,
    )

    outputs = model(
        curr_obs={"states": torch.randn(1, 112)},
        next_obs={"states": torch.randn(1, 112)},
        actions=torch.randn(1, 16),
        rewards=torch.randn(1, 1),
        dones=torch.zeros(1, 1, dtype=torch.bool),
    )

    assert outputs["posterior_features"].shape == (1, 1, 12)
    assert outputs["reconstructions"].shape == (1, 1, 112)


def test_dreamzero_default_forward_rejects_action_head_rl_payload():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)

    policy = FakeDreamZeroPolicy()
    with pytest.raises(KeyError, match="OpenPI-style PPO"):
        policy.default_forward(
            forward_inputs={
                "dreamzero_rl.action": torch.ones(2, 4, 7),
            }
        )


def test_dreamzero_policy_syncs_action_head_device_helpers():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeActionHead:
        _device = "meta"
        _vae_device_ready = True

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.action_head = FakeActionHead()

    policy = FakeDreamZeroPolicy()
    policy._sync_action_head_device()

    assert policy.action_head._device == str(policy.weight.device)
    assert policy.action_head._vae_device_ready is False
    assert policy.action_head.trt_engine is None
    assert policy.action_head.trt_context is None


def test_dreamzero_wan_forward_blocks_skips_video_head_for_action_only():
    patch_module = importlib.import_module(
        "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_inference"
    )

    class FakeBlock:
        def __call__(self, *, x, **kwargs):
            return x + 1.0, torch.ones(1)

    class FakeWanModel:
        def __init__(self):
            self.dim = 4
            self.freq_dim = 4
            self.freqs_action = torch.zeros(1)
            self.freqs_state = torch.zeros(1)
            self.gradient_checkpointing = False
            self.blocks = [FakeBlock()]
            self.head_calls = 0

        def action_encoder(self, action, timestep_action, embodiment_id):
            del timestep_action, embodiment_id
            return torch.ones(action.shape[0], action.shape[1], self.dim)

        def state_encoder(self, state, embodiment_id):
            del embodiment_id
            return torch.ones(state.shape[0], state.shape[1], self.dim)

        def action_decoder(self, action_tokens, embodiment_id):
            del embodiment_id
            return action_tokens

        def time_embedding(self, timestep_embedding):
            return torch.zeros(
                timestep_embedding.shape[0],
                self.dim,
                dtype=timestep_embedding.dtype,
                device=timestep_embedding.device,
            )

        def time_projection(self, embeddings):
            return torch.zeros(
                *embeddings.shape[:-1],
                6 * self.dim,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )

        def text_embedding(self, context):
            return context

        def head(self, x_video, e_video):
            del e_video
            self.head_calls += 1
            return x_video

    model = FakeWanModel()
    x_video, action_noise_pred, updated_kv_caches = patch_module._forward_blocks(
        model,
        x=torch.zeros(1, 4, 1, 1, 2),
        seq_len=2,
        freqs=torch.zeros(1),
        timestep=torch.zeros(1, 1, dtype=torch.int64),
        context=torch.zeros(1, 2, 4),
        clip_feature=None,
        embodiment_id=torch.zeros(1, dtype=torch.long),
        action=torch.zeros(1, 1, 3),
        timestep_action=torch.zeros(1, 1, dtype=torch.int64),
        state=torch.zeros(1, 1, 3),
        kv_cache=[torch.zeros(1)],
        current_start_frame=0,
        return_video_pred=False,
    )

    assert x_video is None
    assert model.head_calls == 0
    assert action_noise_pred is not None
    assert len(updated_kv_caches) == 1


def test_dreamzero_wan_forward_inference_skips_unpatchify_for_action_only():
    patch_module = importlib.import_module(
        "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_inference"
    )

    class FakeWanModel:
        model_type = "t2v"
        concat_first_frame_latent = False
        text_len = 2

        def __init__(self):
            self.unpatchify_calls = 0
            self._forward_blocks = types.MethodType(patch_module._forward_blocks, self)

        def patch_embedding(self, x):
            return x

        def _create_freqs(self, *, grid_size, start_frame):
            del grid_size, start_frame
            return torch.zeros(1)

        def unpatchify(self, x_video, grid_size):
            del x_video, grid_size
            self.unpatchify_calls += 1
            raise AssertionError("unpatchify should be skipped in action-only inference")

        dim = 4
        freq_dim = 4
        freqs_action = torch.zeros(1)
        freqs_state = torch.zeros(1)
        gradient_checkpointing = False
        blocks = []

        def action_encoder(self, action, timestep_action, embodiment_id):
            del timestep_action, embodiment_id
            return torch.ones(action.shape[0], action.shape[1], self.dim)

        def state_encoder(self, state, embodiment_id):
            del embodiment_id
            return torch.ones(state.shape[0], state.shape[1], self.dim)

        def action_decoder(self, action_tokens, embodiment_id):
            del embodiment_id
            return action_tokens

        def time_embedding(self, timestep_embedding):
            return torch.zeros(
                timestep_embedding.shape[0],
                self.dim,
                dtype=timestep_embedding.dtype,
                device=timestep_embedding.device,
            )

        def time_projection(self, embeddings):
            return torch.zeros(
                *embeddings.shape[:-1],
                6 * self.dim,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )

        def text_embedding(self, context):
            return context

        def head(self, x_video, e_video):
            del e_video
            return x_video

    model = FakeWanModel()
    video_noise_pred, action_noise_pred, updated_kv_caches = patch_module._forward_inference(
        model,
        x=torch.zeros(1, 4, 1, 1, 2),
        timestep=torch.zeros(1, 1, dtype=torch.int64),
        context=torch.zeros(1, 2, 4),
        seq_len=2,
        kv_cache=[],
        crossattn_cache=[],
        current_start_frame=0,
        y=None,
        clip_feature=None,
        action=torch.zeros(1, 1, 3),
        timestep_action=torch.zeros(1, 1, dtype=torch.int64),
        state=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
        return_video_pred=False,
    )

    assert video_noise_pred is None
    assert model.unpatchify_calls == 0
    assert action_noise_pred is not None
    assert updated_kv_caches == []


def test_dreamzero_run_diffusion_steps_passes_action_only_flag_to_model():
    patch_module = importlib.import_module(
        "rlinf.models.embodiment.dreamzero.patch.wan_policy_head_action_only"
    )

    class FakePolicyHead:
        trt_engine = None
        ip_size = 1

        def __init__(self):
            self.seen_return_video_flags = []

        def model(self, *args, **kwargs):
            del args
            self.seen_return_video_flags.append(kwargs["return_video_pred"])
            return None, kwargs["action"] + 1.0, [torch.ones(1)]

        def _exchange_predictions(self, predictions):
            return predictions

    head = FakePolicyHead()
    with patch_module._action_only_video_context(enabled=True):
        predictions = patch_module._run_diffusion_steps(
            head,
            noisy_input=torch.zeros(1, 1, 1, 1, 2),
            timestep=torch.zeros(1, 1, dtype=torch.int64),
            action=torch.zeros(1, 2, 3),
            timestep_action=torch.zeros(1, 2, dtype=torch.int64),
            state=torch.zeros(1, 1, 3),
            embodiment_id=torch.zeros(1, dtype=torch.long),
            context=[torch.zeros(1, 2, 4)],
            seq_len=2,
            y=torch.zeros(1, 1, 1, 1, 2),
            clip_feature=torch.zeros(1, 4),
            kv_caches=[[torch.zeros(1)]],
            crossattn_caches=[[torch.zeros(1)]],
            kv_cache_metadata={"start_frame": 0, "update_kv_cache": False},
        )

    assert head.seen_return_video_flags == [False]
    assert torch.equal(predictions[0][0], torch.zeros(1, 1, 1, 1, 2))
    assert torch.equal(predictions[0][1], torch.ones(1, 2, 3))


def test_dreamzero_action_only_video_scheduler_step_is_noop():
    patch_module = importlib.import_module(
        "rlinf.models.embodiment.dreamzero.patch.wan_policy_head_action_only"
    )
    called = {"value": False}

    def original_step(self, *, model_output, timestep, sample, step_index, return_dict=True):
        del self, model_output, timestep, sample, step_index, return_dict
        called["value"] = True
        raise AssertionError("video scheduler step should be skipped in action-only mode")

    wrapped_step = patch_module.flow_unipc_step(original_step)
    sample = torch.zeros(1, 1, 1, 1, 2)

    with patch_module._action_only_video_context(enabled=True):
        result = wrapped_step(
            object(),
            model_output=torch.ones_like(sample),
            timestep=torch.tensor(0),
            sample=sample,
            step_index=0,
            return_dict=False,
        )

    assert called["value"] is False
    assert result == (sample,)


def test_dreamzero_get_model_registers_action_only_video_postprocess_patches(monkeypatch):
    dreamzero_module = importlib.import_module("rlinf.models.embodiment.dreamzero")
    patcher_module = importlib.import_module("rlinf.utils.patcher")
    original_apply = patcher_module.Patcher.apply

    def stop_after_patch_registration():
        raise RuntimeError("stop after patch registration")

    monkeypatch.setattr(patcher_module.Patcher, "apply", stop_after_patch_registration)

    with pytest.raises(RuntimeError, match="stop after patch registration"):
        dreamzero_module.get_model(OmegaConf.create({"model_path": "/tmp/not-needed"}))

    mappings = patcher_module.Patcher._mappings_dict
    wrappers = patcher_module.Patcher._wrappers_dict
    assert (
        mappings[
            "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel._forward_inference"
        ]
        == "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_inference._forward_inference"
    )
    assert (
        mappings[
            "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead._run_diffusion_steps"
        ]
        == "rlinf.models.embodiment.dreamzero.patch.wan_policy_head_action_only._run_diffusion_steps"
    )
    assert (
        "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead.lazy_joint_video_action"
        in wrappers
    )
    assert (
        "groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler.FlowUniPCMultistepScheduler.step"
        in wrappers
    )

    patcher_module.Patcher.apply = original_apply
    patcher_module.Patcher.clear()


def test_rollout_generate_releases_dreamzero_inference_cache():
    rollout_module = pytest.importorskip(
        "rlinf.workers.rollout.hf.huggingface_worker"
    )
    MultiStepRolloutWorker = rollout_module.MultiStepRolloutWorker

    class FakeModel:
        def __init__(self):
            self.release_count = 0

        def release_inference_cache(self):
            self.release_count += 1

    class FakeWorker(MultiStepRolloutWorker):
        def __init__(self):
            self.enable_offload = False
            self.rollout_epoch = 2
            self._rank = 0
            self.hf_model = FakeModel()
            self.generated = 0
            self.empty_cache_count = 0

            class FakeTorchPlatform:
                def __init__(inner_self, outer):
                    inner_self.outer = outer

                def empty_cache(inner_self):
                    inner_self.outer.empty_cache_count += 1

            self.torch_platform = FakeTorchPlatform(self)

        async def generate_one_epoch(self, input_channel, output_channel):
            self.generated += 1

    worker = FakeWorker()
    asyncio.run(worker.generate(input_channel=None, output_channel=None))

    assert worker.generated == 2
    assert worker.hf_model.release_count == 1
    assert worker.empty_cache_count == 1


def test_rollout_sync_filters_dreamzero_lora_receiver_state_dict():
    rollout_module = pytest.importorskip(
        "rlinf.workers.rollout.hf.huggingface_worker"
    )
    MultiStepRolloutWorker = rollout_module.MultiStepRolloutWorker

    class FakeModel:
        def state_dict(self):
            return {
                "action_head.model.base_model.model.blocks.0.attn.q.lora_A.default.weight": torch.ones(1),
                "action_head.model.base_model.model.action_decoder.weight": torch.ones(1),
                "action_head.model.base_model.model.action_encoder.weight": torch.ones(1),
                "action_head.model.base_model.model.state_encoder.weight": torch.ones(1),
                "action_head.model.base_model.model.blocks.0.attn.q.base_layer.weight": torch.ones(1),
            }

        def set_global_step(self, step):
            self.global_step = step

    class FakeWeightSyncer:
        def __init__(self):
            self.receiver_state_keys = None

        def receiver_initialized(self):
            return False

        async def init_receiver(self, *, state_dict, recv, send):
            del recv, send
            self.receiver_state_keys = set(state_dict)

        async def apply(self, model, recv):
            del model, recv
            return 0

    class FakeWorker(MultiStepRolloutWorker):
        def __init__(self):
            self.cfg = OmegaConf.create(
                {
                    "actor": {
                        "model": {"model_type": "dreamzero", "is_lora": True},
                    },
                }
            )
            self.actor_group_name = "ActorGroup"
            self.actor_weight_src_rank = 0
            self._group_name = "RolloutGroup"
            self._weight_sync_rollout_ranks = [0]
            self._weight_sync_is_sender = False
            self._sync_weight_comm_options = None
            self.weight_syncer = FakeWeightSyncer()
            self.hf_model = FakeModel()
            self.finished_episodes = 0

            class FakeTorchPlatform:
                def empty_cache(inner_self):
                    return None

            self.torch_platform = FakeTorchPlatform()

        def broadcast(self, *args, **kwargs):
            raise AssertionError("syncer should not call recv in this unit test")

    worker = FakeWorker()
    asyncio.run(worker.sync_model_from_actor())

    assert worker.weight_syncer.receiver_state_keys == {
        "action_head.model.base_model.model.blocks.0.attn.q.lora_A.default.weight",
        "action_head.model.base_model.model.action_decoder.weight",
        "action_head.model.base_model.model.action_encoder.weight",
        "action_head.model.base_model.model.state_encoder.weight",
    }


def test_actor_rollout_state_dict_filters_dreamzero_lora_sync_names():
    actor_module = pytest.importorskip("rlinf.workers.actor.fsdp_actor_worker")
    EmbodiedFSDPActor = actor_module.EmbodiedFSDPActor

    class FakeActor(EmbodiedFSDPActor):
        def __init__(self):
            self.cfg = OmegaConf.create(
                {
                    "actor": {
                        "model": {"model_type": "dreamzero", "is_lora": True},
                    },
                }
            )
            self.param_names_need_sync = [
                "action_head.model.base_model.model.blocks.0.attn.q.lora_A.default.weight",
                "action_head.model.base_model.model.blocks.0.attn.q.base_layer.weight",
                "action_head.model.base_model.model.action_decoder.weight",
                "action_head.model.base_model.model.action_encoder.weight",
                "action_head.model.base_model.model.state_encoder.weight",
                "world_model.rssm.weight",
            ]

        def get_model_state_dict(self, *, cpu_offload, full_state_dict):
            assert cpu_offload is False
            assert full_state_dict is False
            return {
                key: torch.ones(1)
                for key in self.param_names_need_sync
            }

    actor = FakeActor()
    state_dict = actor.get_rollout_state_dict()

    assert set(state_dict) == {
        "action_head.model.base_model.model.blocks.0.attn.q.lora_A.default.weight",
        "action_head.model.base_model.model.action_decoder.weight",
        "action_head.model.base_model.model.action_encoder.weight",
        "action_head.model.base_model.model.state_encoder.weight",
    }
    assert actor.param_names_need_sync == list(state_dict.keys())


def test_actor_training_progress_logging_samples_long_runs():
    logged = [
        idx for idx in range(64) if should_log_actor_training_progress(idx, total=64)
    ]

    assert logged[0] == 0
    assert logged[-1] == 63
    assert len(logged) <= 18
    assert 3 in logged


def test_dreamzero_advantage_uses_unified_registry_entry():
    rewards = torch.tensor(
        [
            [[1.0], [0.0]],
            [[0.5], [0.25]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros(3, 2, 1, dtype=torch.bool)
    dones[-1] = True
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="dreamzero",
        rewards=rewards,
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
        group_size=1,
        reward_type="chunk_level",
        loss_mask=loss_mask,
    )

    assert set(result) == {"advantages", "returns"}
    assert result["advantages"].shape == rewards.shape
    assert result["returns"].shape == rewards.shape
    assert torch.all(result["advantages"][loss_mask] >= 0)


def test_dreamzero_loader_disables_torch_compile_from_env(monkeypatch):
    dreamzero_model_module = pytest.importorskip("rlinf.models.embodiment.dreamzero")

    monkeypatch.setenv("DREAMZERO_DISABLE_TORCH_COMPILE", "1")

    def fn(x):
        return x

    original_compile = torch.compile
    try:
        dreamzero_model_module._disable_torch_compile_for_dreamzero()
        assert torch.compile(fn) is fn
        assert torch.compile()(fn) is fn
        assert dreamzero_model_module._dreamzero_disable_torch_compile() is True
    finally:
        monkeypatch.setattr(torch, "compile", original_compile)


def test_dreamzero_wan_policy_head_uses_configured_inference_timesteps():
    wan_module = pytest.importorskip(
        "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf"
    )

    cfg = wan_module.WANPolicyHeadConfig(
        skip_component_loading=True,
        train_architecture="none",
        tune_diffusion_model=False,
        text_encoder_cfg={"_target_": "torch.nn.Identity"},
        image_encoder_cfg={"_target_": "torch.nn.Identity"},
        vae_cfg={"_target_": "torch.nn.Identity"},
        diffusion_model_cfg={
            "_target_": "torch.nn.Linear",
            "in_features": 1,
            "out_features": 1,
        },
        action_dim=2,
        action_horizon=2,
        num_frames=1,
        num_inference_timesteps=4,
    )

    head = wan_module.WANPolicyHead(cfg)

    assert head.num_inference_timesteps == 4
    assert head.num_inference_steps == 4


def test_dreamzero_causal_inference_blocks_use_gradient_checkpointing(monkeypatch):
    import torch.utils.checkpoint

    wan_module = pytest.importorskip(
        "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk"
    )

    checkpoint_calls = []

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs.pop("use_reentrant", None))
        return function(*args, **kwargs)

    class FakeBlock(torch.nn.Module):
        def forward(self, x, **kwargs):
            return x + 1, kwargs.get("kv_cache")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", fake_checkpoint)

    model = wan_module.CausalWanModel(
        model_type="t2v",
        patch_size=(1, 1, 1),
        frame_seqlen=1,
        text_len=2,
        in_dim=4,
        dim=4,
        ffn_dim=8,
        freq_dim=4,
        text_dim=4,
        out_dim=4,
        num_heads=2,
        num_layers=2,
        max_chunk_size=-1,
        cross_attn_norm=False,
        concat_first_frame_latent=False,
    )
    model.blocks = torch.nn.ModuleList([FakeBlock(), FakeBlock()])
    model.gradient_checkpointing = True

    x = torch.ones(1, 4, 1, 1, 1, requires_grad=True)
    timestep = torch.zeros(1, 1, dtype=torch.long)
    context = torch.zeros(1, 2, 4)
    freqs = torch.zeros(1, 1, 2)

    output, action_pred, caches = model._forward_blocks(
        x=x,
        seq_len=1,
        freqs=freqs,
        timestep=timestep,
        context=context,
        clip_feature=None,
        embodiment_id=None,
        action=None,
        timestep_action=None,
        state=None,
        kv_cache=[None, None],
        current_start_frame=1,
    )

    assert output.requires_grad
    assert action_pred is None
    assert caches == [None, None]
    assert checkpoint_calls == [False, False]


def test_dreamzero_libero_observation_transform_builds_inference_modalities():
    transforms = pytest.importorskip(
        "rlinf.data.datasets.dreamzero.data_transforms.observation"
    )

    transform = transforms.DreamZeroLiberoObservationTransform(num_history_frames=4)
    env_obs = {
        "main_images": torch.zeros(2, 256, 256, 3, dtype=torch.uint8),
        "wrist_images": torch.ones(2, 256, 256, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 8, dtype=torch.float32),
        "task_descriptions": ["pick up the bowl", "open the drawer"],
    }

    converted = transform.convert(env_obs)

    assert converted["video.image"].shape == (2, 1, 256, 256, 3)
    assert converted["video.wrist_image"].shape == (2, 1, 256, 256, 3)
    assert converted["state.state"].shape == (2, 1, 8)
    assert converted["state.joint_position"].shape == (2, 1, 7)
    assert converted["state.gripper_position"].shape == (2, 1, 1)
    assert converted["annotation.language.task_description"] == [
        "pick up the bowl",
        "open the drawer",
    ]


def test_build_dreamzero_forward_inputs_preserves_batch_time_layout():
    rollout_batch = {
        "curr_obs": {
            "states": torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
        },
        "next_obs": {
            "states": (100 + torch.arange(2 * 3 * 8, dtype=torch.float32)).reshape(
                2, 3, 8
            )
        },
        "actions": torch.arange(2 * 3 * 4 * 7, dtype=torch.float32).reshape(2, 3, 4, 7),
        "rewards": torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
        "dones": torch.zeros(3, 3, 4, dtype=torch.bool),
        "forward_inputs": {
            "action": torch.zeros(2, 3, 28),
            "model_action": torch.zeros(2, 3, 4, 32),
            "states": torch.zeros(2, 3, 8),
        },
    }

    result = build_dreamzero_forward_inputs(rollout_batch)

    assert result["curr_states"].shape == (2, 3, 4, 8)
    assert result["next_states"].shape == (2, 3, 4, 8)
    assert result["actions"].shape == (2, 3, 4, 7)
    assert result["model_action"].shape == (2, 3, 128)
    assert result["rewards"].shape == (2, 3, 4, 1)
    assert result["dones"].shape == (2, 3, 4, 1)
    assert not any(key.startswith("dreamzero_rl.") for key in result)
    assert torch.equal(
        result["curr_states"][0, 0],
        rollout_batch["curr_obs"]["states"][0, 0].expand(4, 8),
    )
    assert torch.equal(result["actions"][0, 0], rollout_batch["actions"][0, 0])


def test_build_dreamzero_forward_inputs_keeps_model_action_without_rl_payload():
    rollout_batch = {
        "curr_obs": {"states": torch.zeros(1, 1, 8)},
        "next_obs": {"states": torch.zeros(1, 1, 8)},
        "actions": torch.full((1, 1, 2, 7), 10.0),
        "rewards": torch.zeros(1, 1, 2),
        "dones": torch.zeros(1, 1, 2, dtype=torch.bool),
        "forward_inputs": {
            "model_action": torch.tensor([[[[-2.0, -0.5], [0.25, 2.0]]]]),
            "dreamzero_rl.action": torch.tensor([[[[-1.5, 0.5], [2.5, -0.25]]]]),
        },
    }

    result = build_dreamzero_forward_inputs(rollout_batch)

    assert result["model_action"].min() >= -1.0
    assert result["model_action"].max() <= 1.0
    assert not any(key.startswith("dreamzero_rl.") for key in result)
    assert torch.equal(result["actions"], rollout_batch["actions"])


def test_dreamzero_ppo_uses_prev_logprob_payload_size_before_shuffle():
    rollout_batch = {
        "prev_logprobs": torch.zeros(5, 4, 16, 7),
        "advantages": torch.zeros(5, 4, 16),
        "returns": torch.zeros(5, 4, 16),
        "actions": torch.zeros(5, 4, 16, 7),
        "forward_inputs": {
            "dreamzero_ppo.action_obs": {"state": torch.zeros(5, 4, 8)},
            "chains": torch.zeros(5, 4, 11, 16, 32),
            "denoise_inds": torch.zeros(5, 4, 1, dtype=torch.long),
            "dreamzero_ppo.action_mask": torch.ones(5, 4, 16, 32, dtype=torch.bool),
            "model_action": torch.zeros(5, 4, 16, 32),
            "action": torch.zeros(5, 4, 16, 7),
        },
    }
    shuffle_id = torch.randperm(
        rollout_batch["prev_logprobs"].shape[0]
        * rollout_batch["prev_logprobs"].shape[1]
    )

    processed = process_nested_dict_for_train(rollout_batch, shuffle_id)

    rollout_size = get_dreamzero_train_rollout_size(
        rollout_batch,
        loss_type="actor_critic",
    )
    flattened_rollout_size = get_dreamzero_train_rollout_size(
        processed,
        loss_type="actor_critic",
        is_flattened=True,
    )

    assert rollout_size == 20
    assert flattened_rollout_size == 20
    assert processed["prev_logprobs"].shape == (20, 16, 7)
    assert processed["forward_inputs"]["chains"].shape == (20, 11, 16, 32)
