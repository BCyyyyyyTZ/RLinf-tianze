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

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
from tianshou.data import Batch
from transformers.configuration_utils import PretrainedConfig

from rlinf.data.datasets.dreamzero.data_transforms import (
    DreamZeroObservationTransform,
)
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.dreamzero.patch.wan_policy_head_action_only import (
    _action_only_video_context,
)
from rlinf.models.embodiment.dreamzero.ppo_policy import DreamZeroPPOPolicyMixin
from rlinf.utils.logging import get_logger


@dataclass
class DreamZeroConfig(VLAConfig):
    model_type = "dreamzero"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    env_action_dim: int = field(
        default=None, metadata={"help": "Environment action dimension."}
    )
    num_action_chunks: int = field(
        default=16, metadata={"help": "Number of action chunks."}
    )

    relative_action: bool = field(default=False, metadata={"help": "Relative action."})
    relative_action_per_horizon: bool = field(
        default=False, metadata={"help": "Relative action per horizon."}
    )
    relative_action_keys: list = field(
        default_factory=list, metadata={"help": "Relative action keys."}
    )

    data_transforms: ComposedModalityTransform = field(
        default=None,
        metadata={
            "help": "Transforming data modalities, e.g. video frame augmentation or action normalization."
        },
    )

    gradient_checkpointing: bool = False
    add_value_head: bool = False
    noise_method: str = "flow_sde"
    noise_level: float = 0.5
    safe_get_logprob: bool = False
    joint_logprob: bool = False
    num_steps: int = 10
    ppo_deterministic_eval: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DreamZeroPolicy(DreamZeroPPOPolicyMixin, VLA, BasePolicy):
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead."""

    _no_split_modules = [
        "T5SelfAttention",  # text encoder
        "AttentionBlock",  # vae
        "CausalWanAttentionBlock",  # action head
    ]

    def __init__(
        self,
        config: DreamZeroConfig,
    ):
        super().__init__(config)
        self.config = config
        try:
            diffusion_model = getattr(getattr(self, "action_head", None), "model", None)
            enabled = self.config.gradient_checkpointing
            if diffusion_model is not None:
                if hasattr(diffusion_model, "_set_gradient_checkpointing"):
                    diffusion_model._set_gradient_checkpointing(
                        diffusion_model, enabled
                    )
                elif hasattr(diffusion_model, "gradient_checkpointing"):
                    diffusion_model.gradient_checkpointing = enabled
        except Exception:
            pass
        self.observation_transform: DreamZeroObservationTransform | None = getattr(
            config, "observation_transform", None
        )
        if bool(getattr(config, "add_value_head", False)):
            self._setup_ppo_heads(value_input_dim=int(getattr(config, "action_dim", 32)))

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Normalize inputs"""
        obs = batch.obs
        normalized_input = self.config.data_transforms(obs)
        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: Optional[dict] = None, **kwargs):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": batch.normalized_action.cpu()}
        )

        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.config.relative_action
        relative_action_per_horizon = self.config.relative_action_per_horizon
        relative_action_keys = self.config.relative_action_keys
        if (
            (relative_action or relative_action_per_horizon)
            and relative_action_keys
            and obs is not None
        ):
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"

                if action_key not in unnormalized_action:
                    continue

                # Try to find the state data - check multiple possible key formats
                last_state = None

                # Format 1: Direct key like "state.joint_position"
                if state_key in obs:
                    last_state = obs[state_key]
                else:
                    # Format 2: Search for keys containing both "state" and the key name
                    for obs_key in obs.keys():
                        if "state" in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break

                    # Format 3: If key is "joint_position" and obs has "state" key directly
                    # This handles cases where the observation uses modality-level keys
                    if last_state is None and "state" in obs:
                        state_data = obs["state"]
                        # Check if the state data shape matches the action shape
                        action_dim = unnormalized_action[action_key].shape[-1]
                        if torch.is_tensor(state_data):
                            state_dim = state_data.shape[-1]
                        elif isinstance(state_data, np.ndarray):
                            state_dim = state_data.shape[-1]
                        else:
                            state_dim = None

                        if state_dim == action_dim:
                            last_state = state_data

                if last_state is None:
                    continue

                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()

                # Shape is (B, T, D) or (T, D), we want the last timestep
                # After indexing: (B, D) or (D,)
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]  # Get the last timestep

                # Action shape is (horizon, D) or (B, horizon, D)
                # Expand dims to broadcast: (D,) -> (1, D) or (B, D) -> (B, 1, D)
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(
                        last_state, axis=-2
                    )  # Add horizon dimension

                # Add state to relative action to get absolute action
                unnormalized_action[action_key] = (
                    unnormalized_action[action_key] + last_state
                )

        batch.act = unnormalized_action
        return batch

    def _process_batch(self, batch: Batch) -> Batch:
        """Process batch."""
        self._sync_action_head_device()
        # Normalize / transform
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        # If the normalized input is still a Batch, flatten it into a pure dict
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        # Do dtype cast if needed
        target_dtype = next(self.parameters()).dtype
        for k, v in normalized_input.items():
            if (
                torch.is_tensor(v)
                and v.dtype == torch.float32
                and target_dtype != torch.float32
            ):
                normalized_input[k] = v.to(dtype=target_dtype)
        return normalized_input

    def _sync_action_head_device(self) -> None:
        action_head = getattr(self, "action_head", None)
        if action_head is None:
            return
        try:
            device = next(self.parameters()).device
        except StopIteration:
            return
        current = getattr(action_head, "_device", None)
        if current is None or torch.device(current) != device:
            action_head._device = str(device)
            if hasattr(action_head, "_vae_device_ready"):
                action_head._vae_device_ready = False
        if not hasattr(action_head, "trt_engine"):
            action_head.trt_engine = None
        if not hasattr(action_head, "trt_context"):
            action_head.trt_context = None

    @staticmethod
    def _to_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        return tensor

    @staticmethod
    def _action_debug_stats(value: Any) -> dict[str, Any]:
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        arr = np.asarray(arr)
        flat = arr.astype(np.float32, copy=False).reshape(-1)
        if flat.size == 0:
            return {"shape": tuple(arr.shape), "empty": True}
        return {
            "shape": tuple(arr.shape),
            "min": float(np.nanmin(flat)),
            "max": float(np.nanmax(flat)),
            "mean": float(np.nanmean(flat)),
            "std": float(np.nanstd(flat)),
            "sample": arr.reshape(-1, arr.shape[-1])[:2].tolist()
            if arr.ndim > 0
            else arr.tolist(),
        }

    def _debug_action_payload(
        self,
        *,
        normalized_action: torch.Tensor,
        unnormalized_action: dict[str, Any],
        env_actions: np.ndarray,
    ) -> None:
        if os.getenv("DREAMZERO_DEBUG_ACTIONS", "0").lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return
        if getattr(self, "_debug_action_logged", 0) >= int(
            os.getenv("DREAMZERO_DEBUG_ACTIONS_LIMIT", "3")
        ):
            return
        self._debug_action_logged = getattr(self, "_debug_action_logged", 0) + 1
        logger = get_logger()
        logger.info(
            "[DreamZero action debug] normalized_action=%s",
            self._action_debug_stats(normalized_action),
        )
        for key, value in sorted(unnormalized_action.items()):
            if hasattr(value, "shape"):
                logger.info(
                    "[DreamZero action debug] unnormalized %s=%s",
                    key,
                    self._action_debug_stats(value),
                )
        logger.info(
            "[DreamZero action debug] env_actions=%s gripper_counts=%s",
            self._action_debug_stats(env_actions),
            {
                str(k): int(v)
                for k, v in zip(
                    *np.unique(env_actions[..., -1].reshape(-1), return_counts=True)
                )
            },
        )

    @staticmethod
    def _map_gripper_for_env(actions: np.ndarray) -> np.ndarray:
        mode = os.getenv("DREAMZERO_GRIPPER_MODE", "pm1").lower()
        raw = actions[..., -1].copy()
        if mode in ("pm1", "plus_minus_one", "default"):
            actions[..., -1] = np.where(raw > 0, 1.0, -1.0).astype(actions.dtype)
        elif mode in ("zero_one", "01"):
            actions[..., -1] = np.where(raw > 0.5, 1.0, 0.0).astype(actions.dtype)
        elif mode in ("flip_pm1", "pm1_flip"):
            actions[..., -1] = np.where(raw > 0, -1.0, 1.0).astype(actions.dtype)
        elif mode in ("raw", "none"):
            actions[..., -1] = raw.astype(actions.dtype)
        else:
            raise ValueError(
                "Unsupported DREAMZERO_GRIPPER_MODE="
                f"{mode!r}; use pm1, zero_one, flip_pm1, or raw."
            )
        return actions

    def _action_obs_to_device(self, action_obs: dict[str, Any]) -> dict[str, Any]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        def move(value: Any) -> Any:
            if torch.is_tensor(value):
                value = value.to(device=device)
                if value.dtype == torch.float32 and dtype != torch.float32:
                    value = value.to(dtype=dtype)
                return value
            if isinstance(value, dict):
                return {key: move(child) for key, child in value.items()}
            return value

        return {key: move(value) for key, value in action_obs.items()}

    def _sample_initial_action_noise(
        self, batch_size: int, *, device: torch.device
    ) -> torch.Tensor:
        return torch.randn(
            batch_size,
            int(self.config.action_horizon),
            int(self.config.action_dim),
            device=device,
            dtype=torch.float32,
        )

    def _predict_action_velocity(
        self,
        action_obs: dict[str, Any],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_head = getattr(self, "action_head", None)
        if action_head is not None and hasattr(action_head, "predict_action_velocity_only"):
            output = action_head.predict_action_velocity_only(action_obs, x_t, timestep)
            if isinstance(output, dict):
                velocity = output["velocity"]
                value_feature = output.get("value_feature", velocity.float().mean(dim=1))
                return velocity.float(), value_feature.float()
            return output.float(), output.float().mean(dim=1)

        self._reset_action_chain_inference_state()
        obs = dict(action_obs)
        obs["action"] = x_t.to(
            device=next(self.parameters()).device, dtype=next(self.parameters()).dtype
        )
        obs["has_real_action"] = torch.ones(
            x_t.shape[0],
            device=x_t.device,
            dtype=torch.bool,
        )
        with _action_only_video_context(enabled=True):
            pred = self.lazy_joint_video_action_causal(obs)["action_pred"]
        pred = pred.to(device=x_t.device, dtype=torch.float32)
        timestep = timestep.to(device=x_t.device, dtype=torch.float32)
        if timestep.ndim == 1:
            timestep = timestep[:, None, None]
        timestep = timestep.clamp_min(1e-4)
        velocity = (x_t.float() - pred) / timestep
        return velocity, pred.float().mean(dim=1)

    def _reset_action_chain_inference_state(self) -> None:
        action_head = getattr(self, "action_head", None)
        if action_head is None:
            return
        if hasattr(action_head, "release_inference_cache"):
            action_head.release_inference_cache()
            return
        for name in (
            "kv_cache1",
            "kv_cache_neg",
            "crossattn_cache",
            "crossattn_cache_neg",
            "clip_feas",
            "ys",
            "language",
        ):
            if hasattr(action_head, name):
                setattr(action_head, name, None)
        if hasattr(action_head, "current_start_frame"):
            action_head.current_start_frame = 0
        if hasattr(action_head, "skip_countdown"):
            action_head.skip_countdown = 0

    def release_inference_cache(self) -> None:
        self._reset_action_chain_inference_state()

    def _infer_real_action_dim(self, fallback: int) -> int:
        transforms = getattr(getattr(self.config, "data_transforms", None), "transforms", [])
        for transform in reversed(transforms):
            action_order = getattr(transform, "action_concat_order", None)
            if not action_order:
                continue
            try:
                return sum(
                    transform.get_state_action_dims_post_transform(key)
                    for key in action_order
                )
            except Exception:
                return fallback
        return fallback

    def _extract_env_actions(self, act: dict[str, Any]) -> np.ndarray:
        if "action.actions" in act:
            actions = act["action.actions"]
        elif "action" in act:
            actions = act["action"]
        else:
            action_keys = [
                key
                for key in ("action.joint_position", "action.gripper_position")
                if key in act
            ]
            if not action_keys:
                action_keys = sorted(
                    key
                    for key, value in act.items()
                    if key.startswith("action.") and hasattr(value, "shape")
                )
            if not action_keys:
                raise KeyError(f"No action tensors found in DreamZero output: {list(act.keys())}")
            tensors = [act[key] for key in action_keys]
            if any(torch.is_tensor(tensor) for tensor in tensors):
                tensors = [
                    tensor if torch.is_tensor(tensor) else torch.as_tensor(tensor)
                    for tensor in tensors
                ]
                actions = torch.cat(tensors, dim=-1)
            else:
                actions = np.concatenate(tensors, axis=-1)
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        env_action_dim = self.config.env_action_dim
        if (
            env_action_dim is not None
            and actions.ndim == 3
            and actions.shape[-2] == env_action_dim
            and actions.shape[-1] != env_action_dim
        ):
            actions = np.swapaxes(actions, -1, -2)
        if env_action_dim is not None and actions.shape[-1] != env_action_dim:
            if actions.shape[-1] > env_action_dim and env_action_dim >= 2:
                actions = np.concatenate(
                    [actions[..., : env_action_dim - 1], actions[..., -1:]], axis=-1
                )
            elif actions.shape[-1] > env_action_dim:
                actions = actions[..., :env_action_dim]
            else:
                padded = np.zeros((*actions.shape[:-1], env_action_dim), dtype=actions.dtype)
                padded[..., : actions.shape[-1]] = actions
                actions = padded
        num_action_chunks = self.config.num_action_chunks
        if num_action_chunks is not None and actions.ndim >= 3:
            actions = actions[:, :num_action_chunks, :]
        return actions

    def _actions_from_unapply(self, act: dict[str, Any]) -> np.ndarray:
        action_keys = getattr(self, "_action_keys", None)
        if action_keys:
            values = [act[key] for key in action_keys if key in act]
            if values:
                if any(torch.is_tensor(value) for value in values):
                    tensors = [
                        value if torch.is_tensor(value) else torch.as_tensor(value)
                        for value in values
                    ]
                    return torch.cat(tensors, dim=-1).detach().cpu().numpy()
                return np.concatenate(values, axis=-1)
        return self._extract_env_actions(act)

    def _observation_convert(self, env_obs: dict) -> dict:
        """Convert environment observation to DreamZero model input."""
        if self.observation_transform is None:
            raise RuntimeError(
                "DreamZeroPolicy requires config.observation_transform for env "
                "observation conversion."
            )
        converted_obs = self.observation_transform.convert(env_obs)
        prompts = converted_obs.get("annotation.language.task_description", [])
        if os.getenv("DREAMZERO_DEBUG_LANGUAGE", "0").lower() in ("1", "true", "yes"):
            preview = list(prompts)[: min(2, len(prompts))]
            get_logger().info("[DreamZero language] task_descriptions=%s", preview)
        return converted_obs

    def predict_action_batch(self, env_obs, mode, **kwargs) -> np.ndarray:
        """
        input:
            env_obs:
                - main_images: [B,H,W,C] uint8
                - extra_view_images: [B,H,W,C]
                - states: [B,D]
                - task_descriptions: list[str] or None
        output:
            actions: np.ndarray [B, num_action_chunks, 8]  # 6ee + 1 gripper
            result: dict  # compatible with rollout interface"""

        converted_obs = self._observation_convert(env_obs)
        batch = Batch(obs=converted_obs)
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            action_obs = self._prepare_action_only_obs(normalized_input)
            batch_size = next(
                value.shape[0]
                for value in action_obs.values()
                if torch.is_tensor(value)
            )
            initial_noise = self._sample_initial_action_noise(
                batch_size, device=next(self.parameters()).device
            )
            chains, denoise_inds, prev_logprobs, prev_values = self._sample_action_chain(
                action_obs=action_obs,
                initial_noise=initial_noise,
                mode=mode,
                compute_values=bool(getattr(self.config, "add_value_head", False)),
            )

        normalized_action = chains[:, -1].float()

        # Unnormalize actions (pass obs for relative action normalization)
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": normalized_action.cpu()}
        )
        batch.act = unnormalized_action

        actions = self._extract_env_actions(batch.act)
        actions = self._map_gripper_for_env(actions)
        self._debug_action_payload(
            normalized_action=normalized_action,
            unnormalized_action=batch.act,
            env_actions=actions,
        )

        assert actions.shape[-1] == self.config.env_action_dim, (
            f"Action shape mismatch: {actions.shape} != {self.config.env_action_dim}"
        )

        env_action_tensor = torch.as_tensor(actions, dtype=torch.float32).cpu()
        real_action_dim = self._infer_real_action_dim(
            min(normalized_action.shape[-1], actions.shape[-1] + 1)
        )
        action_mask = torch.zeros_like(normalized_action, dtype=torch.bool)
        action_mask[..., :real_action_dim] = True
        forward_inputs = self._build_ppo_forward_inputs(
            action_obs=action_obs,
            chains=chains.cpu(),
            denoise_inds=denoise_inds.cpu(),
            action_mask=action_mask.cpu(),
            model_action=normalized_action.cpu().contiguous(),
            env_action=env_action_tensor.cpu().contiguous(),
        )
        result = {
            "prev_logprobs": prev_logprobs.detach().cpu().to(dtype=torch.float32),
            "prev_values": prev_values.detach().cpu().to(dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data=None, **kwargs):
        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError("sft_forward requires `data` from the SFT dataloader.")
        outputs = super().forward(data)
        if "loss" not in outputs:
            raise ValueError("sft_forward requires `loss` in the outputs.")
        return outputs

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Default forward pass."""
        if forward_inputs is None:
            raise ValueError("DreamZero default_forward requires `forward_inputs`.")
        if "dreamzero_ppo.action_obs" in forward_inputs:
            return self.ppo_forward(forward_inputs=forward_inputs, **kwargs)
        raise KeyError(
            "DreamZero default_forward only supports OpenPI-style PPO inputs "
            "('dreamzero_ppo.action_obs')."
        )
