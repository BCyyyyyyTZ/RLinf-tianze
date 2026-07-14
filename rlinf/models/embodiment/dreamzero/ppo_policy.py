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

from contextlib import contextmanager
from typing import Any

import torch

from rlinf.models.embodiment.dreamzero.action_only import DreamZeroActionOnlyMixin
from rlinf.models.embodiment.dreamzero.ppo_utils import (
    dreamzero_action_gaussian_entropy,
    dreamzero_action_get_logprob_norm,
    tensor_payload_to_cpu,
)
from rlinf.models.embodiment.modules.value_head import ValueHead


class DreamZeroPPOPolicyMixin(DreamZeroActionOnlyMixin):
    def _setup_ppo_heads(self, *, value_input_dim: int) -> None:
        self.value_head = ValueHead(
            input_dim=value_input_dim,
            hidden_sizes=(256, 128),
            output_dim=1,
            activation="relu",
            bias_last=True,
        )

    def _predict_action_velocity(
        self,
        action_obs: dict[str, Any],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        *,
        use_velocity_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _build_action_mask(self, action: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(action, dtype=torch.bool)

    @staticmethod
    def _get_dreamzero_timesteps(
        denoise_steps: int, device: torch.device
    ) -> torch.Tensor:
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        return torch.cat([timesteps, torch.tensor([0.0], device=device)])

    def _compute_values_from_action_step(self, x_t_mean: torch.Tensor) -> torch.Tensor:
        pooled = x_t_mean.float().mean(dim=1)
        return self.value_head(pooled).float()

    def _get_noise_level(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        noise_level = float(getattr(self.config, "noise_level", 0.5))
        return torch.tensor(noise_level, device=device, dtype=dtype)

    def _use_velocity_only_for_ppo(self) -> bool:
        return bool(getattr(self.config, "ppo_use_velocity_only", False))

    @staticmethod
    def _use_velocity_only_from_forward_inputs(
        forward_inputs: dict[str, Any],
        default: bool,
    ) -> bool:
        value = forward_inputs.get("dreamzero_ppo.use_velocity_only", default)
        if torch.is_tensor(value):
            return bool(value.detach().reshape(-1)[0].item())
        return bool(value)

    @contextmanager
    def _dreamzero_ppo_deterministic_context(self):
        was_training = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(was_training)

    def _select_policy_action_dims(
        self,
        tensor: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if action_mask is not None:
            mask = action_mask.to(device=tensor.device, dtype=torch.bool)
            while mask.ndim < tensor.ndim:
                mask = mask.unsqueeze(1)
            if mask.shape == tensor.shape:
                dim_mask = mask.reshape(-1, mask.shape[-1]).any(dim=0)
                if bool(dim_mask.any()):
                    return tensor.index_select(
                        -1,
                        torch.nonzero(dim_mask, as_tuple=False).flatten(),
                    )

        action_dim = int(getattr(self.config, "action_dim", tensor.shape[-1]))
        env_action_dim = int(getattr(self.config, "env_action_dim", action_dim))
        policy_action_dim = int(
            getattr(
                self.config,
                "ppo_loss_action_dim",
                min(action_dim, env_action_dim),
            )
        )
        policy_action_dim = max(1, min(policy_action_dim, tensor.shape[-1]))
        return tensor[..., :policy_action_dim]

    def dreamzero_action_sample_mean_var_val(
        self,
        *,
        x_t: torch.Tensor,
        idx: int | torch.Tensor,
        action_obs: dict[str, Any],
        sample_method: str,
        denoise_steps: int,
        compute_values: bool = True,
        use_velocity_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = x_t.shape[0]
        device = x_t.device
        if isinstance(idx, int):
            idx = torch.full((bsize,), idx, device=device, dtype=torch.long)
        else:
            idx = idx.to(device=device, dtype=torch.long)

        timesteps = self._get_dreamzero_timesteps(denoise_steps, device)
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        v_t, value_features = self._predict_action_only_velocity(
            action_obs, x_t, t_input, use_velocity_only=use_velocity_only
        )
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if sample_method == "flow_ode":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif sample_method == "flow_sde":
            noise_level = self._get_noise_level(device=device, dtype=x_t.dtype)
            denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
            sigma_ratio = timesteps / (1 - denom_timesteps)
            sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
            x_t_std = torch.sqrt(delta) * sigma_i
        else:
            raise ValueError(f"Invalid DreamZero PPO noise method: {sample_method}")

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        if compute_values:
            if torch.is_tensor(value_features) and value_features.ndim == 2:
                value_dtype = next(self.value_head.parameters()).dtype
                value_t = self.value_head(value_features.to(dtype=value_dtype))[:, 0].float()
            else:
                value_t = self._compute_values_from_action_step(x_t_mean)[:, 0]
        else:
            value_t = torch.zeros((bsize,), device=device)
        return x_t_mean, x_t_std, value_t, v_t

    def _sample_action_chain(
        self,
        *,
        action_obs: dict[str, Any],
        initial_noise: torch.Tensor,
        mode: str,
        compute_values: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = initial_noise.float()
        bsize = x_t.shape[0]
        num_steps = int(getattr(self.config, "num_steps", 10))
        chains = [x_t]
        log_probs = []
        values = []
        joint_logprob = bool(getattr(self.config, "joint_logprob", False))
        use_velocity_only = self._use_velocity_only_for_ppo()
        if mode == "train":
            if joint_logprob:
                denoise_inds = torch.arange(num_steps, device=x_t.device)[None].repeat(
                    bsize, 1
                )
            else:
                selected_denoise_ind = torch.randint(
                    0, num_steps, (1,), device=x_t.device
                )
                denoise_inds = selected_denoise_ind[None].repeat(bsize, 1)
        else:
            denoise_inds = torch.full(
                (bsize, 1), num_steps - 1, device=x_t.device, dtype=torch.long
            )

        configured_sample_method = str(getattr(self.config, "noise_method", "flow_sde"))
        with self._dreamzero_ppo_deterministic_context():
            for idx in range(num_steps):
                if joint_logprob:
                    sample_method = configured_sample_method
                elif mode == "train":
                    is_selected_step = bool(torch.all(denoise_inds[:, 0] == idx).item())
                    sample_method = (
                        configured_sample_method
                        if is_selected_step
                        else "flow_ode"
                    )
                else:
                    sample_method = "flow_ode"
                x_t_mean, x_t_std, value_t, _ = self.dreamzero_action_sample_mean_var_val(
                    x_t=x_t,
                    idx=idx,
                    action_obs=action_obs,
                    sample_method=sample_method,
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                    use_velocity_only=use_velocity_only,
                )
                x_t = x_t_mean + torch.randn_like(x_t_mean) * x_t_std
                chains.append(x_t)
                log_probs.append(
                    dreamzero_action_get_logprob_norm(
                        x_t,
                        x_t_mean,
                        x_t_std,
                        safe_get_logprob=bool(
                            getattr(self.config, "safe_get_logprob", False)
                        ),
                    )
                )
                values.append(value_t)

        chains_tensor = torch.stack(chains, dim=1)
        logprob_tensor = torch.stack(log_probs, dim=1)
        value_tensor = torch.stack(values, dim=1)
        action_mask = self._build_action_mask(chains_tensor[:, -1])

        if joint_logprob and mode == "train":
            selected_logprobs = logprob_tensor
            selected_values = value_tensor.mean(dim=1)
            selected_logprobs = selected_logprobs.masked_fill(
                ~action_mask[:, None], 0.0
            )
            prev_logprobs = selected_logprobs.reshape(bsize, num_steps, -1).mean(dim=1)
            prev_logprobs = prev_logprobs.reshape_as(chains_tensor[:, -1])
            prev_logprobs = self._select_policy_action_dims(prev_logprobs, action_mask)
            prev_values = selected_values[:, None]
            return chains_tensor, denoise_inds, prev_logprobs, prev_values

        batch_indices = torch.arange(bsize, device=x_t.device)
        selected_logprobs = logprob_tensor[batch_indices, denoise_inds[:, 0]]
        selected_values = value_tensor[batch_indices, denoise_inds[:, 0]]
        selected_logprobs = selected_logprobs.masked_fill(~action_mask, 0.0)
        prev_logprobs = selected_logprobs.reshape_as(chains_tensor[:, -1])
        prev_logprobs = self._select_policy_action_dims(prev_logprobs, action_mask)
        prev_values = selected_values[:, None]
        return chains_tensor, denoise_inds, prev_logprobs, prev_values

    def _build_ppo_forward_inputs(
        self,
        *,
        action_obs: dict[str, Any],
        chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        action_mask: torch.Tensor,
        model_action: torch.Tensor,
        env_action: torch.Tensor,
    ) -> dict[str, Any]:
        return tensor_payload_to_cpu(
            {
                "dreamzero_ppo.action_obs": self._prepare_action_only_obs(action_obs),
                "chains": chains,
                "denoise_inds": denoise_inds,
                "dreamzero_ppo.action_mask": action_mask,
                "dreamzero_ppo.use_velocity_only": torch.full(
                    (chains.shape[0],),
                    self._use_velocity_only_for_ppo(),
                    dtype=torch.bool,
                    device=chains.device,
                ),
                "model_action": model_action,
                "action": env_action,
            }
        )

    def ppo_forward(
        self,
        *,
        forward_inputs: dict[str, Any],
        compute_logprobs: bool = True,
        compute_values: bool = True,
        compute_entropy: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        device = next(self.parameters()).device
        action_obs = forward_inputs["dreamzero_ppo.action_obs"]
        chains = forward_inputs["chains"].to(device)
        denoise_inds = forward_inputs["denoise_inds"].to(device)
        action_mask = forward_inputs.get("dreamzero_ppo.action_mask", None)
        if action_mask is not None:
            action_mask = action_mask.to(device)
        log_probs, values, entropy = self.dreamzero_action_get_log_prob_value(
            action_obs=action_obs,
            chains=chains,
            denoise_inds=denoise_inds,
            action_mask=action_mask,
            compute_values=compute_values,
            use_velocity_only=self._use_velocity_only_from_forward_inputs(
                forward_inputs,
                self._use_velocity_only_for_ppo(),
            ),
        )

        result: dict[str, torch.Tensor] = {}
        if compute_logprobs:
            result["logprobs"] = log_probs.float()
        if compute_values:
            result["values"] = values.float()
        if compute_entropy:
            result["entropy"] = entropy.float()
        return result

    def dreamzero_action_get_log_prob_value(
        self,
        *,
        action_obs: dict[str, Any],
        chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        action_mask: torch.Tensor | None,
        compute_values: bool,
        use_velocity_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = chains.shape[0]
        batch_indices = torch.arange(bsize, device=chains.device)
        if bool(getattr(self.config, "joint_logprob", False)):
            num_steps = int(getattr(self.config, "num_steps", chains.shape[1] - 1))
        else:
            num_steps = 1
        logprob_steps = []
        value_steps = []
        entropy_steps = []
        with self._dreamzero_ppo_deterministic_context():
            for idx in range(num_steps):
                denoise_ind = denoise_inds[:, idx]
                chains_pre = chains[batch_indices, denoise_ind]
                chains_next = chains[batch_indices, denoise_ind + 1]
                x_t_mean, x_t_std, value_t, _ = self.dreamzero_action_sample_mean_var_val(
                    x_t=chains_pre,
                    idx=denoise_ind,
                    action_obs=action_obs,
                    sample_method=str(getattr(self.config, "noise_method", "flow_sde")),
                    denoise_steps=int(
                        getattr(self.config, "num_steps", chains.shape[1] - 1)
                    ),
                    compute_values=compute_values,
                    use_velocity_only=use_velocity_only,
                )
                logprob_steps.append(
                    dreamzero_action_get_logprob_norm(
                        chains_next,
                        x_t_mean,
                        x_t_std,
                        safe_get_logprob=bool(
                            getattr(self.config, "safe_get_logprob", False)
                        ),
                        mask=action_mask,
                    )
                )
                entropy_steps.append(
                    dreamzero_action_gaussian_entropy(x_t_std, mask=action_mask)
                )
                value_steps.append(value_t)

        per_dim_logprob = torch.stack(logprob_steps, dim=1)
        per_dim_entropy = torch.stack(entropy_steps, dim=1)
        values = torch.stack(value_steps, dim=1).mean(dim=1, keepdim=True)
        if bool(getattr(self.config, "joint_logprob", False)):
            logprobs = per_dim_logprob.reshape(bsize, num_steps, -1).mean(dim=1)
            entropy = per_dim_entropy.reshape(bsize, num_steps, -1).mean(dim=1)
            logprobs = logprobs.reshape_as(chains[:, -1])
            entropy = entropy.reshape_as(chains[:, -1])
            return (
                self._select_policy_action_dims(logprobs, action_mask),
                values,
                self._select_policy_action_dims(entropy, action_mask),
            )
        logprobs = per_dim_logprob[:, 0].reshape_as(chains[:, -1])
        entropy = per_dim_entropy[:, 0].reshape_as(chains[:, -1])
        return (
            self._select_policy_action_dims(logprobs, action_mask),
            values,
            self._select_policy_action_dims(entropy, action_mask),
        )
