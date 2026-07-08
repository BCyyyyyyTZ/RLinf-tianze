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

import math
from typing import Any

import torch


def normalize_action_payload(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    if action.ndim == 2:
        if action.shape[-1] % action_dim == 0 and action.shape[-1] != action_dim:
            action = action.reshape(action.shape[0], -1, action_dim)
        else:
            action = action.unsqueeze(1)
    if action.ndim != 3:
        raise ValueError(
            f"DreamZero PPO action must be [B,T,D], got {tuple(action.shape)}"
        )
    return torch.nan_to_num(
        action.float(), nan=0.0, posinf=1.0, neginf=-1.0
    ).clamp(-1.0, 1.0)


def align_action_mask(mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    mask = mask.to(device=target.device, dtype=torch.bool)
    if mask.shape == target.shape:
        return mask
    if mask.numel() == target.numel():
        return mask.reshape_as(target)
    raise ValueError(
        f"DreamZero PPO action mask shape {tuple(mask.shape)} cannot align to "
        f"{tuple(target.shape)}"
    )


def dreamzero_action_get_logprob_norm(
    sample: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    safe_get_logprob: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    sample = sample.to(device=mu.device, dtype=torch.float32)
    mu = mu.float()
    sigma = sigma.to(device=mu.device, dtype=torch.float32)
    if sample.shape != mu.shape:
        sample = sample.reshape_as(mu)
    if sigma.shape != mu.shape:
        sigma = sigma.expand_as(mu)
    if safe_get_logprob:
        logprob = -(sample - mu).pow(2)
    else:
        zero_std = sigma == 0
        sigma_safe = torch.where(zero_std, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * math.log(2.0 * math.pi)
        exponent_term = -0.5 * ((sample - mu) / sigma_safe).pow(2)
        logprob = constant_term + exponent_term
        logprob = torch.where(zero_std, torch.zeros_like(logprob), logprob)
    valid = align_action_mask(mask, mu)
    return torch.where(valid, logprob, torch.zeros_like(logprob))


def dreamzero_action_gaussian_entropy(
    sigma: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    sigma = sigma.float()
    zero_std = sigma == 0
    sigma_safe = torch.where(zero_std, torch.ones_like(sigma), sigma)
    entropy = 0.5 * torch.log(2.0 * math.pi * math.e * sigma_safe.pow(2))
    entropy = torch.where(zero_std, torch.zeros_like(entropy), entropy)
    valid = align_action_mask(mask, sigma)
    return torch.where(valid, entropy, torch.zeros_like(entropy))


def tensor_payload_to_cpu(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().contiguous()
        elif isinstance(value, dict):
            result[key] = tensor_payload_to_cpu(value)
        else:
            result[key] = value
    return result
