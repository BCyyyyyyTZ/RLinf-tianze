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

"""DreamZero algorithms aligned with RLinf registries."""

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import safe_normalize


@register_advantage("dreamzero")
def compute_dreamzero_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = False,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute DreamZero advantages through the standard advantage interface.

    This placeholder intentionally reuses GAE-style return propagation so the
    algorithm can run end-to-end while DreamZero-specific world-model math is
    developed behind the same registry contract.
    """
    batch_size = kwargs.get("batch_size", rewards.shape[-1])
    n_steps = kwargs.get("n_steps", rewards.shape[0])
    if (
        batch_size is not None
        and n_steps is not None
        and (rewards.ndim != 2 or rewards.shape != (n_steps, batch_size))
    ):
        score_rewards = rewards.reshape(-1)
        if score_rewards.numel() != batch_size:
            raise ValueError(
                "DreamZero placeholder advantages expected one score per "
                f"environment after preprocessing, got {score_rewards.numel()} "
                f"scores for batch_size={batch_size}."
            )
        step_rewards = torch.zeros(
            (n_steps, batch_size), dtype=rewards.dtype, device=rewards.device
        )
        step_rewards[-1] = score_rewards
        rewards = step_rewards

    rewards = rewards.reshape(n_steps, batch_size)
    if dones is None:
        dones = torch.zeros(
            (n_steps + 1, batch_size), dtype=torch.bool, device=rewards.device
        )
        dones[-1] = True
    else:
        dones = dones.reshape(dones.shape[0], -1).to(torch.bool)
        if dones.shape[1] != batch_size:
            dones = dones[:, :batch_size]

    returns = torch.zeros_like(rewards)
    running_return = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)
    for step in reversed(range(n_steps)):
        running_return = rewards[step] + gamma * running_return * (~dones[step + 1])
        returns[step] = running_return

    if values is not None:
        values = values.reshape(values.shape[0], -1)[:n_steps, :batch_size]
        advantages = returns - values
    else:
        advantages = returns.clone()

    if normalize_advantages and loss_mask is not None:
        advantages = safe_normalize(
            advantages, loss_mask=loss_mask.reshape_as(advantages)
        )
    if normalize_returns and loss_mask is not None:
        returns = safe_normalize(returns, loss_mask=loss_mask.reshape_as(returns))

    return advantages, returns
