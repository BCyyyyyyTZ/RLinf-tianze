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

from typing import Any

import torch


class DreamZeroActionOnlyMixin:
    """Action-only facade for DreamZero PPO.

    The PPO path trains and replays action-head behavior only. It must not expose
    generated video, VAE latents, world-model transitions, or video losses through
    rollout forward_inputs.
    """

    def _prepare_action_only_obs(self, normalized_input: dict[str, Any]) -> dict[str, Any]:
        return normalized_input

    def _predict_action_only_velocity(
        self,
        action_obs: dict[str, Any],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        *,
        use_velocity_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._predict_action_velocity(
            action_obs, x_t, timestep, use_velocity_only=use_velocity_only
        )
