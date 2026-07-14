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
import functools
import os
from typing import Any, cast

from diffusers.schedulers.scheduling_utils import SchedulerOutput
import torch


_ACTION_ONLY_VIDEO_CONTEXT: list[bool] = []


def _is_action_only_video_context() -> bool:
    return bool(_ACTION_ONLY_VIDEO_CONTEXT and _ACTION_ONLY_VIDEO_CONTEXT[-1])


@contextmanager
def _action_only_video_context(enabled: bool):
    _ACTION_ONLY_VIDEO_CONTEXT.append(enabled)
    try:
        yield
    finally:
        _ACTION_ONLY_VIDEO_CONTEXT.pop()


def lazy_joint_video_action(original_func):
    @functools.wraps(original_func)
    def wrapped(self, *args, **kwargs):
        return_video = args[3] if len(args) >= 4 else kwargs.get("return_video", True)
        with _action_only_video_context(enabled=not return_video):
            return original_func(self, *args, **kwargs)

    return wrapped


def flow_unipc_step(original_func):
    @functools.wraps(original_func)
    def wrapped(self, *args, **kwargs):
        if len(args) >= 3:
            model_output, timestep, sample = args[:3]
            remaining_args = args[3:]
        else:
            model_output = kwargs["model_output"]
            timestep = kwargs["timestep"]
            sample = kwargs["sample"]
            remaining_args = ()

        if _is_action_only_video_context() and sample.ndim == 5:
            if kwargs.get("return_dict", True) is False:
                return (sample,)
            return SchedulerOutput(prev_sample=sample)

        if len(args) >= 3:
            return original_func(self, model_output, timestep, sample, *remaining_args, **kwargs)
        return original_func(self, **kwargs)

    return wrapped


def _run_diffusion_steps(
    self,
    noisy_input: torch.Tensor,
    timestep: torch.Tensor,
    action: torch.Tensor,
    timestep_action: torch.Tensor,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    context: torch.Tensor,
    seq_len: int,
    y: torch.Tensor,
    clip_feature: torch.Tensor,
    kv_caches: list,
    crossattn_caches: list,
    kv_cache_metadata: dict[str, bool | int],
    return_video_pred: bool | None = None,
) -> list[tuple[torch.Tensor | None, torch.Tensor]]:
    if return_video_pred is None:
        return_video_pred = not _is_action_only_video_context()

    predictions = []
    for index, prompt_emb in enumerate(context):
        kv_cache = kv_caches[index]
        crossattn_cache = crossattn_caches[index]
        trt_engine = getattr(self, "trt_engine", None)
        if not kv_cache_metadata["update_kv_cache"] and trt_engine is not None:
            obs_noise_pred, action_noise_pred = trt_engine(
                noisy_input,
                timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                context=prompt_emb,
                y=y,
                clip_feature=clip_feature,
                kv_cache=kv_cache,
            )
        else:
            obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                noisy_input,
                timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=embodiment_id,
                context=prompt_emb,
                seq_len=seq_len,
                y=y,
                clip_feature=clip_feature,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=kv_cache_metadata["start_frame"],
                return_video_pred=return_video_pred,
            )
            if kv_cache_metadata["update_kv_cache"]:
                for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                    kv_cache[block_index] = updated_kv_cache.clone()
        if obs_noise_pred is not None:
            obs_noise_pred = obs_noise_pred.clone()
        elif not return_video_pred:
            obs_noise_pred = torch.zeros_like(noisy_input)
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()
        else:
            action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device)
        predictions.append((obs_noise_pred, action_noise_pred))
    return self._exchange_predictions(predictions)


def _normalize_action_videos(self, videos: torch.Tensor) -> torch.Tensor:
    videos = videos.permute(0, 4, 1, 2, 3).contiguous()
    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
        bsize, channels, frames, height, width = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4).reshape(
            bsize * frames, channels, height, width
        )
        videos = self.normalize_video(videos)
        videos = videos.reshape(bsize, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        )
        assert (
            videos.min() >= -1.0 and videos.max() <= 1.0
        ), "videos must be in [-1,1] range"
    videos = videos.to(device=self._device, dtype=torch.bfloat16)

    target_h = getattr(self.config, "target_video_height", None)
    target_w = getattr(self.config, "target_video_width", None)
    if target_h is None or target_w is None:
        if getattr(self.model, "frame_seqlen", None) in (50, 55):
            target_h, target_w = 176, 320
        else:
            target_h, target_w = None, None
    if target_h is not None and target_w is not None:
        _, _, frames, height, width = videos.shape
        if (height, width) != (target_h, target_w):
            bsize, channels, _, _, _ = videos.shape
            videos = torch.nn.functional.interpolate(
                videos.reshape(bsize * frames, channels, height, width),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(bsize, channels, frames, target_h, target_w)
    return videos


def _prepare_action_velocity_context(
    self,
    action_input: Any,
) -> dict[str, torch.Tensor | list[torch.Tensor] | torch.device | torch.dtype | int]:
    data = action_input
    videos = self._normalize_action_videos(data["images"])
    state_features = action_input.state.to(device=self._device, dtype=torch.bfloat16)
    embodiment_id = action_input.embodiment_id.to(device=self._device)

    text_inputs = self._prepare_text_inputs(data)
    prompt_embs = [
        self.encode_prompt(text.to(self._device), attention_mask.to(self._device))
        for text, attention_mask in text_inputs
    ]

    _, _, _, height, width = videos.shape
    image = videos[:, :, :1].transpose(1, 2)
    clip_feas, ys, image_latent = self.encode_image(
        image, self.num_frames, height, width
    )

    batch_size = image_latent.shape[0]
    dtype = image_latent.dtype
    device = image_latent.device
    _, _, _, latent_h, latent_w = image_latent.shape
    tokens_per_frame = (latent_h // 2) * (latent_w // 2)
    velocity_num_frames = 1 + self.num_frame_per_block
    seq_len = velocity_num_frames * tokens_per_frame

    return {
        "prompt_embs": prompt_embs,
        "state_features": state_features,
        "embodiment_id": embodiment_id,
        "clip_feas": clip_feas,
        "ys": ys,
        "batch_size": batch_size,
        "velocity_num_frames": velocity_num_frames,
        "seq_len": seq_len,
        "frame_seqlen": tokens_per_frame,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "dtype": dtype,
        "device": device,
    }


def _ppo_timestep_to_wan_timestep(
    self,
    timestep: torch.Tensor,
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    timestep = timestep.to(device=device, dtype=torch.float32)
    if timestep.ndim == 0:
        timestep = timestep[None]
    timestep = timestep.reshape(-1).clamp(0.0, 1.0)
    scale = int(getattr(self.scheduler, "num_train_timesteps", 1000))
    timestep = torch.round(timestep * scale).to(dtype=torch.int64)
    timestep = timestep.clamp(0, max(scale - 1, 0))
    return timestep[:, None].expand(-1, horizon).contiguous()


def _run_action_velocity_step(
    self,
    *,
    noisy_input: torch.Tensor,
    timestep: torch.Tensor,
    action: torch.Tensor,
    timestep_action: torch.Tensor,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    context: list[torch.Tensor],
    seq_len: int,
    y: torch.Tensor,
    clip_feature: torch.Tensor,
) -> list[tuple[torch.Tensor | None, torch.Tensor]]:
    predictions = []
    for prompt_emb in context:
        model_kwargs = dict(
            action=action,
            timestep_action=timestep_action,
            state=state,
            embodiment_id=embodiment_id,
            context=prompt_emb,
            seq_len=seq_len,
            y=y,
            clip_feature=clip_feature,
            clean_x=None,
        )
        try:
            obs_noise_pred, action_noise_pred = self.model(
                noisy_input,
                timestep,
                **model_kwargs,
                return_video_pred=False,
            )[:2]
        except TypeError:
            obs_noise_pred, action_noise_pred = self.model(
                noisy_input,
                timestep,
                **model_kwargs,
            )[:2]
        if obs_noise_pred is None:
            obs_noise_pred = torch.zeros_like(noisy_input)
        if action_noise_pred is None:
            action_noise_pred = torch.zeros_like(action)
        predictions.append((obs_noise_pred, action_noise_pred.clone()))
    return self._exchange_predictions(predictions)


def predict_action_velocity_only(
    self,
    action_obs: dict[str, Any],
    x_t: torch.Tensor,
    timestep: torch.Tensor,
) -> dict[str, torch.Tensor]:
    self.set_frozen_modules_to_eval_mode()
    # PPO training should match OpenPI's stateless policy forward semantics.
    # Keep DreamZero inference caches from spanning optimization micro-batches.
    if os.environ.get(
        "DREAMZERO_RELEASE_PPO_VELOCITY_CACHE_EACH_STEP", "true"
    ).lower() in {"1", "true", "yes", "on"}:
        self.release_inference_cache()

    action_input = self.prepare_input(action_obs)
    context = self._prepare_action_velocity_context(action_input)

    device = cast(torch.device, context["device"])
    dtype = cast(torch.dtype, context["dtype"])
    batch_size = cast(int, context["batch_size"])
    velocity_num_frames = cast(int, context["velocity_num_frames"])
    latent_h = cast(int, context["latent_h"])
    latent_w = cast(int, context["latent_w"])
    seq_len = cast(int, context["seq_len"])
    clip_feas = cast(torch.Tensor, context["clip_feas"])
    ys = cast(torch.Tensor, context["ys"])
    prompt_embs = cast(list[torch.Tensor], context["prompt_embs"])
    state_features = cast(torch.Tensor, context["state_features"])
    embodiment_id = cast(torch.Tensor, context["embodiment_id"])

    noisy_input = torch.zeros(
        batch_size,
        self.model.in_dim,
        velocity_num_frames,
        latent_h,
        latent_w,
        device=device,
        dtype=dtype,
    )
    video_timestep = torch.zeros(
        [batch_size, velocity_num_frames],
        device=device,
        dtype=torch.int64,
    )
    action = x_t.to(device=device, dtype=dtype)
    timestep_action = self._ppo_timestep_to_wan_timestep(
        timestep,
        horizon=action.shape[1],
        device=device,
    )
    if velocity_num_frames <= ys.shape[2] - 1:
        y = ys[:, :, 1 : 1 + velocity_num_frames]
    else:
        y = ys[:, :, -velocity_num_frames:]

    predictions = self._run_action_velocity_step(
        noisy_input=noisy_input,
        timestep=video_timestep,
        action=action,
        timestep_action=timestep_action,
        state=state_features,
        embodiment_id=embodiment_id,
        context=prompt_embs,
        seq_len=seq_len,
        y=y,
        clip_feature=clip_feas,
    )
    if len(predictions) == 1:
        action_velocity = predictions[0][1]
    else:
        action_velocity = predictions[1][1] + self.cfg_scale * (
            predictions[0][1] - predictions[1][1]
        )
    action_velocity = action_velocity.to(device=x_t.device, dtype=torch.float32)
    return {
        "velocity": action_velocity,
        "value_feature": action_velocity.mean(dim=1),
    }
