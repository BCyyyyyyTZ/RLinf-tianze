import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)


def _all_tensors_contiguous_in_dict(data: dict) -> bool:
    for value in data.values():
        if isinstance(value, torch.Tensor):
            if not value.is_contiguous():
                return False
        elif isinstance(value, dict):
            if not _all_tensors_contiguous_in_dict(value):
                return False
    return True


def test_to_splited_trajectories_returns_contiguous_tensors():
    rollout = EmbodiedRolloutResult(max_episode_length=8)
    bsz = 4

    # Create non-contiguous tensors via transpose/view-like operations.
    actions = torch.randn(bsz, 6).transpose(0, 1).transpose(0, 1)
    prev_logprobs = torch.randn(bsz, 2, 3).transpose(1, 2)
    prev_values = torch.randn(bsz, 2).transpose(0, 1).transpose(0, 1)
    rewards = torch.randn(bsz, 2).transpose(1, 0).transpose(1, 0)
    dones = torch.zeros(bsz, 2, dtype=torch.bool).transpose(1, 0).transpose(1, 0)
    versions = torch.ones(bsz, 2).transpose(0, 1).transpose(0, 1)

    forward_inputs = {
        "action": torch.randn(bsz, 6).transpose(0, 1).transpose(0, 1),
        "model_action": torch.randn(bsz, 6).transpose(0, 1).transpose(0, 1),
    }
    chunk = ChunkStepResult(
        actions=actions,
        prev_logprobs=prev_logprobs,
        prev_values=prev_values,
        rewards=rewards,
        dones=dones,
        terminations=dones,
        truncations=dones,
        versions=versions,
        forward_inputs=forward_inputs,
    )
    rollout.append_step_result(chunk)

    splited = rollout.to_splited_trajectories(split_size=2)
    assert len(splited) == 2
    for traj in splited:
        assert traj.actions is None or traj.actions.is_contiguous()
        assert traj.prev_logprobs is None or traj.prev_logprobs.is_contiguous()
        assert traj.prev_values is None or traj.prev_values.is_contiguous()
        assert traj.rewards is None or traj.rewards.is_contiguous()
        assert traj.versions is None or traj.versions.is_contiguous()
        if traj.forward_inputs:
            assert _all_tensors_contiguous_in_dict(traj.forward_inputs)


def test_to_splited_trajectories_keeps_nested_forward_inputs():
    rollout = EmbodiedRolloutResult(max_episode_length=8)
    bsz = 4
    dones = torch.zeros(bsz, 2, dtype=torch.bool)
    rollout.append_step_result(
        ChunkStepResult(
            actions=torch.randn(bsz, 6),
            prev_logprobs=torch.randn(bsz, 2, 3),
            prev_values=torch.randn(bsz, 2),
            rewards=torch.randn(bsz, 2),
            dones=dones,
            terminations=dones,
            truncations=dones,
            versions=torch.ones(bsz, 2),
            forward_inputs={
                "chains": torch.randn(bsz, 2, 3),
                "dreamzero_ppo.action_obs": {
                    "state": torch.arange(bsz * 4).reshape(bsz, 4),
                },
            },
        )
    )

    splited = rollout.to_splited_trajectories(split_size=2)

    assert len(splited) == 2
    assert splited[0].forward_inputs["dreamzero_ppo.action_obs"]["state"].shape == (
        1,
        2,
        4,
    )
    assert splited[1].forward_inputs["dreamzero_ppo.action_obs"]["state"].shape == (
        1,
        2,
        4,
    )


def test_convert_trajectories_to_batch_keeps_nested_forward_inputs():
    trajectories = [
        Trajectory(
            forward_inputs={
                "chains": torch.ones(2, 1, 3),
                "dreamzero_ppo.action_obs": {
                    "state": torch.full((2, 1, 4), 1.0),
                },
            },
            rewards=torch.ones(2, 1),
        ),
        Trajectory(
            forward_inputs={
                "chains": torch.full((2, 1, 3), 2.0),
                "dreamzero_ppo.action_obs": {
                    "state": torch.full((2, 1, 4), 2.0),
                },
            },
            rewards=torch.full((2, 1), 2.0),
        ),
    ]

    batch = convert_trajectories_to_batch(trajectories)

    assert batch["forward_inputs"]["chains"].shape == (2, 2, 3)
    assert batch["forward_inputs"]["dreamzero_ppo.action_obs"]["state"].shape == (
        2,
        2,
        4,
    )
    torch.testing.assert_close(
        batch["forward_inputs"]["dreamzero_ppo.action_obs"]["state"][:, 0],
        torch.full((2, 4), 1.0),
    )
    torch.testing.assert_close(
        batch["forward_inputs"]["dreamzero_ppo.action_obs"]["state"][:, 1],
        torch.full((2, 4), 2.0),
    )
