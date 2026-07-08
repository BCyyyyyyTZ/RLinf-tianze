import torch

from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_split_rollout_result_splits_nested_forward_inputs():
    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    rollout_result = RolloutResult(
        actions=torch.arange(6).reshape(3, 2),
        prev_logprobs=torch.arange(3, dtype=torch.float32).reshape(3, 1),
        prev_values=torch.arange(3, dtype=torch.float32).reshape(3, 1),
        forward_inputs={
            "chains": torch.arange(24).reshape(3, 2, 4),
            "dreamzero_ppo.action_obs": {
                "state": torch.arange(12).reshape(3, 4),
            },
        },
        versions=torch.zeros(3, 1),
    )

    split_results = worker._split_rollout_result(rollout_result, [2, 1])

    assert len(split_results) == 2
    assert split_results[0].actions.shape == (2, 2)
    assert split_results[1].actions.shape == (1, 2)
    assert split_results[0].forward_inputs["chains"].shape == (2, 2, 4)
    assert split_results[1].forward_inputs["chains"].shape == (1, 2, 4)
    assert split_results[0].forward_inputs["dreamzero_ppo.action_obs"][
        "state"
    ].shape == (2, 4)
    assert torch.equal(
        split_results[1].forward_inputs["dreamzero_ppo.action_obs"]["state"],
        torch.arange(8, 12).reshape(1, 4),
    )
