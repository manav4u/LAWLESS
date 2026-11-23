
"""Bounded CPU/CUDA observation equivalence through the public P1 API.

This test checks that equal-seed resets produce equivalent initial layouts
and state on CPU and CUDA before comparing observations from fixed actions.
It does not compare independently generated CPU and CUDA RNG streams.
"""

from dataclasses import fields, is_dataclass
import math

import pytest
import torch

from lawless.environment import reset, step
from lawless.world.config import EnvironmentConfig, EpisodeConfig


# Retain the tolerances used by the existing CPU batch-equivalence test.
# The observation assembler uses float64 working tensors but returns the
# configured physics dtype. CUDA physics and sensor operations can differ
# slightly from CPU floating-point execution.
_RTOL = 1.0e-5
_ATOL = 1.0e-6

_NUM_WORLDS = 2
_NUM_AGENTS = 2
_NUM_STEPS = 3
_SEED = 710


def _config():
    """Use the existing P1 batch-equivalence fixture configuration."""
    config = EnvironmentConfig(seed=_SEED)
    config.world.width_m = 1000.0
    config.world.height_m = 1000.0
    config.world.obstacle_count = 2
    config.objectives.core_count = 2
    config.episode = EpisodeConfig(
        episode_duration_s=0.5,
        shift_time_s=0.2,
        dt_s=0.1,
    )
    config.validate()
    return config


def _fixed_actions():
    """Return three explicit [B, N, 2] action tensors in float32."""
    return torch.tensor(
        [
            [
                [[0.10, -0.20], [0.30, 0.05]],
                [[-0.15, 0.25], [0.05, -0.10]],
            ],
            [
                [[-0.20, 0.10], [0.15, 0.20]],
                [[0.25, -0.05], [-0.10, 0.15]],
            ],
            [
                [[0.05, 0.15], [-0.25, 0.10]],
                [[0.20, -0.15], [0.10, 0.25]],
            ],
        ],
        dtype=torch.float32,
        device="cpu",
    )


def _assert_tree_equivalent(cpu_value, cuda_value, path="root"):
    """Recursively validate structure, tensor metadata, finiteness and values.

    Floating tensors use the test tolerances. Boolean and integer tensors
    must match exactly. Device type must be CPU on the left and CUDA on
    the right. Tensor dtype and shape must match exactly.
    """
    if isinstance(cpu_value, torch.Tensor):
        assert isinstance(cuda_value, torch.Tensor), path
        assert cpu_value.shape == cuda_value.shape, path
        assert cpu_value.dtype == cuda_value.dtype, path
        assert cpu_value.device.type == "cpu", path
        assert cuda_value.device.type == "cuda", path

        if cpu_value.is_floating_point() or cpu_value.is_complex():
            assert torch.isfinite(cpu_value).all().item(), path
            assert torch.isfinite(cuda_value).all().item(), path
            torch.testing.assert_close(
                cpu_value,
                cuda_value.to(device="cpu"),
                rtol=_RTOL,
                atol=_ATOL,
                msg=lambda message: f"{path}: {message}",
            )
        else:
            assert torch.equal(
                cpu_value,
                cuda_value.to(device="cpu"),
            ), path
        return

    if isinstance(cpu_value, dict):
        assert isinstance(cuda_value, dict), path
        assert cpu_value.keys() == cuda_value.keys(), path
        for key in cpu_value:
            _assert_tree_equivalent(
                cpu_value[key],
                cuda_value[key],
                f"{path}.{key}",
            )
        return

    if isinstance(cpu_value, (tuple, list)):
        assert type(cpu_value) is type(cuda_value), path
        assert len(cpu_value) == len(cuda_value), path
        for index, (cpu_item, cuda_item) in enumerate(
            zip(cpu_value, cuda_value)
        ):
            _assert_tree_equivalent(
                cpu_item,
                cuda_item,
                f"{path}[{index}]",
            )
        return

    if is_dataclass(cpu_value) and not isinstance(cpu_value, type):
        assert type(cpu_value) is type(cuda_value), path
        for field in fields(cpu_value):
            _assert_tree_equivalent(
                getattr(cpu_value, field.name),
                getattr(cuda_value, field.name),
                f"{path}.{field.name}",
            )
        return

    if hasattr(cpu_value, "__dict__"):
        assert type(cpu_value) is type(cuda_value), path
        assert vars(cpu_value).keys() == vars(cuda_value).keys(), path
        for key, cpu_item in vars(cpu_value).items():
            _assert_tree_equivalent(
                cpu_item,
                vars(cuda_value)[key],
                f"{path}.{key}",
            )
        return

    if isinstance(cpu_value, float):
        assert math.isfinite(cpu_value), path
        assert isinstance(cuda_value, float), path
        assert math.isfinite(cuda_value), path
        assert math.isclose(
            cpu_value,
            cuda_value,
            rel_tol=_RTOL,
            abs_tol=_ATOL,
        ), path
        return

    assert type(cpu_value) is type(cuda_value), path
    assert cpu_value == cuda_value, path


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA observation equivalence requires an available CUDA device",
)
def test_cpu_cuda_reset_and_public_step_observations_are_equivalent():
    """Compare equal-seed layouts and observations under identical actions."""
    cuda_device = torch.device("cuda:0")
    assert torch.cuda.device_count() >= 1

    config = _config()

    cpu_environment, cpu_observations, cpu_reset_info = reset(
        config,
        num_worlds=_NUM_WORLDS,
        num_agents=_NUM_AGENTS,
        device="cpu",
        dtype=torch.float32,
    )
    cuda_environment, cuda_observations, cuda_reset_info = reset(
        config,
        num_worlds=_NUM_WORLDS,
        num_agents=_NUM_AGENTS,
        device=cuda_device,
        dtype=torch.float32,
    )

    expected_observation_shape = (
        _NUM_WORLDS,
        _NUM_AGENTS,
        23,
    )
    assert tuple(cpu_observations.shape) == expected_observation_shape
    assert tuple(cuda_observations.shape) == expected_observation_shape

    # Verify the reset actually produced equivalent layouts and state.
    # This is essential: equal seed derivation does not, by itself, prove
    # that target-device world generation produced identical geometry.
    _assert_tree_equivalent(
        cpu_environment,
        cuda_environment,
        "reset.environment",
    )
    _assert_tree_equivalent(
        cpu_reset_info,
        cuda_reset_info,
        "reset.info",
    )
    _assert_tree_equivalent(
        cpu_observations,
        cuda_observations,
        "reset.observations",
    )

    # Actions are authored once on CPU and transferred unchanged to CUDA.
    actions_sequence = _fixed_actions()
    assert tuple(actions_sequence.shape) == (
        _NUM_STEPS,
        _NUM_WORLDS,
        _NUM_AGENTS,
        2,
    )

    for step_index in range(_NUM_STEPS):
        cpu_actions = actions_sequence[step_index]
        cuda_actions = cpu_actions.to(device=cuda_device)

        # Confirm that the two public calls receive identical action values.
        assert torch.equal(
            cpu_actions,
            cuda_actions.to(device="cpu"),
        )

        (
            cpu_environment,
            cpu_observations,
            cpu_rewards,
            cpu_terminated,
            cpu_truncated,
            cpu_info,
        ) = step(cpu_environment, cpu_actions)

        (
            cuda_environment,
            cuda_observations,
            cuda_rewards,
            cuda_terminated,
            cuda_truncated,
            cuda_info,
        ) = step(cuda_environment, cuda_actions)

        prefix = f"step[{step_index}]"

        assert tuple(cpu_observations.shape) == expected_observation_shape
        assert tuple(cuda_observations.shape) == expected_observation_shape

        # Includes all public step outputs and the complete resulting
        # environment state, recursively checking structure and tensors.
        _assert_tree_equivalent(
            cpu_observations,
            cuda_observations,
            f"{prefix}.observations",
        )
        _assert_tree_equivalent(
            cpu_rewards,
            cuda_rewards,
            f"{prefix}.rewards",
        )
        _assert_tree_equivalent(
            cpu_terminated,
            cuda_terminated,
            f"{prefix}.terminated",
        )
        _assert_tree_equivalent(
            cpu_truncated,
            cuda_truncated,
            f"{prefix}.truncated",
        )
        _assert_tree_equivalent(
            cpu_info,
            cuda_info,
            f"{prefix}.info",
        )
        _assert_tree_equivalent(
            cpu_environment,
            cuda_environment,
            f"{prefix}.environment",
        )

        # The short fixture must remain active throughout these controlled
        # transitions; this test is not an episode-completion test.
        assert not cpu_environment.all_completed
        assert not cuda_environment.all_completed
