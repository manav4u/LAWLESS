# tests/test_p1_batch_a_rng_regression.py

import json
from pathlib import Path

import pytest
import torch

from lawless.world.config import WorldConfig
from lawless.world.generator import generate_obstacle_world


FIXTURE_DIR = Path(__file__).parent / "fixtures"

EXPECTED_SOURCE_COMMIT = (
    "26d663ad04472c4201099bd8693e4e3a80518c0c"
)
EXPECTED_EXECUTION_COMMIT = (
    "c464122f6df9334c36662907c8f332fcb45527f8"
)
EXPECTED_SOURCE_PATH = "lawless/world/generator.py"
EXPECTED_SOURCE_BLOB = (
    "f741a4efb66573208cb65ba4989c778e2fed39ec"
)


def _load_matching_fixture():
    candidates = sorted(
        FIXTURE_DIR.glob("batch_a_rng_baseline__*.json")
    )

    matching = []
    for path in candidates:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if fixture.get("torch_version") == torch.__version__:
            matching.append((path, fixture))

    if not matching:
        pytest.skip(
            "No Batch A golden RNG fixture matches this exact PyTorch "
            f"build ({torch.__version__}); cross-version bitwise RNG "
            "identity is not promised. Capture and commit a baseline "
            "for this build before requiring this regression check."
        )

    assert len(matching) == 1, (
        "Expected exactly one immutable Batch A fixture for PyTorch "
        f"{torch.__version__}, found: "
        + ", ".join(path.name for path, _ in matching)
    )

    return matching[0]


def _tensor_from_hex(values, shape, dtype):
    result = torch.tensor(
        [float.fromhex(value) for value in values],
        dtype=dtype,
        device="cpu",
    )
    return result.reshape(shape)


def test_batch_a_generator_matches_matching_torch_build_baseline():
    fixture_path, fixture = _load_matching_fixture()

    # Exact PyTorch build selection: never compare against another build.
    assert fixture["torch_version"] == torch.__version__

    # Capture-time Git verification established that the generator blob
    # at both commits was identical. Assert that immutable provenance here
    # without requiring those historical Git objects in a shallow checkout.
    assert fixture["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert fixture["execution_commit"] == EXPECTED_EXECUTION_COMMIT
    assert fixture["source_path"] == EXPECTED_SOURCE_PATH
    assert fixture["source_blob"] == EXPECTED_SOURCE_BLOB
    assert fixture["device"] == "cpu"

    assert fixture["seeds"] == [7, 42, 2026]
    assert fixture["dtypes"] == ["torch.float32", "torch.float64"]

    config = WorldConfig(**fixture["config"])
    config.validate()

    expected_dtype_names = {
        torch.float32: "torch.float32",
        torch.float64: "torch.float64",
    }

    assert len(fixture["records"]) == 6

    for record in fixture["records"]:
        dtype = {
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
        }[record["dtype"]]

        expected_centers = _tensor_from_hex(
            record["centers_hex"],
            record["centers_shape"],
            dtype,
        )
        expected_radii = _tensor_from_hex(
            record["radii_hex"],
            record["radii_shape"],
            dtype,
        )

        actual = generate_obstacle_world(
            config,
            seed=record["seed"],
            device="cpu",
            dtype=dtype,
        )

        assert actual.obstacles.centers.dtype == dtype
        assert actual.obstacles.radii.dtype == dtype
        assert actual.obstacles.centers.device.type == "cpu"
        assert actual.obstacles.radii.device.type == "cpu"

        assert record["dtype"] == expected_dtype_names[dtype]

        # Exact comparison is intentional and applies only to the
        # exact PyTorch build identified by the selected fixture.
        assert torch.equal(
            actual.obstacles.centers,
            expected_centers,
        ), (
            f"Center regression for seed={record['seed']}, "
            f"dtype={dtype}, fixture={fixture_path.name}"
        )

        assert torch.equal(
            actual.obstacles.radii,
            expected_radii,
        ), (
            f"Radius regression for seed={record['seed']}, "
            f"dtype={dtype}, fixture={fixture_path.name}"
        )
