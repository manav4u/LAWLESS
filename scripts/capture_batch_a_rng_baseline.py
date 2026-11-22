# scripts/capture_batch_a_rng_baseline.py

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from lawless.world.config import WorldConfig
from lawless.world.generator import generate_obstacle_world


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "26d663ad04472c4201099bd8693e4e3a80518c0c"
GENERATOR_PATH = "lawless/world/generator.py"
EXPECTED_EXECUTION_COMMIT = (
    "c464122f6df9334c36662907c8f332fcb45527f8"
)
import hashlib
import re

def fixture_path_for_torch_build(torch_version: str) -> Path:
    # Keep a readable build label and add a hash to avoid filename collisions.
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", torch_version)
    digest = hashlib.sha256(torch_version.encode("utf-8")).hexdigest()[:12]

    return (
        REPOSITORY_ROOT
        / "tests"
        / "fixtures"
        / f"batch_a_rng_baseline__{readable}__{digest}.json"
    )

SEEDS = (7, 42, 2026)
DTYPES = (torch.float32, torch.float64)


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def main():
    execution_commit = git("rev-parse", "HEAD")
    if execution_commit != EXPECTED_EXECUTION_COMMIT:
        raise RuntimeError(
            f"Expected execution commit {EXPECTED_EXECUTION_COMMIT}, "
            f"got {execution_commit}"
        )

    # Ensure the generator has no uncommitted modifications.
    if git("status", "--porcelain", "--", GENERATOR_PATH):
        raise RuntimeError(
            f"{GENERATOR_PATH} has uncommitted changes"
        )

    reference_blob = git(
        "rev-parse",
        f"{SOURCE_COMMIT}:{GENERATOR_PATH}",
    )
    execution_blob = git(
        "rev-parse",
        f"HEAD:{GENERATOR_PATH}",
    )
    if reference_blob != execution_blob:
        raise RuntimeError(
            "Current generator.py does not match the validated "
            f"{SOURCE_COMMIT} source blob"
        )

    config = WorldConfig()
    config.validate()

    records = []

    for dtype in DTYPES:
        for seed in SEEDS:
            world = generate_obstacle_world(
                config,
                seed=seed,
                device="cpu",
                dtype=dtype,
            )

            records.append(
                {
                    "seed": seed,
                    "dtype": str(dtype),
                    "centers_shape": list(world.obstacles.centers.shape),
                    "radii_shape": list(world.obstacles.radii.shape),
                    # Hex strings preserve exact Python-float round trips.
                    "centers_hex": [
                        float(value).hex()
                        for value in world.obstacles.centers
                        .contiguous()
                        .view(-1)
                        .tolist()
                    ],
                    "radii_hex": [
                        float(value).hex()
                        for value in world.obstacles.radii
                        .contiguous()
                        .tolist()
                    ],
                }
            )

    fixture = {
        "purpose": (
            "Golden outputs from the validated Batch A obstacle generator"
        ),
        "source_commit": SOURCE_COMMIT,
        "source_path": GENERATOR_PATH,
        "source_blob": reference_blob,
        "execution_commit": execution_commit,
        "torch_version": torch.__version__,
        "device": "cpu",
        "config": asdict(config),
        "seeds": list(SEEDS),
        "dtypes": [str(dtype) for dtype in DTYPES],
        "records": records,
    }

    output_path = fixture_path_for_torch_build(torch.__version__)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Torch-build baseline: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Captured {len(records)} reference cases")
    print(f"Source commit: {SOURCE_COMMIT}")
    print(f"Source blob:   {reference_blob}")
    print(f"Executed at:   {execution_commit}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"Fixture:       {output_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
