"""Reproducible microbenchmark for vectorized physics stepping."""
import argparse
import statistics
import time
import sys
from pathlib import Path

# Allow direct execution from a source checkout without requiring installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lawless.physics.constants import PhysicsConfig
from lawless.physics.state import make_state
from lawless.physics.simulator import step


def benchmark(device: torch.device, agents: int, steps: int, repeats: int, dtype: torch.dtype):
    cfg = PhysicsConfig()
    state = make_state(agents, device=device, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(1234)
    actions = torch.rand((agents, 2), generator=gen, device=device, dtype=dtype) * 2 - 1
    gravity = torch.tensor([0.0, 4.0], device=device, dtype=dtype)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Warm-up, excluded from measurements.
    for _ in range(min(100, steps)):
        state, _ = step(state, actions, gravity, cfg)
    sync()

    samples = []
    for _ in range(repeats):
        state = make_state(agents, device=device, dtype=dtype)
        sync()
        start = time.perf_counter()
        for _ in range(steps):
            state, _ = step(state, actions, gravity, cfg)
        sync()
        samples.append(time.perf_counter() - start)

    median_s = statistics.median(samples)
    total_agent_steps = agents * steps
    return {
        "device": str(device), "agents": agents, "steps": steps,
        "median_seconds": median_s,
        "physics_steps_per_second": steps / median_s,
        "agent_steps_per_second": total_agent_steps / median_s,
        "samples_seconds": samples,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", type=int, default=[100, 500, 1000, 2500, 5000])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    print(f"PyTorch={torch.__version__} CUDA={torch.version.cuda} dtype={dtype}")
    for device in devices:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for n in args.agents:
            result = benchmark(device, n, args.steps, args.repeats, dtype)
            print(result)


if __name__ == "__main__":
    main()
