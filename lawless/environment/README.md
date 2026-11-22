# LAWLESS P1.7 environment API

This package composes the accepted P0 physics, generated-world, episode-state,
collision, core, extraction, lidar, and observation modules. It does not modify
those modules and does not introduce controllers, evolution, or fitness scoring.

## Public API

```python
from lawless.environment import reset, step

environment, observations, reset_info = reset(
    config,
    num_worlds=B,
    num_agents=N,
)

(
    environment,
    observations,
    rewards,
    terminated,
    truncated,
    info,
) = step(environment, actions)  # actions: [B, N, 2]
```

- Observations: `[B, N, 23]`
- Rewards: `[B, N]`, physics floating dtype
- `terminated`, `truncated`: `[B, N]`, bool
- Actions must have exact shape `[B, N, 2]` and be finite. Finite values
  outside `[-1, 1]` are clipped according to the accepted P0 contract.
- `step` is functional: it returns a new `EnvironmentState`; it does not mutate
  the supplied state.

## Paired-world topology and spawn policy

`B` is a batch of independent worlds. Every world owns a separate
`GeneratedWorld`, `AgentEpisodeState`, condition sample, clock, and completion
state.

Each world is generated with exactly one spawn slot. `AgentEpisodeState` is
initialized with an explicit `spawn_indices` tensor of shape `[N]`, filled
with zero. Thus all N candidates start at the same position and share the
world's static obstacles, extraction center, and core geometry.

These are independent candidate rollouts over shared static geometry, not
simultaneous competing agents. There is no agent-agent collision, and each
candidate owns an independent core-collection mask. One candidate collecting
a core does not deplete it for another candidate. Distinct-start experiments
are not part of the P1.7 default.

Because spawn count is fixed at one, changing N does not change generator
draw count or the extraction/core geometry. Changing B adds independently
seeded worlds.

## Deterministic random streams

For root seed `R`, zero-based world index `i`, and stream identifier `S`:

```text
mixed = (R + 0x9E3779B97F4A7C15 * (i + 1) + S) mod 2**64
seed  = splitmix64(mixed) & (2**63 - 1)
```

The SplitMix64 finalizer adds the golden-ratio constant once more, then applies
the standard 64-bit xor-shift/multiply finalizer. The world stream identifier
is `0x574F524C445F5631`; the condition stream identifier is
`0x434F4E445F5631`.

`generate_world` uses its existing private CPU generator seeded by the world
seed. A separate private CPU generator seeded by the condition seed samples
initial gravity direction, linear damping, and the ±90-degree shift direction.
Neither stream consumes or changes global RNG state. Seeds depend on root seed
and world index, not B, N, or iteration order.

## Transition, reward, and completion semantics

A transition runs P0, endpoint collision resolution, finite-difference
acceleration measurement from the collision-resolved velocity, core collection,
extraction, and collision-failure annotation. The recorded action is the
finite action after conversion and P0 clipping. P0 acceleration diagnostics
are not used as measured acceleration.

`rewards[b, n]` is exactly the raw newly-extracted indicator for that
candidate on that transition (1 or 0). It is not a shaped reward or final V1
fitness. Core counts, actual energy restored/spent, collision reports,
failure annotations, energy deaths, extraction events, and timeout masks are
separate `info` entries.

The transition at the shift boundary uses the original gravity. The shift is
applied after that transition; the next transition uses rotated gravity.
Diagnostics expose `gravity_used`, `gravity_next`, `shift_applied_this_step`,
and cumulative `shift_applied`. A zero-time shift is applied at reset.

Extraction is checked before timeout truncation and remains eligible at the
inclusive deadline. At the deadline, physically alive agents become truncated
without changing `physics.alive` or adding a failure annotation. A world also
finishes early when every candidate is physically inactive. A completed world
does not advance while other worlds continue; its observations and
`terminated`/`truncated` masks remain stable. When all worlds are complete,
`step` rejects another call before changing state.

The full action tensor is shape- and finiteness-validated before any world is
advanced.
