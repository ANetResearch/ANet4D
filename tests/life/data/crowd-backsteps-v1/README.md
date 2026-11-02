# Space-aware retreat selection at the population failure

The population-handoff-v1 diagnostic stopped at 372.2 s because person-023
remained stalled for 60.1 s. This folder preserves a geometry probe of its
actual final frame, not a resumed simulation or a release witness.

`probe.py.txt` ran against that immutable build's walking/body implementation,
measured navigation, SUMO permissions and original human assets. The separate
`input-verification.json` confirms those inputs against the build manifest and
actor table. `geometry-report.json` binds the captured `geometry.npz` by SHA256.
Other bodies are frozen at their observed poses during this probe; current
activity phases and active retreat controllers were not captured by that old
failure recorder. Eligibility in the fixture test is therefore diagnostic.

Two concrete selection defects are reproduced:

* Person-023 has a clear retreat, but its preferred higher-index peer 217 has
  no static-clear direction. Fixed identifier priority prevented 023 yielding.
* Person-196's first static-clear retreat intersects person-102. Another
  candidate direction clears the observed bodies, while peer 232 has no
  static-clear direction. Choosing the first static-clear direction repeatedly
  proposed an unusable movement.

The actual grounded loop now selects short retreats against a common observed
frame, tries another eligible peer when the preferred one has no space, and
checks all candidate accelerating body sweeps against nearby observed bodies.
Active and newly selected retreats reserve disjoint swept regions. It retains
the original activity destination, static walking map, speed, distance, dynamic
collision/support checks, timeout and measured-pose handoff. This proposal
check does not predict every future neighbour motion; the normal per-tick
checks remain authoritative. No teleport, reduced body size or relaxed gate.

Validation: 58 retreat/navigation/activity/driver/vehicle/support tests passed,
plus 29 body/walking/network/capsule/static-walk tests. The captured failure
tests demonstrate feasible alternatives and reservation consistency, not that
the full 300-person simulation is now deadlock-free. A fresh unchanged-seed
1500 s population recording is required. The component population configuration
still has six legacy trip cars; it is not acceptance of the final fleet, which
requires at least 30 distinct recurrent cars with real driver activity chains.
