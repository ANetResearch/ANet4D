"""Target life engine (awr.life): residents, vehicles, activities and traffic of a world, compiled to a Canonical Event Tape.

Transplanted from the CHIMERA benchmark (houqiii/chimera, 2026-11-02) with the namespaces rewritten:

- ``awr.life.civil``: scene semantics, population, activity demand and schedules, the build pipeline, QA and tape assembly;
- ``awr.life.motion``: the ``life_v1`` runtime (decisions, walking, driving, boarding, crowd, recording, audits);
- ``awr.life.ambient``: the external solver backend (Eclipse SUMO for vehicles, JuPedSim for pedestrians);
- ``awr.life.envpack``: static environment diagnostics toward the feature admission contract;
- ``awr.life.appearance``: appearance identities, plates, wheels and the asset catalogue;
- ``awr.life.common``: schema identifiers, integrity hashes, admission errors, entity listings.

Repository environment only (shapely, scipy, numba, PyYAML; ``anet4d[life]`` and ``anet4d[life-ext]`` extras). Not part
of the D4 numpy-only kernel closure. Documents written by this package carry ``awr.life.<name>.v<N>`` identifiers; the
``chimera.*`` and ``d4ale.*`` identifiers of records produced before the transplant compare equal through
``awr.life.common.schema.schema_id``.
"""
