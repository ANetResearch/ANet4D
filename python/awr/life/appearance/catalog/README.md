# Real actor assets

The canonical event references an asset ID **and its SHA-256**, not a mutable filename.
The renderer verifies geometry and texture hashes before loading. Rebuilding the catalog
archives previous specifications in `history/` and buffers by hash in `compiled/`, so
an existing tape does not silently acquire new geometry.

Sources:

- [MakeHuman official CC0 system assets](https://static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html): body, skin, clothing, hair, eyes and eyebrows. The base mesh, default rig, weights and body morph targets are pinned to MakeHuman commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`. See `humans.lock.json` for individual hashes and URLs. This uses CC0 data; it does not embed the AGPL MakeHuman application.
- [Innerscene original CC0 vehicles](https://www.innerscene.com/tools/library/3d-parts/graphite-four-door-sedan-eed96d12): seven different mesh families (sedan, hatchback, SUV, van, pickup, bus, truck). `vehicles.lock.json` links each original license/download page and records the downloaded bytes' hash.

Build on the documented native environment:

```sh
scripts/deploy/run_native.sh scripts/assets/fetch_humans.py
scripts/deploy/run_native.sh scripts/assets/fetch_assets.py
scripts/deploy/run_native.sh scripts/assets/build_assets.py
```

Downloaded source packs and compiled buffers are intentionally not Git objects. Keep
`catalog.json`, both lock files and referenced `history/*.json` with the deployment.
The source fetch scripts are acquisition tools: they record what was fetched. The
renderer performs verification against the catalog; do not refresh lock files blindly
when reproducing a published event.

Human variants combine actual morphological targets, skin, fitted clothing, hair,
eyes and skeletal weights. Clothes are fitted through the authored MHCLO barycentric
bindings. The walk cycle uses two-link leg IK, then linear blend skinning. Its phase is
derived from recorded arc length and does not advance during waiting or hidden
transitions. These are procedural locomotion poses, not motion capture or a whole-body
physics simulation. Cloth/hair dynamics and detailed boarding gestures are not included.

Vehicle geometry uses authored physical dimensions for both simulation collision boxes
and rendering. Wheel rotation derives from arc length and tire radius. Road pitch/roll
are stored in tape v2. Paint and glass currently use textured diffuse shading, with
geometry-derived actor shadows; this is not a fully calibrated PBR sensor model.

Testbed extension: `scripts/assets/fetch_street.py` downloads six CC0 roadside
assets; `street.lock.json` records each source and byte hash. Sources include
[the city bicycle](https://www.innerscene.com/tools/library/3d-parts/city-bicycle-58a2a324),
[park bench](https://www.innerscene.com/tools/library/3d-parts/avenue-slatted-park-bench-bde5a75b),
[bin](https://www.innerscene.com/tools/library/3d-parts/slatted-park-litter-bin-6334976f),
[hydrant](https://www.innerscene.com/tools/library/3d-parts/fire-hydrant-6f53bb5d),
[bollard](https://www.innerscene.com/tools/library/3d-parts/steel-security-bollard-5e948999), and
[bicycle rack](https://www.innerscene.com/tools/library/3d-parts/three-loop-bicycle-rack-3a1b8cc5).
Run the street fetch before `build_assets.py` to include these meshes. The catalog
now has twelve human combinations, seven vehicles and six static objects.
Static bicycles have no riding animation or contact/stability solver.
