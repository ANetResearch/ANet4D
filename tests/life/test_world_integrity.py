import hashlib
import json

import pytest

from awr.life.common.integrity import content_version, seal_derived_world


def record(path, data):
    return dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def package(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    files = {"geometry/data.bin": b"original mesh", "coordinate.json": b'{"frame":"ENU"}'}
    rows = []
    for name, data in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        rows.append(record(name, data))
    world = dict(id="town", contentVersion=content_version(rows), files=rows)
    raw = json.dumps(world).encode()
    (root / "world.json").write_bytes(raw)
    derived = tmp_path / "derived"
    derived.mkdir()
    for name in files:
        p = derived / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(root / name)
    return root, derived, raw, world


def test_derived_geometry_updates_hash_and_version_without_modifying_source(tmp_path):
    root, out, raw, world = package(tmp_path)
    p = out / "geometry/data.bin"
    p.unlink()
    p.write_bytes(b"clean geometry")
    new = seal_derived_world(out, raw, world, changed={"geometry/data.bin"})
    assert new["contentVersion"] != world["contentVersion"]
    assert new["contentVersion"] == content_version(new["files"])
    assert next(r for r in new["files"] if r["path"] == "geometry/data.bin") == record(
        "geometry/data.bin", b"clean geometry"
    )
    assert (root / "geometry/data.bin").read_bytes() == b"original mesh"
    assert (root / "world.json").read_bytes() == raw
    assert new["derivedFrom"]["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads((out / "world.json").read_bytes()) == new
    assert world["files"][0]["sha256"] == record("geometry/data.bin", b"original mesh")["sha256"]


def test_unrelated_changed_input_is_not_silently_rehashed(tmp_path):
    root, out, raw, world = package(tmp_path)
    (root / "coordinate.json").write_bytes(b"changed upstream")
    with pytest.raises(ValueError, match="Undeclared content change"):
        seal_derived_world(out, raw, world, changed={"geometry/data.bin"})
    assert not (out / "world.json").exists()


def test_stale_nested_appearance_index_prevents_sealing(tmp_path):
    _, out, raw, world = package(tmp_path)
    app = out / "visual/mesh/appearance"
    app.mkdir(parents=True)
    (app / "appearance.glb").write_bytes(b"clean mesh")
    (app / "index.json").write_text(json.dumps({"glb": dict(href="appearance.glb", bytes=9, sha256="0" * 64)}))
    changed = {"visual/mesh/appearance/index.json", "visual/mesh/appearance/appearance.glb"}
    with pytest.raises(ValueError, match="Appearance index"):
        seal_derived_world(out, raw, world, changed=changed)
    index = {"glb": {"href": "appearance.glb", **{k: v for k, v in record("", b"clean mesh").items() if k != "path"}}}
    (app / "index.json").write_text(json.dumps(index))
    world["layers"] = [{"id": "mesh.appearance", "bytes": 999}]
    new = seal_derived_world(out, raw, world, changed=changed)
    assert new["layers"][0]["bytes"] == 10


def test_removed_pointcloud_content_must_really_be_removed(tmp_path):
    _, out, raw, world = package(tmp_path)
    with pytest.raises(ValueError, match="still present"):
        seal_derived_world(out, raw, world, changed=set(), removed={"geometry/data.bin"})
    (out / "geometry/data.bin").unlink()
    new = seal_derived_world(out, raw, world, changed=set(), removed={"geometry/data.bin"})
    assert len(new["files"]) == 1
    assert new["derivedFrom"]["removed_paths"] == ["geometry/data.bin"]


def test_manifest_symlink_never_replaces_source(tmp_path):
    root, out, raw, world = package(tmp_path)
    (out / "world.json").symlink_to(root / "world.json")
    with pytest.raises(ValueError, match="symlink"):
        seal_derived_world(out, raw, world, changed=set())
    assert (root / "world.json").read_bytes() == raw


@pytest.mark.parametrize("path", ["../outside", "/absolute", "world.json", "a/../b", "a//b"])
def test_unsafe_output_declarations_rejected(tmp_path, path):
    _, out, raw, world = package(tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        seal_derived_world(out, raw, world, changed={path})


def test_content_version_is_order_independent_and_matches_upstream_contract():
    files = [record("z", b"last"), record("a", b"first")]
    expected = hashlib.sha256("".join(f"{r['path']}:{r['sha256']}\n" for r in reversed(files)).encode()).hexdigest()[
        :12
    ]
    assert content_version(files) == expected == content_version(list(reversed(files)))
