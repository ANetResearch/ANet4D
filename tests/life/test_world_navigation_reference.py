import json

import pytest

from awr.life.civil.world_reference import navigation_hashes


def write_nav(root, profile, digest):
    folder = root / "geometry/nav"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (profile + ".json")).write_text(json.dumps({"bin": {"sha256": digest}}))


def test_person_only_package_omits_vehicle_without_fabricating_hash(tmp_path):
    write_nav(tmp_path, "person", "a" * 64)
    assert navigation_hashes(tmp_path) == {"person": "a" * 64}
    with pytest.raises(FileNotFoundError):
        navigation_hashes(tmp_path, require_vehicle=True)


def test_existing_optional_profile_remains_bound(tmp_path):
    write_nav(tmp_path, "person", "a" * 64)
    write_nav(tmp_path, "vehicle", "b" * 64)
    assert (
        navigation_hashes(tmp_path)
        == navigation_hashes(tmp_path, require_vehicle=True)
        == {"person": "a" * 64, "vehicle": "b" * 64}
    )


@pytest.mark.parametrize("digest", [None, "", "not-a-hash"])
def test_missing_or_malformed_digest_is_not_silently_optional(tmp_path, digest):
    write_nav(tmp_path, "person", "a" * 64)
    write_nav(tmp_path, "vehicle", digest)
    with pytest.raises(ValueError, match="navigation digest"):
        navigation_hashes(tmp_path)


def test_person_navigation_remains_required(tmp_path):
    write_nav(tmp_path, "vehicle", "b" * 64)
    with pytest.raises(FileNotFoundError):
        navigation_hashes(tmp_path)
