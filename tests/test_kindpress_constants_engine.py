import json
from pathlib import Path

from kindpress.constants_engine import (
    ScanConfig,
    derive_constants_key_material,
    load_snapshot,
    save_snapshot,
    scan_constants,
)


def test_scan_constants_finds_repeated_tokens(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir(parents=True)

    (root / "a.json").write_text(
        json.dumps({"kindpath_mode": "community_sync", "status": "active"}),
        encoding="utf-8",
    )
    (root / "b.json").write_text(
        json.dumps({"kindpath_mode": "community_sync", "status": "active"}),
        encoding="utf-8",
    )
    (root / "c.md").write_text(
        "kindpath_mode active community_sync\nkindpath_mode active",
        encoding="utf-8",
    )

    snapshot = scan_constants(
        ScanConfig(
            roots=[str(root)],
            max_files=500,
            min_occurrences=2,
            max_constants=100,
        )
    )

    assert snapshot.constants_count > 0
    tokens = {c.token for c in snapshot.constants}
    assert "kindpath_mode" in tokens
    assert "active" in tokens


def test_constants_key_derivation_is_deterministic(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    (root / "x.md").write_text("alpha alpha alpha beta beta", encoding="utf-8")
    (root / "y.md").write_text("alpha beta gamma", encoding="utf-8")

    snapshot = scan_constants(
        ScanConfig(roots=[str(root)], min_occurrences=2, max_files=100)
    )

    k1 = derive_constants_key_material(snapshot.constants)
    k2 = derive_constants_key_material(snapshot.constants)

    assert isinstance(k1, bytes)
    assert len(k1) == 32
    assert k1 == k2


def test_snapshot_save_and_load_roundtrip(tmp_path: Path):
    root = tmp_path / "workspace"
    out = tmp_path / "out"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("community community constants constants", encoding="utf-8")

    snapshot = scan_constants(
        ScanConfig(roots=[str(root)], min_occurrences=2, max_files=100)
    )
    written = save_snapshot(snapshot, out)

    latest = Path(written["latest"])
    assert latest.exists()

    loaded = load_snapshot(latest)
    assert loaded is not None
    assert loaded["snapshot_id"] == snapshot.snapshot_id
    assert loaded["constants_count"] == snapshot.constants_count
