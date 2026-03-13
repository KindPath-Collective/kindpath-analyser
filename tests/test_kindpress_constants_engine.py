import json
from pathlib import Path

from kindpress.constants_engine import (
    ScanConfig,
    derive_constants_key_material,
    load_snapshot,
    save_snapshot,
    scan_constants,
    verify_provenance_chain,
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


def test_snapshot_write_creates_provenance_chain(tmp_path: Path):
    root = tmp_path / "workspace"
    out = tmp_path / "out"
    root.mkdir(parents=True)

    (root / "a.txt").write_text("kindpath_signal kindpath_signal stable_anchor", encoding="utf-8")
    first = scan_constants(ScanConfig(roots=[str(root)], min_occurrences=1, max_files=100))
    written_one = save_snapshot(first, out)

    (root / "b.txt").write_text("kindpath_signal stable_anchor social_contract", encoding="utf-8")
    second = scan_constants(
        ScanConfig(roots=[str(root)], min_occurrences=1, max_files=100),
        previous_snapshot=load_snapshot(Path(written_one["latest"])),
    )
    written_two = save_snapshot(second, out)

    provenance_path = Path(written_two["provenance"])
    assert provenance_path.exists()

    lines = [line for line in provenance_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    verify = verify_provenance_chain(provenance_path)
    assert verify["ok"] is True
    assert verify["events_verified"] == 2


def test_token_normalisation_and_stopword_filtering(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir(parents=True)

    (root / "mix.txt").write_text(
        "Audit audit AUDIT kindpath_mode kindpath_mode description version optional",
        encoding="utf-8",
    )

    snapshot = scan_constants(ScanConfig(roots=[str(root)], min_occurrences=1, max_files=100))
    tokens = {c.token for c in snapshot.constants}

    # General tokens are normalised to lowercase and deduplicated.
    assert "audit" in tokens
    assert "Audit" not in tokens

    # Low-value documentation terms are filtered from constants.
    assert "description" not in tokens
    assert "version" not in tokens
    assert "optional" not in tokens
