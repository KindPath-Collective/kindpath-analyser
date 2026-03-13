"""
CLI runner for KindPress constants engine.

Runs one-shot or continuous scans to evolve the constants baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from kindpress.constants_engine import (
    ScanConfig,
    load_snapshot,
    save_snapshot,
    scan_constants,
    snapshot_to_dict,
)


DEFAULT_ROOTS = [
    "/Users/sam/dev/KindPath-Collective",
    "/Users/sam/kindai",
]


def _resolve_roots(raw: str | None) -> list[str]:
    if raw:
        roots = [p.strip() for p in raw.split(":") if p.strip()]
        if roots:
            return roots
    env_roots = os.environ.get("KINDPRESS_CONSTANT_SCAN_ROOTS", "")
    if env_roots:
        roots = [p.strip() for p in env_roots.split(":") if p.strip()]
        if roots:
            return roots
    return DEFAULT_ROOTS


def run_once(args) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    latest_path = output_dir / "constants.latest.json"
    previous = load_snapshot(latest_path)

    config = ScanConfig(
        roots=_resolve_roots(args.roots),
        max_files=args.max_files,
        max_file_size_bytes=args.max_file_size,
        min_occurrences=args.min_occurrences,
        max_constants=args.max_constants,
    )

    snapshot = scan_constants(config, previous_snapshot=previous)
    written = save_snapshot(snapshot, output_dir)

    summary = {
        "snapshot_id": snapshot.snapshot_id,
        "generated_at": snapshot.generated_at,
        "constants_count": snapshot.constants_count,
        "total_files_scanned": snapshot.total_files_scanned,
        "total_tokens_seen": snapshot.total_tokens_seen,
        "constants_key_fingerprint": snapshot.constants_key_fingerprint,
        "written": written,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KindPress constants discovery engine")
    parser.add_argument("--roots", help="Colon-separated scan roots")
    parser.add_argument("--output-dir", default="kindpress/constants", help="Output directory for snapshots")
    parser.add_argument("--max-files", type=int, default=15000)
    parser.add_argument("--max-file-size", type=int, default=2_000_000)
    parser.add_argument("--min-occurrences", type=int, default=4)
    parser.add_argument("--max-constants", type=int, default=2000)
    parser.add_argument("--watch", action="store_true", help="Run continuously")
    parser.add_argument("--interval-seconds", type=int, default=900)
    args = parser.parse_args()

    if not args.watch:
        summary = run_once(args)
        print(json.dumps(summary, indent=2))
        return

    print("[constants_runner] watch mode enabled")
    while True:
        summary = run_once(args)
        print(json.dumps(summary, indent=2))
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    main()
