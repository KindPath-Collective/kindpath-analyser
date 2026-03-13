"""
KindPath Analyser :: Seedbank Fork Log

The canonical audit trail of every moment the k-universe branched.

Every time the constants bank shifts — through a tag revision, a recomputation,
or a lineage confluence — this log records what changed, when, and how many
records were affected. It answers the question:

    "On this date, k was revised in this way,
     which forked N records into a new universe of interpretation."

This is the missing layer between the tags_registry (per-tag version history)
and the reading_history on individual records (per-record fork history).
The fork log is the cross-cutting audit: the single document of every moment
the archive entered a new state.

It also surfaces doctrinal shifts: tag revisions that originate in kindpath-canon
and propagate downstream through this analyser are visible here as k-universe
branch points. A revision to `fingerprint/generative-error` may reflect a
conceptual refinement in CONSCIOUS_CONTEXTUALISATION.md — the fork log makes
that connection explicit and auditable across the whole organisation.

Five event types are logged:
    "deposit"           — birth of a new record (universe created at rest)
    "tag_revision"      — a tag constant was redefined (k shifted; forks are pending)
    "tag_propagation"   — the revision was flagged across N records (forks materialised)
    "record_fork"       — a single record received a new reading (single-lineage fork)
    "record_confluence" — a confluence reading was synthesised from multiple k-universes

Storage: seedbank/fork_log.json — append-only. Events are never deleted, only accumulated.
A Markdown summary can be exported via export_doctrinal_shifts(), suitable for
committing to .github/FORK_LOG.md as a cross-repo signal.

See: kindpath-canon/CONSCIOUS_CONTEXTUALISATION.md — Fork-and-Retain Principle
"""

import json
import math
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional

_DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "fork_log.json")

LOG_PATH: str = os.environ.get(
    "KINDPATH_FORK_LOG",
    _DEFAULT_LOG_PATH,
)


@dataclass
class ForkEvent:
    """
    One entry in the fork audit log.

    Covers five event types, each marking a distinct kind of k-universe shift.
    The event_type field determines which optional fields are populated:

    deposit:             record_id, filename, new_baseline_version, notes
    tag_revision:        tag_name, old_version, new_version, revision_reason, revision_source
    tag_propagation:     tag_name, new_version, affected_record_count
    record_fork:         record_id, filename, old_baseline_version, new_baseline_version,
                         residue_delta, computation_source
    record_confluence:   record_id, filename, new_baseline_version, residue_delta,
                         parent_reading_ids, num_parents
    """

    event_id: str               # UUID identifying this log entry
    event_type: str             # One of the five event types above
    occurred_at: str            # ISO 8601 timestamp

    # ── Tag events ────────────────────────────────────────────────────────────
    tag_name: Optional[str] = None
    old_version: Optional[int] = None       # Version number before this revision
    new_version: Optional[int] = None       # Version number after this revision
    revision_reason: Optional[str] = None
    revision_source: Optional[str] = None
    affected_record_count: Optional[int] = None  # None = not yet propagated

    # ── Record events ─────────────────────────────────────────────────────────
    record_id: Optional[str] = None
    filename: Optional[str] = None
    old_baseline_version: Optional[str] = None   # None for deposits (no prior universe)
    new_baseline_version: Optional[str] = None
    residue_delta: Optional[float] = None         # None for deposits (no prior to compare)
    computation_source: Optional[str] = None      # "deposit" | "recompute" | "confluent"

    # ── Cross events ──────────────────────────────────────────────────────────
    parent_reading_ids: Optional[list] = None     # computed_at timestamps of parent readings
    num_parents: Optional[int] = None             # Dimensionality of this cross

    # ── Free-text context ─────────────────────────────────────────────────────
    notes: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal I/O — append-only, atomic write
# ─────────────────────────────────────────────────────────────────────────────

def _load_log() -> dict:
    """Load the fork log from disk. Returns empty log structure if not found."""
    if not os.path.exists(LOG_PATH):
        return {"schema_version": 1, "last_updated": None, "events": []}
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"schema_version": 1, "last_updated": None, "events": []}


def _save_log(log: dict) -> None:
    """Write the fork log atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(LOG_PATH)), exist_ok=True)
    log["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = LOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    os.replace(tmp, LOG_PATH)


def _append_event(event: ForkEvent) -> None:
    """Append a single event to the fork log. Never overwrites existing entries."""
    log = _load_log()
    log["events"].append(asdict(event))
    _save_log(log)


# ─────────────────────────────────────────────────────────────────────────────
# Logging calls — called from deposit.py, tags_registry.py, recompute.py
# ─────────────────────────────────────────────────────────────────────────────

def log_deposit(
    record_id: str,
    filename: str,
    baseline_version: str,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
) -> ForkEvent:
    """
    Log the birth of a new record into the seedbank.

    The birth event is the k-universe at rest: no prior reading, no residue delta.
    All future forks descend from this moment — it is the genome checkpoint.
    Every record's multiverse history begins here.
    """
    tag_summary = f"Tags: {', '.join(tags)}" if tags else None
    event = ForkEvent(
        event_id=str(uuid.uuid4()),
        event_type="deposit",
        occurred_at=datetime.now(timezone.utc).isoformat(),
        record_id=record_id,
        filename=filename,
        new_baseline_version=baseline_version,
        computation_source="deposit",
        notes=notes or tag_summary,
    )
    _append_event(event)
    return event


def log_tag_revision(
    tag_name: str,
    old_version: int,
    new_version: int,
    revision_reason: str,
    revision_source: Optional[str] = None,
    notes: Optional[str] = None,
) -> ForkEvent:
    """
    Log a k-constant redefinition — the moment understanding shifted.

    The affected_record_count is not yet known at revision time. It becomes
    concrete when flag_stale_records() runs and log_tag_propagation() is called.
    Together, the tag_revision + tag_propagation pair forms the complete picture:
    what changed and how far it rippled through the archive.
    """
    event = ForkEvent(
        event_id=str(uuid.uuid4()),
        event_type="tag_revision",
        occurred_at=datetime.now(timezone.utc).isoformat(),
        tag_name=tag_name,
        old_version=old_version,
        new_version=new_version,
        revision_reason=revision_reason,
        revision_source=revision_source,
        affected_record_count=None,  # Becomes concrete when flag_stale_records() runs
        notes=notes,
    )
    _append_event(event)
    return event


def log_tag_propagation(
    tag_name: str,
    new_version: int,
    affected_record_count: int,
    notes: Optional[str] = None,
) -> ForkEvent:
    """
    Log the propagation of a tag revision across the archive.

    Called from flag_stale_records() after it has counted the affected records.
    This is the moment the pending fork becomes concrete in the archive:
    N records have been flagged for recomputation — N branches now exist
    between an old universe and a new one awaiting traversal.
    """
    event = ForkEvent(
        event_id=str(uuid.uuid4()),
        event_type="tag_propagation",
        occurred_at=datetime.now(timezone.utc).isoformat(),
        tag_name=tag_name,
        new_version=new_version,
        affected_record_count=affected_record_count,
        notes=notes or f"{affected_record_count} record(s) flagged stale for recomputation.",
    )
    _append_event(event)
    return event


def log_record_fork(
    record_id: str,
    filename: str,
    old_baseline_version: str,
    new_baseline_version: str,
    residue_delta: float,
    computation_source: str = "recompute",
    notes: Optional[str] = None,
) -> ForkEvent:
    """
    Log a single-lineage fork on a record.

    Called from add_reading() when computation_source is not "confluent".
    The residue_delta is the distance between the two k-universes — how much
    the model's understanding of this piece has moved between revisions.
    A large delta signals a piece whose fingerprint is highly sensitive to k.
    A near-zero delta signals a piece whose creative signal is k-invariant.
    """
    event = ForkEvent(
        event_id=str(uuid.uuid4()),
        event_type="record_fork",
        occurred_at=datetime.now(timezone.utc).isoformat(),
        record_id=record_id,
        filename=filename,
        old_baseline_version=old_baseline_version,
        new_baseline_version=new_baseline_version,
        residue_delta=residue_delta,
        computation_source=computation_source,
        notes=notes,
    )
    _append_event(event)
    return event


def log_record_confluence(
    record_id: str,
    filename: str,
    parent_reading_ids: list,
    composite_baseline: str,
    residue_delta: float,
    notes: Optional[str] = None,
) -> ForkEvent:
    """
    Log a lineage confluence reading event.

    Called from add_reading() when computation_source is "confluent".
    The num_parents reflects the dimensionality of this particular confluence —
    the contribution to the effective_n exponent in Φ = km^n.
    Each additional parent lineage contributes log2(num_parents) to n, pushing
    the piece deeper into the non-linear living regime of the field equation.
    """
    num_parents = len(parent_reading_ids)
    # n contribution: each confluence draws from multiple lineages simultaneously
    n_contribution = round(math.log2(num_parents), 4) if num_parents > 1 else 1.0
    event = ForkEvent(
        event_id=str(uuid.uuid4()),
        event_type="record_confluence",
        occurred_at=datetime.now(timezone.utc).isoformat(),
        record_id=record_id,
        filename=filename,
        new_baseline_version=composite_baseline,
        residue_delta=residue_delta,
        computation_source="confluent",
        parent_reading_ids=list(parent_reading_ids),
        num_parents=num_parents,
        notes=notes or f"{num_parents}-way confluence (n contribution: +{n_contribution})",
    )
    _append_event(event)
    return event


# ─────────────────────────────────────────────────────────────────────────────
# Query interface
# ─────────────────────────────────────────────────────────────────────────────

def get_fork_events(
    since: Optional[str] = None,
    until: Optional[str] = None,
    tag: Optional[str] = None,
    record_id: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
) -> List[ForkEvent]:
    """
    Query the fork log with optional filters. Returns most recent events first.

    since / until: ISO 8601 timestamp strings, e.g. "2026-01-01T00:00:00+00:00"
    tag:           filter to events involving this tag name
    record_id:     filter to events involving this specific record
    event_type:    one of "deposit" | "tag_revision" | "tag_propagation" |
                          "record_fork" | "record_confluence"
    limit:         maximum number of events to return

    Examples:
        # All k-constant shifts this month
        get_fork_events(since="2026-03-01", event_type="tag_revision")

        # Full history for one record
        get_fork_events(record_id="abc123...")

        # Every confluence event in the archive
        get_fork_events(event_type="record_confluence")
    """
    log = _load_log()
    events = sorted(
        log.get("events", []),
        key=lambda e: e.get("occurred_at", ""),
        reverse=True,
    )

    results = []
    for raw in events:
        if event_type and raw.get("event_type") != event_type:
            continue
        if tag and raw.get("tag_name") != tag:
            continue
        if record_id and raw.get("record_id") != record_id:
            continue
        occurred = raw.get("occurred_at", "")
        if since and occurred < since:
            continue
        if until and occurred > until:
            continue

        results.append(ForkEvent(
            event_id=raw.get("event_id", ""),
            event_type=raw.get("event_type", ""),
            occurred_at=raw.get("occurred_at", ""),
            tag_name=raw.get("tag_name"),
            old_version=raw.get("old_version"),
            new_version=raw.get("new_version"),
            revision_reason=raw.get("revision_reason"),
            revision_source=raw.get("revision_source"),
            affected_record_count=raw.get("affected_record_count"),
            record_id=raw.get("record_id"),
            filename=raw.get("filename"),
            old_baseline_version=raw.get("old_baseline_version"),
            new_baseline_version=raw.get("new_baseline_version"),
            residue_delta=raw.get("residue_delta"),
            computation_source=raw.get("computation_source"),
            parent_reading_ids=raw.get("parent_reading_ids"),
            num_parents=raw.get("num_parents"),
            notes=raw.get("notes"),
        ))

        if len(results) >= limit:
            break

    return results


def get_k_revision_history() -> List[ForkEvent]:
    """
    Return all tag_revision and tag_propagation events, newest first.

    This is the doctrinal shift log — every moment the constants bank was
    updated and the change propagated forward through the archive. Cross-repo
    note: these revisions often originate in kindpath-canon and arrive here
    via deliberate human decision to update tag definitions. The fork log
    makes that chain of custody visible.
    """
    revisions = get_fork_events(event_type="tag_revision")
    propagations = get_fork_events(event_type="tag_propagation")
    combined = revisions + propagations
    return sorted(combined, key=lambda e: e.occurred_at, reverse=True)


def get_universe_count() -> int:
    """
    Total number of distinct k-universes that have ever existed in this archive.

    Each deposit creates one universe. Each record_fork creates one more.
    Each record_confluence creates one more (synthesised from existing lineages).
    This is the total dimensionality of the multiverse at the current moment.
    It is the upper bound on the effective_n exponent summed across the corpus.
    """
    log = _load_log()
    return sum(
        1 for e in log.get("events", [])
        if e.get("event_type") in ("deposit", "record_fork", "record_confluence")
    )


def get_log_stats() -> dict:
    """
    Summary statistics for the fork log.

    Returns: total event count, counts by type, date range, universe count,
    and the tag most frequently revised (the most active k-constant).
    """
    log = _load_log()
    events = log.get("events", [])
    if not events:
        return {
            "total_events": 0,
            "by_type": {},
            "first_event": None,
            "last_event": None,
            "universe_count": 0,
            "most_revised_tag": None,
        }

    by_type: dict = {}
    tag_revisions: dict = {}
    dates = []

    for e in events:
        et = e.get("event_type", "unknown")
        by_type[et] = by_type.get(et, 0) + 1
        dates.append(e.get("occurred_at", ""))
        if et == "tag_revision" and e.get("tag_name"):
            t = e["tag_name"]
            tag_revisions[t] = tag_revisions.get(t, 0) + 1

    dates_sorted = sorted(d for d in dates if d)
    most_revised = max(tag_revisions, key=tag_revisions.get) if tag_revisions else None

    return {
        "total_events": len(events),
        "by_type": by_type,
        "first_event": dates_sorted[0] if dates_sorted else None,
        "last_event": dates_sorted[-1] if dates_sorted else None,
        "universe_count": get_universe_count(),
        "most_revised_tag": most_revised,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Export — Markdown for cross-repo visibility
# ─────────────────────────────────────────────────────────────────────────────

def export_doctrinal_shifts(output_path: Optional[str] = None) -> str:
    """
    Generate a human-readable Markdown summary of all fork log events.

    Suitable for exporting to .github/FORK_LOG.md as a cross-repo signal —
    making doctrinal k-shifts visible across the entire KindPath organisation.
    Tag revisions that originated in kindpath-canon surface here as concrete
    archive branch points with record counts and residue deltas attached.

    Returns the Markdown string.
    If output_path is provided, also writes the file to that path.

    Example usage:
        from seedbank.fork_log import export_doctrinal_shifts
        export_doctrinal_shifts(".github/FORK_LOG.md")
    """
    log = _load_log()
    events = sorted(
        log.get("events", []),
        key=lambda e: e.get("occurred_at", ""),
        reverse=True,
    )
    stats = get_log_stats()

    lines = [
        "# KindPath Fork Log — Archive of k-Universe Branch Points",
        "",
        "> **What is this?**  ",
        "> Every time the constants bank (k) shifts — through a tag revision, a recomputation,",
        "> or a lineage confluence — this log records what changed, when, and how many",
        "> records were affected. It is the cross-cutting audit trail of every selection moment",
        "> in the multiverse of interpretation.",
        ">",
        "> Tag revisions here reflect doctrinal shifts in `kindpath-canon` that propagate",
        "> downstream through `kindpath-analyser`. A revision to `fingerprint/generative-error`",
        "> may originate in `CONSCIOUS_CONTEXTUALISATION.md` — this log makes that chain",
        "> of custody visible and auditable across the entire organisation.",
        ">",
        "> **Actual log data:** `kindpath-analyser/seedbank/fork_log.json`  ",
        "> **Regenerate this file:** `from seedbank.fork_log import export_doctrinal_shifts`",
        "",
        "---",
        "",
        "## Log Statistics",
        "",
        f"- **Total events:** {stats['total_events']}",
        f"- **Total k-universes created:** {stats['universe_count']}",
        f"- **First event:** {stats['first_event'] or 'none'}",
        f"- **Most recent event:** {stats['last_event'] or 'none'}",
    ]

    if stats.get("by_type"):
        lines.append("- **Events by type:**")
        for etype, count in sorted(stats["by_type"].items()):
            lines.append(f"  - `{etype}`: {count}")
    if stats.get("most_revised_tag"):
        lines.append(f"- **Most revised tag (most active k-constant):** `{stats['most_revised_tag']}`")

    lines += ["", "---", "", "## Events (newest first)", ""]

    if not events:
        lines.append("_No events logged yet._")
    else:
        current_month = None
        for e in events:
            occurred = e.get("occurred_at", "unknown")
            month = occurred[:7] if len(occurred) >= 7 else "unknown"
            if month != current_month:
                current_month = month
                lines.append(f"### {month}")
                lines.append("")

            etype = e.get("event_type", "?")

            if etype == "deposit":
                lines.append(
                    f"- **`deposit`** `{occurred[:19]}` — "
                    f"`{e.get('filename', '?')}` born into archive "
                    f"(baseline: `{_short_baseline(e.get('new_baseline_version', ''))}`) "
                    f"· id: `{(e.get('record_id') or '')[:8]}…`"
                )

            elif etype == "tag_revision":
                lines.append(
                    f"- **`tag_revision`** `{occurred[:19]}` — "
                    f"`{e.get('tag_name', '?')}` "
                    f"v{e.get('old_version', '?')} → v{e.get('new_version', '?')} "
                    f"· reason: _{e.get('revision_reason', 'not specified')}_ "
                    f"· source: {e.get('revision_source', 'not specified')} "
                    f"· propagation pending"
                )

            elif etype == "tag_propagation":
                lines.append(
                    f"- **`tag_propagation`** `{occurred[:19]}` — "
                    f"`{e.get('tag_name', '?')}` v{e.get('new_version', '?')} "
                    f"flagged **{e.get('affected_record_count', 0)} record(s)** stale"
                )

            elif etype == "record_fork":
                delta = e.get("residue_delta")
                delta_str = f"Δresidue={delta:+.3f}" if delta is not None else "Δresidue=?"
                lines.append(
                    f"- **`record_fork`** `{occurred[:19]}` — "
                    f"`{e.get('filename', '?')}` ({e.get('computation_source', '?')}) "
                    f"· {delta_str} "
                    f"· id: `{(e.get('record_id') or '')[:8]}…`"
                )

            elif etype == "record_confluence":
                lines.append(
                    f"- **`record_confluence`** `{occurred[:19]}` — "
                    f"`{e.get('filename', '?')}` "
                    f"{e.get('num_parents', '?')}-way confluence "
                    f"· id: `{(e.get('record_id') or '')[:8]}…`"
                )

            else:
                lines.append(
                    f"- **`{etype}`** `{occurred[:19]}` — "
                    f"{e.get('notes', 'no notes')}"
                )

    lines += [
        "",
        "---",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
    ]

    output = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[FORK_LOG] Exported doctrinal shifts → {output_path}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _short_baseline(baseline: str) -> str:
    """Truncate a long baseline_version string for display in the Markdown export."""
    if not baseline:
        return "none"
    if len(baseline) <= 40:
        return baseline
    return baseline[:37] + "…"
