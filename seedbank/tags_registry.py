"""
KindPath Analyser :: Seedbank Tags Registry

The constants bank for the seedbank. Tags are not static labels — they are
versioned definitions of shared context. When understanding improves, tags
are revised, and that revision propagates through past fingerprints.

This module implements the Living Constants Principle from:
  kindpath-canon/CONSCIOUS_CONTEXTUALISATION.md

The registry is stored at seedbank/tags_registry.json.
It is human-readable, human-editable, and version-controlled.

Design principles:
- No tag definition is ever deleted. Old versions are archived.
- Every revision records what changed and why.
- Records encoded under old tag versions are flagged for recomputation,
  not silently overwritten.
- The delta between old and new residue values is itself a signal.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict

_DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "tags_registry.json")

REGISTRY_PATH: str = os.environ.get(
    "KINDPATH_TAGS_REGISTRY",
    _DEFAULT_REGISTRY_PATH,
)


@dataclass
class TagVersion:
    """
    One version of a tag's definition.

    Tags evolve as understanding deepens. Each version carries a description
    of what the tag means at that point in time, when it was defined, and
    what prompted the revision (if it's not the first version).
    """
    version: int
    defined_at: str                      # ISO 8601 timestamp
    description: str                     # What this tag means — plain language
    scope: str                           # What kinds of signal this tag applies to
    revision_reason: Optional[str]       # Why this version differs from the previous
    revision_source: Optional[str]       # Where the new understanding came from
    examples: List[str] = field(default_factory=list)   # Concrete examples
    anti_examples: List[str] = field(default_factory=list)  # What this tag is NOT


@dataclass
class TagDefinition:
    """
    The full versioned definition of a single tag.

    current_version is the version number of the currently recommended
    interpretation. All prior versions are retained in history.
    """
    name: str
    category: str                        # e.g. "fingerprint", "signal", "context", "experiment"
    current_version: int
    versions: List[TagVersion]           # Full history, oldest first
    deprecated: bool = False
    deprecated_reason: Optional[str] = None
    superseded_by: Optional[str] = None  # If deprecated, which tag replaced it


def _load_registry() -> Dict[str, dict]:
    """Load the registry from disk. Returns empty dict if not found."""
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_registry(registry: Dict[str, dict]) -> None:
    """Write the registry atomically."""
    os.makedirs(os.path.dirname(REGISTRY_PATH) or ".", exist_ok=True)
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    os.replace(tmp, REGISTRY_PATH)


def get_tag(name: str) -> Optional[TagDefinition]:
    """
    Return the current definition of a tag, or None if it doesn't exist.
    """
    registry = _load_registry()
    raw = registry.get(name)
    if not raw:
        return None
    versions = [TagVersion(**v) for v in raw.get("versions", [])]
    return TagDefinition(
        name=raw["name"],
        category=raw["category"],
        current_version=raw["current_version"],
        versions=versions,
        deprecated=raw.get("deprecated", False),
        deprecated_reason=raw.get("deprecated_reason"),
        superseded_by=raw.get("superseded_by"),
    )


def list_tags(category: Optional[str] = None, include_deprecated: bool = False) -> List[TagDefinition]:
    """
    List all tag definitions, optionally filtered by category.
    Deprecated tags are excluded by default.
    """
    registry = _load_registry()
    results = []
    for raw in registry.values():
        if not include_deprecated and raw.get("deprecated", False):
            continue
        if category and raw.get("category") != category:
            continue
        versions = [TagVersion(**v) for v in raw.get("versions", [])]
        results.append(TagDefinition(
            name=raw["name"],
            category=raw["category"],
            current_version=raw["current_version"],
            versions=versions,
            deprecated=raw.get("deprecated", False),
            deprecated_reason=raw.get("deprecated_reason"),
            superseded_by=raw.get("superseded_by"),
        ))
    return sorted(results, key=lambda t: t.name)


def define_tag(
    name: str,
    category: str,
    description: str,
    scope: str,
    examples: Optional[List[str]] = None,
    anti_examples: Optional[List[str]] = None,
) -> TagDefinition:
    """
    Define a new tag. Raises ValueError if the tag already exists
    (use revise_tag() to update an existing one).
    """
    registry = _load_registry()
    if name in registry:
        raise ValueError(
            f"Tag '{name}' already exists. Use revise_tag() to update it."
        )

    first_version = TagVersion(
        version=1,
        defined_at=datetime.now(timezone.utc).isoformat(),
        description=description,
        scope=scope,
        revision_reason=None,
        revision_source=None,
        examples=examples or [],
        anti_examples=anti_examples or [],
    )
    tag = TagDefinition(
        name=name,
        category=category,
        current_version=1,
        versions=[first_version],
    )
    registry[name] = asdict(tag)
    _save_registry(registry)
    print(f"[TAGS] Defined new tag: {name} (v1)")
    return tag


def propose_tag_revision(
    name: str,
    description: str,
    scope: str,
    chunk_sizes: Optional[List[int]] = None,
    max_sample: int = 200,
    seed: int = 42,
) -> "TagRevisionReport":  # noqa: F821 — forward ref resolved at call time
    """
    Run the full validation protocol for a proposed tag revision WITHOUT committing it.

    This is the correct entry point for any tag change. It returns a TagRevisionReport
    with a recommendation (APPROVE / CAUTION / REJECT) and a revision_rationale you
    can pass directly to revise_tag() as the revision_reason.

    Raises ImportError if kindpress.validate is not available.
    """
    # Lazy import to avoid circular dependency at module load time
    from kindpress.validate import validate_tag_revision  # noqa: PLC0415
    return validate_tag_revision(
        tag_name=name,
        proposed_description=description,
        proposed_scope=scope,
        chunk_sizes=chunk_sizes,
        max_sample=max_sample,
        seed=seed,
    )


def revise_tag(
    name: str,
    description: str,
    scope: str,
    revision_reason: str,
    revision_source: Optional[str] = None,
    examples: Optional[List[str]] = None,
    anti_examples: Optional[List[str]] = None,
    validate_first: bool = True,
    _validation_report=None,
) -> TagDefinition:
    """
    Revise an existing tag's definition.

    Adds a new version to the tag's history. Does not delete the old version.
    Returns the updated TagDefinition.

    validate_first (default True): Run the full KindPress validation protocol before
    committing. A REJECT recommendation raises ValueError — the revision is blocked
    until the proposed change is reworked. A CAUTION recommendation prints a warning
    and proceeds. Pass validate_first=False only when the caller holds a pre-computed
    validation report (pass it as _validation_report) or for emergency dry-run scenarios
    where the corpus is empty.

    _validation_report: an already-computed TagRevisionReport to use instead of
    re-running the protocol. Only honoured if validate_first=True.

    After revision, any seedbank records tagged with this tag and encoded
    under a prior version should be considered stale — run
    flag_stale_records(tag_name) to mark them for recomputation.
    """
    if validate_first:
        # Use a pre-computed report if provided, otherwise run the full protocol.
        if _validation_report is None:
            try:
                from kindpress.validate import validate_tag_revision  # noqa: PLC0415
                _validation_report = validate_tag_revision(
                    tag_name=name,
                    proposed_description=description,
                    proposed_scope=scope,
                )
            except ImportError:
                # kindpress not installed — degrade gracefully, warn loudly.
                print(
                    f"[TAGS] WARNING: kindpress.validate unavailable — proceeding without "
                    f"HMoE gate for '{name}'. Install kindpress to enforce the revision gate."
                )
                _validation_report = None

        if _validation_report is not None:
            rec = _validation_report.recommendation
            if rec == "REJECT":
                raise ValueError(
                    f"Tag revision for '{name}' REJECTED by HMoE validation protocol.\n"
                    f"Rationale: {_validation_report.revision_rationale or _validation_report.implication_summary}\n"
                    f"Run propose_tag_revision('{name}', ...) to diagnose the issue, "
                    f"then rework the proposed description before retrying."
                )
            if rec == "CAUTION":
                print(
                    f"[TAGS] CAUTION: Revision for '{name}' is sound at small scale but "
                    f"degrades at corpus optimum. Proceeding — monitor for HMoE drift.\n"
                    f"Rationale: {_validation_report.revision_rationale or _validation_report.implication_summary}"
                )

    registry = _load_registry()
    if name not in registry:
        raise ValueError(
            f"Tag '{name}' not found. Use define_tag() to create it."
        )

    raw = registry[name]
    if raw.get("deprecated"):
        raise ValueError(
            f"Tag '{name}' is deprecated. Revise its successor instead."
        )

    new_version_num = raw["current_version"] + 1
    new_version = TagVersion(
        version=new_version_num,
        defined_at=datetime.now(timezone.utc).isoformat(),
        description=description,
        scope=scope,
        revision_reason=revision_reason,
        revision_source=revision_source,
        examples=examples or [],
        anti_examples=anti_examples or [],
    )
    raw["versions"].append(asdict(new_version))
    raw["current_version"] = new_version_num
    registry[name] = raw
    _save_registry(registry)
    print(f"[TAGS] Revised tag: {name} → v{new_version_num}. Run flag_stale_records('{name}') to propagate.")
    # Log the k-constant shift to the fork audit trail.
    # The affected_record_count is unknown here — it becomes concrete after
    # flag_stale_records() runs and logs the companion tag_propagation event.
    try:
        import seedbank.fork_log as _fork_log
        _fork_log.log_tag_revision(
            tag_name=name,
            old_version=new_version_num - 1,
            new_version=new_version_num,
            revision_reason=revision_reason,
            revision_source=revision_source,
        )
    except Exception:
        pass
    return get_tag(name)


def deprecate_tag(
    name: str,
    reason: str,
    superseded_by: Optional[str] = None,
) -> None:
    """
    Mark a tag as deprecated. Does not delete it — all past records
    that used it retain the reference. Deprecated tags are excluded
    from search by default but remain in the registry for historical
    reconstruction.
    """
    registry = _load_registry()
    if name not in registry:
        raise ValueError(f"Tag '{name}' not found.")
    registry[name]["deprecated"] = True
    registry[name]["deprecated_reason"] = reason
    registry[name]["superseded_by"] = superseded_by
    _save_registry(registry)
    print(f"[TAGS] Deprecated: {name}. {'Superseded by: ' + superseded_by if superseded_by else ''}")


def flag_stale_records(tag_name: str) -> List[str]:
    """
    After a tag is revised, scan all seedbank records and flag any that:
    - include this tag, AND
    - were encoded under a prior version of the tag definition

    Marks each affected record with creative_residue_stale: true in the index
    and in the full profile JSON.

    Returns the list of record IDs that were flagged.

    This does NOT recompute the residue — it only flags. Recomputation
    requires the full analysis pipeline and should be done deliberately.
    Call recompute_stale_records() once you are ready to re-derive.
    """
    import seedbank.index as _idx

    tag = get_tag(tag_name)
    if not tag:
        raise ValueError(f"Tag '{tag_name}' not found.")

    current_version = tag.current_version
    index = _idx._load_index()
    flagged = []

    for record in index.get("records", []):
        tags = record.get("tags", [])
        if tag_name not in tags:
            continue

        # Check if the record was encoded under an older version
        record_baseline = record.get("baseline_version", "")
        # Baseline version format: "{tag}:v{n}" or just a timestamp/hash
        # We flag if the record has no baseline_version (legacy) or if it
        # references a prior version number for this specific tag
        record_tag_version = _extract_tag_version(record_baseline, tag_name)
        if record_tag_version is None or record_tag_version < current_version:
            record["creative_residue_stale"] = True
            record["stale_tags"] = list(set(record.get("stale_tags", []) + [tag_name]))
            flagged.append(record["id"])

    if flagged:
        index["last_updated"] = datetime.now(timezone.utc).isoformat()
        _idx._save_index(index)
        print(f"[TAGS] Flagged {len(flagged)} records stale for tag '{tag_name}' revision to v{current_version}.")
        # Log the propagation: N branches now exist between the old and new k-universe.
        try:
            import seedbank.fork_log as _fork_log
            _fork_log.log_tag_propagation(
                tag_name=tag_name,
                new_version=current_version,
                affected_record_count=len(flagged),
            )
        except Exception:
            pass
    else:
        print(f"[TAGS] No stale records found for tag '{tag_name}'.")

    return flagged


def _extract_tag_version(baseline_version: str, tag_name: str) -> Optional[int]:
    """
    Parse a baseline_version string to find the version number for a specific tag.

    Format: "tag_name:v2,other_tag:v1" or empty string for legacy records.
    Returns None if the tag is not mentioned (treat as v0 — always stale).
    """
    if not baseline_version:
        return None
    for part in baseline_version.split(","):
        part = part.strip()
        if part.startswith(tag_name + ":v"):
            try:
                return int(part.split(":v")[1])
            except (IndexError, ValueError):
                return None
    return None


def get_stale_records() -> List[dict]:
    """
    Return all index entries currently marked creative_residue_stale: true.
    These are records whose fingerprints were computed under a since-revised
    set of tag definitions and may benefit from recomputation.
    """
    import seedbank.index as _idx
    index = _idx._load_index()
    return [r for r in index.get("records", []) if r.get("creative_residue_stale")]


def current_baseline_version(tags: List[str]) -> str:
    """
    Given a list of tags, return a baseline_version string encoding the
    current version of each tag. This is stored at deposit time so future
    staleness checks can be precise.

    Format: "tag_name:v2,other_tag:v1"

    If a tag is not in the registry yet, it is recorded as v0 (unversioned).
    """
    registry = _load_registry()
    parts = []
    for tag in sorted(tags):
        version = registry.get(tag, {}).get("current_version", 0)
        parts.append(f"{tag}:v{version}")
    return ",".join(parts)


def seed_default_tags() -> None:
    """
    Seed the registry with the initial set of tags used by the fieldkit
    CONTRIBUTING.md and the analyser's auto-tagging system.

    Safe to call multiple times — skips tags that already exist.
    """
    defaults = [
        dict(
            name="fingerprint/generative-error",
            category="fingerprint",
            description="A deviation that carries structural logic — it reveals a genuine gap in the model or the material rather than random noise. The error has a fingerprint: it follows the geometry of the system rather than disrupting it arbitrarily.",
            scope="Field practice, documentation, analysis anomalies",
            examples=["A sensor reading that drifts in response to soil compaction in a specific site", "A PR that exposes ambiguity in a field guide by failing to apply it in a real community context"],
            anti_examples=["A typo", "A corrupted data file", "A configuration mistake"],
        ),
        dict(
            name="fingerprint/context-local",
            category="fingerprint",
            description="A contribution or signal that is specific to a particular place, community, ecological context, or cultural moment. It should not be generalised away — its specificity is the value.",
            scope="Field contributions, site-specific data, community-specific practice",
            examples=["Northern Rivers flood-season soil readings", "Nimbin community governance patterns", "A field guide interpretation that only applies to clay-rich coastal soils"],
            anti_examples=["A universal algorithmic improvement", "A documentation fix with no geographic or community specificity"],
        ),
        dict(
            name="signal/miscontextualised",
            category="signal",
            description="Existing material that is technically accurate but not legible or applicable in the real-world context encountered. The map is correct but oriented to the wrong territory.",
            scope="Documentation, field guides, analysis outputs",
            examples=["A pilot guide that assumes infrastructure not available in rural settings", "A metric definition that works for urban cohorts but not remote communities"],
            anti_examples=["Material that is simply wrong — that's a different category"],
        ),
        dict(
            name="signal/extractive-precision",
            category="signal",
            description="A place where the system is too rigid, template-heavy, or efficiency-oriented at the expense of adaptability, authentic signal, or community agency. Optimisation that eats its own margin of error.",
            scope="Process design, automation, contribution requirements, data pipeline design",
            examples=["A contribution template so strict it filters out valid field context", "An analysis pipeline that smooths out anomalies before they can be examined", "A governance process that requires uniform documentation across diverse community contexts"],
            anti_examples=["Well-designed structure that enables rather than constrains"],
        ),
        dict(
            name="experiment/noise",
            category="experiment",
            description="A failed attempt whose conditions and failure mode are worth preserving. Not random error — a structured experiment that produced unexpected results in a specific context. The negative result is data.",
            scope="Technical experiments, field trials, analysis attempts",
            examples=["A stem separation attempt that failed for a specific audio format", "A sensor integration that produced unstable readings under specific temperature conditions"],
            anti_examples=["A broken prototype with no documented conditions", "An incomplete implementation with no notes"],
        ),
        dict(
            name="high_lsii",
            category="fingerprint",
            description="Late-song inversion index above 0.4 — the final quarter diverges meaningfully from the Q1-Q3 trajectory. Something changed in the last section that was not established by the frame the piece was built in.",
            scope="Music analysis records",
            examples=[],
            anti_examples=[],
        ),
        dict(
            name="natural_dynamics",
            category="fingerprint",
            description="Crest factor above 12dB and dynamic range above 12dB. Dynamics are preserved — the production chose listener experience and authentic signal over loudness or compression.",
            scope="Music analysis records",
            examples=[],
            anti_examples=[],
        ),
        dict(
            name="independent",
            category="context",
            description="Released or produced independently of major label infrastructure. Not a quality judgment — a context flag that affects baseline comparison (independent releases are not measured against commercial production standards).",
            scope="Music analysis records, creative work contexts",
            examples=[],
            anti_examples=[],
        ),
    ]

    for kwargs in defaults:
        try:
            define_tag(**kwargs)
        except ValueError:
            pass  # Already exists — skip
