"""
KindPress :: constants_engine.py

Environment-wide constant discovery and evolutionary refinement.

Purpose
-------
This module scans one or more repository/workspace roots, discovers stable
repeating constants (keys, enums, tags, terminology), scores them with an
RMOE-like stability metric, and emits versioned constant snapshots.

Why this exists
---------------
KindPress compression depends on a reliable shared baseline (k). This module
builds and evolves that baseline from lived system data rather than hand-curated
lists. As constants become more stable and cross-domain, compression efficiency
improves and semantic drift becomes explicit.

Security model
--------------
The constants snapshot can derive a deterministic constants-key component.
That component is intended as ONE factor in a multi-factor key strategy
(e.g. constants_key + individual delta key + secret salt). It should not be
used as a standalone secret where adversaries know the environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".csv"
}


COMMUNITY_ERROR_CORRECTION_LAWS = [
    "add_never_delete_history",
    "k_alignment_before_decode",
    "uncertainty_must_be_routed",
    "stale_requires_recompute",
    "cross_domain_signal_preserved",
]


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,64}")

STOPWORDS = {
    "the", "and", "for", "from", "with", "that", "this", "not", "are", "was", "were", "has", "have",
    "had", "you", "your", "our", "their", "them", "they", "but", "can", "will", "would", "should",
    "could", "into", "over", "under", "when", "where", "what", "which", "who", "why", "how", "all",
    "any", "each", "more", "most", "some", "such", "only", "also", "than", "then", "very", "just",
    "true", "false", "none", "null", "json", "dict", "list", "data", "value", "record", "records",
    "file", "files", "path", "paths", "output", "input", "result", "results", "test", "tests",
    "python", "module", "class", "function", "return", "import", "raise", "error", "status",
}


@dataclass
class ConstantSignal:
    """A discovered constant candidate and its evolutionary quality metrics."""

    token: str
    occurrences: int
    file_count: int
    domain_count: int
    support: float
    recurrence: float
    domain_coverage: float
    evolutionary_stability: float
    rmoe_score: float
    confidence: float
    categories: list[str] = field(default_factory=list)


@dataclass
class ConstantsSnapshot:
    """Versioned constants baseline (k) used by KindPress refinement and alignment."""

    snapshot_id: str
    generated_at: str
    roots: list[str]
    total_files_scanned: int
    total_tokens_seen: int
    constants_count: int
    laws: list[str]
    constants: list[ConstantSignal]
    constants_key_fingerprint: str
    constants_key_params: dict


@dataclass
class ScanConfig:
    roots: list[str]
    max_files: int = 15000
    max_file_size_bytes: int = 2_000_000
    min_occurrences: int = 4
    max_constants: int = 2000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_files(roots: list[Path], max_files: int) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(files) >= max_files:
                return files
            if not path.is_file():
                continue
            if any(part in {".git", "venv", "__pycache__", ".pytest_cache", "node_modules", "build", "dist"}
                   for part in path.parts):
                continue
            if path.suffix.lower() in TEXT_EXTENSIONS:
                files.append(path)
    return files


def _domain_for_file(path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            rel = path.relative_to(root)
            return rel.parts[0] if rel.parts else root.name
        except ValueError:
            continue
    return path.parts[0] if path.parts else "unknown"


def _classify_token(token: str) -> str:
    if token.startswith("kindpath") or token.startswith("kind"):
        return "kindpath_namespace"
    if token.isupper() and "_" in token:
        return "constant_identifier"
    if token.endswith("_id") or token == "id":
        return "identity_key"
    if re.match(r"^[a-z]+(_[a-z0-9]+)+$", token):
        return "snake_term"
    if re.match(r"^[A-Za-z]+-[A-Za-z0-9\-]+$", token):
        return "tag_or_slug"
    return "general_token"


def _is_signal_token(token: str) -> bool:
    lower = token.lower()
    if lower in STOPWORDS:
        return False

    # Plain natural-language words with no structure are weak constants.
    if token.islower() and "_" not in token and "-" not in token and len(token) < 6:
        return False

    return True


def _extract_tokens_from_json_obj(obj, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str):
                out.extend(TOKEN_RE.findall(key))
            _extract_tokens_from_json_obj(value, out)
    elif isinstance(obj, list):
        for item in obj:
            _extract_tokens_from_json_obj(item, out)
    elif isinstance(obj, str):
        out.extend(TOKEN_RE.findall(obj))


def _extract_tokens(path: Path) -> list[str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    ext = path.suffix.lower()
    tokens: list[str] = []

    if ext == ".json":
        try:
            obj = json.loads(raw)
            _extract_tokens_from_json_obj(obj, tokens)
            return tokens
        except json.JSONDecodeError:
            pass
    elif ext == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                _extract_tokens_from_json_obj(obj, tokens)
            except json.JSONDecodeError:
                tokens.extend(TOKEN_RE.findall(line))
        return tokens

    tokens.extend(TOKEN_RE.findall(raw))
    return tokens


def _canonical_constants_payload(constants: list[ConstantSignal]) -> bytes:
    compact = [
        {
            "t": c.token,
            "r": round(c.rmoe_score, 6),
            "o": c.occurrences,
            "f": c.file_count,
            "d": c.domain_count,
        }
        for c in constants
    ]
    compact = sorted(compact, key=lambda x: x["t"])
    return json.dumps(compact, separators=(",", ":"), sort_keys=True).encode("utf-8")


def derive_constants_key_material(constants: list[ConstantSignal],
                                  *,
                                  context_salt: str = "kindpress.constants.v1",
                                  iterations: int = 600_000) -> bytes:
    """
    Derive deterministic 32-byte constants-key material from constant snapshot.

    This is a deterministic key component from shared constants.
    Use as one factor in a multi-factor derivation strategy.
    """
    payload = _canonical_constants_payload(constants)
    return hashlib.pbkdf2_hmac(
        "sha256",
        payload,
        context_salt.encode("utf-8"),
        iterations=iterations,
        dklen=32,
    )


def scan_constants(config: ScanConfig,
                   previous_snapshot: Optional[dict] = None) -> ConstantsSnapshot:
    """
    Scan roots and build an evolutionary constants snapshot with RMOE scores.
    """
    roots = [Path(p).expanduser().resolve() for p in config.roots]
    files = _iter_files(roots, max_files=config.max_files)

    token_occ: dict[str, int] = {}
    token_files: dict[str, set[str]] = {}
    token_domains: dict[str, set[str]] = {}
    token_categories: dict[str, set[str]] = {}

    total_tokens_seen = 0

    for path in files:
        try:
            if path.stat().st_size > config.max_file_size_bytes:
                continue
        except OSError:
            continue

        tokens = _extract_tokens(path)
        if not tokens:
            continue

        total_tokens_seen += len(tokens)
        file_key = str(path)
        domain = _domain_for_file(path, roots)

        for token in tokens:
            token = token.strip()
            if len(token) < 3:
                continue
            if not _is_signal_token(token):
                continue
            token_occ[token] = token_occ.get(token, 0) + 1
            token_files.setdefault(token, set()).add(file_key)
            token_domains.setdefault(token, set()).add(domain)
            token_categories.setdefault(token, set()).add(_classify_token(token))

    prev_scores: dict[str, float] = {}
    if previous_snapshot:
        for c in previous_snapshot.get("constants", []):
            t = c.get("token")
            s = float(c.get("rmoe_score", 0.0))
            if t:
                prev_scores[t] = s

    if not token_occ:
        constants: list[ConstantSignal] = []
    else:
        max_occ = max(token_occ.values())
        total_files = max(len(files), 1)
        distinct_domains = set()
        for ds in token_domains.values():
            distinct_domains.update(ds)
        total_domains = max(len(distinct_domains), 1)

        constants = []
        for token, occ in token_occ.items():
            if occ < config.min_occurrences:
                continue

            fcount = len(token_files.get(token, set()))
            dcount = len(token_domains.get(token, set()))

            support = fcount / total_files
            recurrence = math.log1p(occ) / math.log1p(max_occ)
            domain_coverage = dcount / total_domains

            prev = prev_scores.get(token)
            if prev is None:
                evolutionary_stability = 0.5
            else:
                # Stability rewards continuity and penalises abrupt volatility.
                evolutionary_stability = max(0.0, 1.0 - abs(prev - (support * 0.5 + recurrence * 0.5)))

            rmoe = (
                0.35 * support
                + 0.25 * recurrence
                + 0.25 * domain_coverage
                + 0.15 * evolutionary_stability
            )
            rmoe = max(0.0, min(1.0, rmoe))

            confidence = max(0.0, min(1.0, 0.6 * recurrence + 0.4 * support))

            constants.append(
                ConstantSignal(
                    token=token,
                    occurrences=occ,
                    file_count=fcount,
                    domain_count=dcount,
                    support=round(support, 6),
                    recurrence=round(recurrence, 6),
                    domain_coverage=round(domain_coverage, 6),
                    evolutionary_stability=round(evolutionary_stability, 6),
                    rmoe_score=round(rmoe, 6),
                    confidence=round(confidence, 6),
                    categories=sorted(token_categories.get(token, set())),
                )
            )

        constants.sort(key=lambda c: (c.rmoe_score, c.occurrences, c.file_count), reverse=True)
        constants = constants[: config.max_constants]

    key_material = derive_constants_key_material(constants)
    key_fp = hashlib.sha256(key_material).hexdigest()[:16]

    return ConstantsSnapshot(
        snapshot_id=str(uuid.uuid4()),
        generated_at=_now_iso(),
        roots=[str(r) for r in roots],
        total_files_scanned=len(files),
        total_tokens_seen=total_tokens_seen,
        constants_count=len(constants),
        laws=COMMUNITY_ERROR_CORRECTION_LAWS,
        constants=constants,
        constants_key_fingerprint=key_fp,
        constants_key_params={
            "kdf": "PBKDF2-HMAC-SHA256",
            "iterations": 600_000,
            "salt": "kindpress.constants.v1",
            "dklen": 32,
        },
    )


def snapshot_to_dict(snapshot: ConstantsSnapshot) -> dict:
    d = asdict(snapshot)
    d["constants"] = [asdict(c) for c in snapshot.constants]
    return d


def load_snapshot(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_snapshot(snapshot: ConstantsSnapshot, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "constants.latest.json"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = output_dir / f"constants.{ts}.json"
    history = output_dir / "constants.history.jsonl"

    payload = snapshot_to_dict(snapshot)

    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    snap_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(history, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")

    return {
        "latest": str(latest),
        "snapshot": str(snap_path),
        "history": str(history),
    }
