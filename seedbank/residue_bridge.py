"""
residue_bridge.py — Seedbank Residue Integration for kindpath-analyser

Every seedbank deposit carries a creative_residue score — what remains of the
authentic creative signal after known genre/era signatures are subtracted.

This module connects the seedbank's creative residue data to the wider
BEC accumulation layer: extracting the residue items embedded in seedbank
deposits and accumulating them in a cross-source-compatible format.

The purpose: when enough seedbank deposits accumulate across enough eras,
genres, and production contexts, the residue corpus becomes the ground truth
for a new model of authentic creative signal — one that transcends any
individual genre frame. See kindpath-canon/BEC_DETECTION_GUIDE.md.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RESIDUE_DB_PATH = Path(__file__).parent / "residue_corpus.db"

# ── Residue types (kindpath-analyser-specific) ─────────────────────────────

RESIDUE_HIGH_CREATIVE_RESIDUE = "creative_residue_high"
# A track whose creative_residue score is high — indicating authentic creative
# choices that don't fit the known fingerprint database. These are the most
# valuable deposits: they extend the model.

RESIDUE_GENRE_FRAME_MISS = "genre_frame_miss"
# Track behaviour inconsistent with its assigned genre model — the genre
# fingerprint failed to account for significant feature variance.

RESIDUE_LSII_UNEXPLAINED = "lsii_unexplained"
# High LSII score not explained by any known production convention —
# the late-song inversion is authentic, not a technique artifact.

RESIDUE_ERA_ANOMALY = "era_anomaly"
# Features that don't match the detected era, in ways that are neither
# anachronism nor production error — genuine cross-era creative vision.


@dataclass
class AnalysisResidueItem:
    """
    One residue item extracted from a seedbank analysis deposit.

    'Residue' here means: the part of the musical signal that the current
    analysis model couldn't account for. The creative_residue score is the
    top-level indicator, but the residue item carries the specific nature
    of what couldn't be explained.
    """
    id: str
    item_type: str
    created_utc: str

    # Source deposit
    seedbank_record_id: str
    filename: str

    # Analysis data at residue point
    feature_description: str                  # What feature produced the residue
    genre_era_baseline: dict                  # The baseline that was subtracted
    observed_features: dict                   # What was actually observed
    creative_residue_score: float             # 0-1: from psychosomatics.py

    # Interpretation
    residue_description: str                  # Plain language: what couldn't be explained
    authentic_emission_score: float           # Corresponding authentic emission score
    lsii_score: Optional[float] = None       # If LSII was involved

    # BEC tracking
    bec_cluster_id: Optional[str] = None
    bec_threshold_crossed: bool = False
    source_repo: str = "kindpath-analyser"


class SeedbankResidueBridge:
    """
    Extracts and accumulates residue items from seedbank deposits.

    Call this after each seedbank deposit is made to extract its residue
    content into the persistent corpus.

    The corpus grows as the seedbank grows. When the BEC threshold is
    crossed, the corpus becomes the basis for a new model of authentic
    creative signal.

    Usage:
        bridge = SeedbankResidueBridge()

        # After a seedbank.deposit() call:
        bridge.extract_from_deposit(
            seedbank_record_id=record.id,
            filename=record.filename,
            psychosomatic_profile=profile,
            fingerprint_report=fingerprints,
        )
    """

    def __init__(self, db_path: Path = RESIDUE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_residue (
                    id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    seedbank_record_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    feature_description TEXT NOT NULL,
                    genre_era_baseline TEXT NOT NULL,
                    observed_features TEXT NOT NULL,
                    creative_residue_score REAL NOT NULL,
                    residue_description TEXT NOT NULL,
                    authentic_emission_score REAL NOT NULL,
                    lsii_score REAL,
                    bec_cluster_id TEXT,
                    bec_threshold_crossed INTEGER DEFAULT 0,
                    source_repo TEXT DEFAULT 'kindpath-analyser'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ar_type
                ON analysis_residue(item_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ar_record
                ON analysis_residue(seedbank_record_id)
            """)
            conn.commit()

    def extract_from_deposit(
        self,
        seedbank_record_id: str,
        filename: str,
        creative_residue_score: float,
        authentic_emission_score: float,
        genre_era_baseline: dict,
        observed_features: dict,
        lsii_score: Optional[float] = None,
        manufacturing_score: float = 0.0,
    ) -> Optional[AnalysisResidueItem]:
        """
        Extract a residue item from a seedbank deposit if warranted.

        A deposit generates a residue item if:
        - creative_residue_score > 0.3 (meaningful residue exists)
        - OR lsii_score > 0.4 (significant late-song divergence)
        - OR the genre_era_baseline doesn't account for >30% of feature variance

        Deposits below all thresholds are high-formula work — they don't
        contribute meaningfully to the residue corpus.
        """
        # Only accumulate residue for high-residue deposits
        lsii_significant = lsii_score is not None and lsii_score > 0.4
        residue_significant = creative_residue_score > 0.3

        if not (residue_significant or lsii_significant):
            return None

        # Determine residue type
        if lsii_significant and creative_residue_score > 0.5:
            item_type = RESIDUE_LSII_UNEXPLAINED
            description = (
                f"High LSII ({lsii_score:.2f}) combined with high creative residue "
                f"({creative_residue_score:.2f}). The late-song inversion is accompanied "
                "by authentic creative choices that transcend the genre model."
            )
        elif creative_residue_score > 0.6:
            item_type = RESIDUE_HIGH_CREATIVE_RESIDUE
            description = (
                f"Creative residue score {creative_residue_score:.2f} — "
                "a significant portion of this track's features are not explained "
                "by the current genre/era fingerprint model. "
                "This deposit extends the model boundary."
            )
        elif lsii_significant:
            item_type = RESIDUE_LSII_UNEXPLAINED
            description = (
                f"LSII {lsii_score:.2f} with no known production convention to explain it. "
                "The late-song divergence appears to be authentic creative expression."
            )
        else:
            item_type = RESIDUE_GENRE_FRAME_MISS
            description = (
                f"Creative residue {creative_residue_score:.2f} — "
                "genre frame doesn't account for all feature variance."
            )

        item = AnalysisResidueItem(
            id=str(uuid.uuid4()),
            item_type=item_type,
            created_utc=datetime.now(timezone.utc).isoformat(),
            seedbank_record_id=seedbank_record_id,
            filename=filename,
            feature_description=description,
            genre_era_baseline=genre_era_baseline,
            observed_features=observed_features,
            creative_residue_score=creative_residue_score,
            residue_description=description,
            authentic_emission_score=authentic_emission_score,
            lsii_score=lsii_score,
        )
        self._store(item)
        return item

    def get_corpus_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM analysis_residue").fetchone()[0]
            by_type = dict(
                conn.execute(
                    "SELECT item_type, COUNT(*) FROM analysis_residue GROUP BY item_type"
                ).fetchall()
            )
            avg_residue = conn.execute(
                "SELECT AVG(creative_residue_score) FROM analysis_residue"
            ).fetchone()[0]

        return {
            "total_items": total,
            "by_type": by_type,
            "average_creative_residue": round(avg_residue or 0.0, 3),
            "distinct_types": len(by_type),
            "source_repo": "kindpath-analyser",
        }

    def _store(self, item: AnalysisResidueItem):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO analysis_residue (
                    id, item_type, created_utc, seedbank_record_id, filename,
                    feature_description, genre_era_baseline, observed_features,
                    creative_residue_score, residue_description,
                    authentic_emission_score, lsii_score,
                    bec_cluster_id, bec_threshold_crossed, source_repo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id, item.item_type, item.created_utc,
                    item.seedbank_record_id, item.filename,
                    item.feature_description,
                    json.dumps(item.genre_era_baseline),
                    json.dumps(item.observed_features),
                    item.creative_residue_score,
                    item.residue_description,
                    item.authentic_emission_score,
                    item.lsii_score,
                    item.bec_cluster_id,
                    int(item.bec_threshold_crossed),
                    item.source_repo,
                ),
            )
            conn.commit()
