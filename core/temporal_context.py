"""
temporal_context.py — Field-Time Context for kindpath-analyser

Music is always made at a specific circadian phase. A track made at 3am in a
home studio is a different organism from a track made at midday in a professional
studio. When metadata is available, this context enriches the analysis.

This module builds and manages TemporalContext objects for analysis sessions
and — where metadata allows — for tracks in the seedbank.

See kindpath-canon/TEMPORAL_SOVEREIGNTY.md for the full framework.
This implementation is adapted from dfte/field_time_bridge.py for the
analysis/seedbank context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal, Optional


CircadianQuarter = Literal[1, 2, 3, 4, 5]
EightPointSeason = Literal[
    "early-spring", "mid-spring", "late-spring",
    "early-summer", "midsummer",
    "early-autumn", "mid-autumn", "late-autumn",
    "early-winter", "deep-winter",
]


@dataclass
class AnalysisTemporalContext:
    """
    Temporal context for an analysis session or a seedbank deposit.

    Two use cases:
    1. Analysis session context — when the analysis itself is happening:
       used to contextualise the analysis output (a track analysed at 3am
       by an exhausted researcher is different data from the same track at noon)

    2. Track creation context — when metadata suggests when the track was made:
       embedded in the seedbank deposit for future corpus-level analysis
       (did this era produce different creative residue in certain seasons?)
    """

    # What this context represents
    context_type: Literal["analysis_session", "track_creation", "unknown"]

    # Temporal position
    circadian_quarter: CircadianQuarter
    circadian_state: str                    # 'morning-activation' | 'midday-consolidation' etc

    # Ecological position
    season: EightPointSeason
    hemisphere: Literal["north", "south", "equatorial"]

    # The forgazi reference
    utc_reference: str
    extraction_frame_label: str

    # Optional enrichment
    latitude_band: Optional[int] = None    # 10-degree band if location context available
    analyst_note: str = ""                 # Any relevant context note

    def to_dict(self) -> dict:
        return {
            "context_type": self.context_type,
            "circadian_quarter": self.circadian_quarter,
            "circadian_state": self.circadian_state,
            "season": self.season,
            "hemisphere": self.hemisphere,
            "latitude_band": self.latitude_band,
            "utc_reference": self.utc_reference,
            "extraction_frame_label": self.extraction_frame_label,
            "analyst_note": self.analyst_note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisTemporalContext":
        return cls(
            context_type=d.get("context_type", "unknown"),
            circadian_quarter=d.get("circadian_quarter", 1),
            circadian_state=d.get("circadian_state", "unknown"),
            season=d.get("season", "early-spring"),
            hemisphere=d.get("hemisphere", "north"),
            latitude_band=d.get("latitude_band"),
            utc_reference=d.get("utc_reference", ""),
            extraction_frame_label=d.get("extraction_frame_label", "UTC+00:00"),
            analyst_note=d.get("analyst_note", ""),
        )


def _solar_times(latitude: float, dt: date) -> tuple[float, float]:
    """Approximate sunrise and sunset in fractional solar hours."""
    B = math.radians((360 / 365) * (dt.timetuple().tm_yday - 81))
    dec = math.radians(23.45 * math.sin(B))
    lat = math.radians(latitude)
    cos_ha = max(-1.0, min(1.0, -math.tan(lat) * math.tan(dec)))
    ha = math.degrees(math.acos(cos_ha))
    return 12.0 - ha / 15.0, 12.0 + ha / 15.0


def _circadian_quarter(local_hour: float, lat: float, dt: date) -> tuple[CircadianQuarter, str]:
    sunrise, sunset = _solar_times(lat, dt)
    daylight = sunset - sunrise
    mins_since_dawn = (local_hour - sunrise) * 60
    total_mins = daylight * 60

    if mins_since_dawn < 0:
        return 5, "deep-repair"
    frac = mins_since_dawn / max(total_mins, 1)
    if frac < 0.35:
        return 1, "morning-activation"
    if frac < 0.50:
        return 2, "midday-consolidation"
    if frac < 0.75:
        return 3, "afternoon-activation"
    if frac < 1.0:
        return 4, "evening-repair"
    post = mins_since_dawn - total_mins
    return (4, "evening-repair") if post < 180 else (5, "deep-repair")


def _season(dt: date, hemisphere: str) -> EightPointSeason:
    m = dt.month
    if m == 3:
        s: EightPointSeason = "early-spring"
    elif m == 4:
        s = "mid-spring"
    elif m == 5:
        s = "late-spring"
    elif m == 6:
        s = "early-summer"
    elif m == 7:
        s = "midsummer"
    elif m == 8:
        s = "early-autumn"
    elif m == 9:
        s = "mid-autumn"
    elif m == 10:
        s = "late-autumn"
    elif m == 11:
        s = "early-winter"
    elif m == 12:
        s = "deep-winter"
    elif m == 1:
        s = "deep-winter"
    else:
        s = "early-spring"

    if hemisphere != "south":
        return s

    inv = {
        "early-spring": "early-autumn", "mid-spring": "mid-autumn",
        "late-spring": "late-autumn", "early-summer": "early-winter",
        "midsummer": "deep-winter", "early-autumn": "early-spring",
        "mid-autumn": "mid-spring", "late-autumn": "late-spring",
        "early-winter": "early-summer", "deep-winter": "midsummer",
    }
    return inv[s]


def build_session_context(
    latitude_band: int = -35,
    hemisphere: Literal["north", "south", "equatorial"] = "south",
    utc_offset_hours: float = 10.0,
    note: str = "",
) -> AnalysisTemporalContext:
    """
    Build a temporal context for the current analysis session.

    Default coordinates: Northern Rivers, NSW (lat -35, UTC+10, southern hemisphere).
    """
    now = datetime.now(timezone.utc)
    local_hour = (now.hour + utc_offset_hours) % 24
    q, state = _circadian_quarter(local_hour, latitude_band, now.date())
    sign = "+" if utc_offset_hours >= 0 else ""
    frame_label = f"UTC{sign}{int(utc_offset_hours):02d}:00"

    return AnalysisTemporalContext(
        context_type="analysis_session",
        circadian_quarter=q,
        circadian_state=state,
        season=_season(now.date(), hemisphere),
        hemisphere=hemisphere,
        latitude_band=latitude_band,
        utc_reference=now.isoformat(),
        extraction_frame_label=frame_label,
        analyst_note=note,
    )


def build_track_creation_context(
    release_year: int,
    release_month: Optional[int] = None,
    known_session_hour: Optional[float] = None,
    known_hemisphere: Literal["north", "south", "equatorial"] = "north",
    known_latitude_band: Optional[int] = None,
    note: str = "",
) -> AnalysisTemporalContext:
    """
    Build a best-estimate temporal context for when a track was created.

    release_year/month are the best available proxy for creation time
    when session metadata is unavailable.

    known_session_hour: if session logs or artist statements give an hour
    in local time, pass it here.
    """
    month = release_month or 6   # Default to midsummer if month unknown
    try:
        approximate_date = date(release_year, month, 15)
    except ValueError:
        approximate_date = date(release_year, 6, 15)

    season = _season(approximate_date, known_hemisphere)

    if known_session_hour is not None and known_latitude_band is not None:
        q, state = _circadian_quarter(
            known_session_hour, known_latitude_band, approximate_date
        )
    else:
        # Unknown session time: assign to Q3 (afternoon-activation) as neutral default
        q, state = 3, "afternoon-activation"

    return AnalysisTemporalContext(
        context_type="track_creation",
        circadian_quarter=q,
        circadian_state=state,
        season=season,
        hemisphere=known_hemisphere,
        latitude_band=known_latitude_band,
        utc_reference=f"{release_year}-{month:02d}-15T12:00:00+00:00",
        extraction_frame_label=f"UTC+00:00 (estimated)",
        analyst_note=note or f"Created ~{release_year}; session time unknown",
    )
