"""
operational_assessment.py

Day 17 (protected): the "so what" engine. Turns Day 16's LOW/MODERATE/HIGH
space weather status into plain-language operational impact statements,
covering the three domains this project's own MVP scope names: GNSS
positioning/timing, HF radio propagation, and satellite operations.

This is still a SIMPLIFIED EDUCATIONAL ASSESSMENT, not a real space
weather warning system -- DISCLAIMER below says so explicitly, and app.py
displays it next to every assessment. The point of this module is to
demonstrate space operations thinking (turning a classification into an
operator-relevant judgment), not to reproduce an official NOAA product.

Every impact statement below is a condensed, hedged paraphrase of NOAA's
own public G-scale (geomagnetic storms) and R-scale (radio blackouts)
operational impact tables, reduced from their native five levels (G1-G5,
R1-R5) down to this project's three bands (LOW/MODERATE/HIGH) -- the same
reduction space_weather_status.py already made for the classification
itself. Nothing here is invented. Where a three-band MODERATE folds two
NOAA levels together (G1+G2, or R1+R2), the more conservative
(lower-severity) language is used, and wording is hedged ("possible,"
"likely," never "will") to reflect that this project classifies on a
3-band reduction of a 5-band scale, not the full scale itself.

Solar wind/Bz still gets no assessment here, for the same reason it gets
no status band in space_weather_status.py: with no ingested proton flux
data, there is no honest scale to grade its severity against. It remains
available in the Space Weather tab as raw, informational data only.
"""

from __future__ import annotations

from dataclasses import dataclass

from space_weather_status import StatusResult

DISCLAIMER = (
    "Simplified educational operational assessment, not a real space "
    "weather warning system. Derived from NOAA's public G-scale and "
    "R-scale impact tables, reduced to three bands -- consult NOAA SWPC "
    "directly (swpc.noaa.gov) for official space weather products."
)


@dataclass(frozen=True)
class OperationalAssessment:
    """
    Plain-language "so what" for one hazard at one status level, split
    across the three operational domains this project's scope names:
    GNSS positioning/timing, HF radio propagation, and satellite
    operations (drag, charging, attitude).
    """

    hazard: str
    level: str
    gnss: str
    hf_radio: str
    sat_ops: str


_GEOMAGNETIC_IMPACT: dict[str, OperationalAssessment] = {
    "LOW": OperationalAssessment(
        hazard="Geomagnetic activity",
        level="LOW",
        gnss="Nominal -- no geomagnetically driven GNSS accuracy degradation expected.",
        hf_radio="Nominal -- no geomagnetically driven HF propagation effects expected.",
        sat_ops="Nominal -- no elevated atmospheric drag or spacecraft charging risk beyond baseline.",
    ),
    "MODERATE": OperationalAssessment(
        hazard="Geomagnetic activity",
        level="MODERATE",
        gnss=(
            "Possible minor GNSS positioning error at high latitudes as ionospheric "
            "density shifts; mid-latitude GNSS not typically affected at this level."
        ),
        hf_radio=(
            "HF propagation may be degraded at high latitudes and in the auroral "
            "zone; mid-latitude HF links largely unaffected."
        ),
        sat_ops=(
            "Slight increase in LEO atmospheric drag -- orbit predictions may need "
            "refinement; minor surface-charging risk for high-inclination and "
            "polar-orbiting spacecraft."
        ),
    ),
    "HIGH": OperationalAssessment(
        hazard="Geomagnetic activity",
        level="HIGH",
        gnss=(
            "GNSS positioning and timing accuracy can be degraded, particularly at "
            "high latitudes, with intermittent effects possible even at mid-latitudes."
        ),
        hf_radio=(
            "HF propagation likely intermittent or blacked out at high latitudes "
            "and in the auroral zone for the storm's duration."
        ),
        sat_ops=(
            "Noticeably increased LEO atmospheric drag raises orbit-prediction "
            "uncertainty; elevated surface-charging risk and possible "
            "attitude/orientation anomalies, especially for high-inclination and "
            "polar-orbiting spacecraft."
        ),
    ),
}

_RADIO_BLACKOUT_IMPACT: dict[str, OperationalAssessment] = {
    "LOW": OperationalAssessment(
        hazard="Radio blackout risk",
        level="LOW",
        gnss="Nominal -- no flare-driven GNSS signal effects expected.",
        hf_radio="Nominal -- no flare-driven HF propagation effects expected.",
        sat_ops="Nominal -- no flare-driven spacecraft effects expected.",
    ),
    "MODERATE": OperationalAssessment(
        hazard="Radio blackout risk",
        level="MODERATE",
        gnss=(
            "Brief GNSS signal degradation possible on the sunlit side during the "
            "flare itself, typically on the order of minutes."
        ),
        hf_radio=(
            "HF radio on the sunlit side may experience brief fades or occasional "
            "loss of contact for tens of minutes."
        ),
        sat_ops=(
            "Minimal direct effect at this level; flare-associated radiation is a "
            "secondary consideration rather than a primary driver."
        ),
    ),
    "HIGH": OperationalAssessment(
        hazard="Radio blackout risk",
        level="HIGH",
        gnss=(
            "GNSS signal degradation likely on the sunlit side for the duration of "
            "the flare, potentially affecting positioning accuracy for tens of "
            "minutes."
        ),
        hf_radio=(
            "Wide-area HF radio blackout likely across the sunlit side, "
            "potentially lasting roughly an hour or more depending on flare "
            "duration."
        ),
        sat_ops=(
            "Elevated risk of communications interference and single-event "
            "effects for sunlit-side spacecraft during the event; ground-station "
            "HF links may be degraded."
        ),
    ),
}


def assess_geomagnetic_impact(status: StatusResult) -> OperationalAssessment:
    """Plain-language operational impact for a geomagnetic StatusResult."""
    return _GEOMAGNETIC_IMPACT[status.level]


def assess_radio_blackout_impact(status: StatusResult) -> OperationalAssessment:
    """Plain-language operational impact for a radio-blackout StatusResult."""
    return _RADIO_BLACKOUT_IMPACT[status.level]