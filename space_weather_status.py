"""
space_weather_status.py

Day 16: rule-based LOW / MODERATE / HIGH status classification, built on
top of the raw NOAA SWPC data space_weather.py (Day 15) already ingests.

This module answers one narrow question: "on a simple three-band scale,
how active is space weather right now?" It does not say what that means
for GNSS, HF radio, or satellite operations -- that plain-language "so
what" translation is Day 17's protected scope, and it will import and
build on top of the classifications this module produces rather than
duplicate them.

Two independent hazards are classified here, each a direct reduction of
an existing public NOAA operational scale (nothing invented):

  - Geomagnetic activity, from the planetary Kp index, reduced from
    NOAA's five-level G-scale (G1 Minor through G5 Extreme).
  - Radio blackout risk, from the GOES X-ray flare class, reduced from
    NOAA's five-level R-scale (R1 Minor through R5 Extreme).

Solar wind speed, density, and Bz (already ingested by space_weather.py)
are deliberately NOT classified into a status band here. NOAA's own scale
for solar-wind-driven hazards, the S-scale (solar radiation storms), is
defined against >=10 MeV proton flux -- a measurement this project does
not ingest. Inventing a speed/Bz threshold with no public scale behind it
would look like a real classification without being one, which conflicts
with this project's honesty constraints (simplified educational
assessment, not a real warning system). Solar wind stays informational
only in the Space Weather tab; Day 17 can still use Bz as a contributing
factor in the operational narrative without this module claiming a
formal severity band for it.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUS_LEVELS = ("LOW", "MODERATE", "HIGH")

# NOAA G-scale is defined directly on Kp: G1 starts at Kp 5, G3 starts at
# Kp 7. This project collapses the five NOAA levels (G1-G5) into two
# boundaries, giving three bands instead of five.
GEOMAGNETIC_MODERATE_KP = 5.0
GEOMAGNETIC_HIGH_KP = 7.0


@dataclass(frozen=True)
class StatusResult:
    """
    One classified hazard.

    level is one of STATUS_LEVELS. basis is a short, plain, FACTUAL string
    (the number and the threshold it crossed) -- deliberately not an
    operational interpretation. "Kp 7.20, at or above the G3 threshold
    (Kp 7)" is Day 16. "HF radio blackouts likely on the sunlit side"
    would be Day 17 -- this module never produces that kind of sentence.
    """
    level: str
    basis: str


def classify_geomagnetic_status(kp: float) -> StatusResult:
    """
    Reduce the planetary Kp index to LOW / MODERATE / HIGH.

    NOAA's G-scale:
      Kp 5: G1 (Minor)      Kp 6: G2 (Moderate)
      Kp 7: G3 (Strong)     Kp 8: G4 (Severe)      Kp 9: G5 (Extreme)
    Below Kp 5 is quiet-to-active but below any officially named storm
    level.

    This project's three bands:
      LOW       Kp <  5   (below G1)
      MODERATE  5 <= Kp < 7   (G1-G2)
      HIGH      Kp >= 7   (G3 and above)

    Pass the real-valued estimated_kp (not the integer kp_index) so a
    reading like 4.83 -- clearly headed toward G1 -- doesn't get rounded
    away before the threshold check sees it.
    """
    if kp >= GEOMAGNETIC_HIGH_KP:
        return StatusResult(
            "HIGH", f"Kp {kp:.2f}, at or above the G3 strong-storm threshold (Kp {GEOMAGNETIC_HIGH_KP:.0f})"
        )
    elif kp >= GEOMAGNETIC_MODERATE_KP:
        return StatusResult(
            "MODERATE", f"Kp {kp:.2f}, at or above the G1 minor-storm threshold (Kp {GEOMAGNETIC_MODERATE_KP:.0f})"
        )
    else:
        return StatusResult(
            "LOW", f"Kp {kp:.2f}, below the G1 minor-storm threshold (Kp {GEOMAGNETIC_MODERATE_KP:.0f})"
        )


def classify_radio_blackout_status(flare_class: str) -> StatusResult:
    """
    Reduce the GOES X-ray flare class to LOW / MODERATE / HIGH.

    NOAA's R-scale (radio blackout) correlates closely with flare class:
      R1-R2 (Minor-Moderate): M-class flares
      R3-R5 (Strong-Extreme): X-class flares
    Below M-class, there is no categorized radio blackout.

    flare_class is the string space_weather.classify_xray_flare already
    produces, e.g. "M5.3", "X1.8", "B3.4", or "below A" -- this function
    only looks at the leading letter, so it works for any value that
    function can return.
    """
    if flare_class.startswith("X"):
        return StatusResult("HIGH", f"{flare_class}-class flare, in the R3+ (strong or greater) blackout range")
    elif flare_class.startswith("M"):
        return StatusResult(
            "MODERATE", f"{flare_class}-class flare, in the R1-R2 (minor-to-moderate) blackout range"
        )
    else:
        return StatusResult("LOW", f"{flare_class}-class flare, below the M-class radio blackout threshold")