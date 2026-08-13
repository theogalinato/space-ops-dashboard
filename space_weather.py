"""
space_weather.py

NOAA SWPC (Space Weather Prediction Center) data ingestion for the Space
Operations Dashboard.

Pulls three data products, all free and requiring no authentication, from
services.swpc.noaa.gov:
  - Planetary Kp index (planetary_k_index_1m.json): a nowcast of global
    geomagnetic activity, updated every minute, on the standard 0 to 9 Kp
    scale used worldwide for space weather reporting.
  - GOES X-ray flux (goes/primary/xrays-6-hour.json): solar X-ray output
    in two wavelength bands from the GOES satellite. The long wavelength
    channel (0.1 to 0.8 nanometers) is the one the standard flare
    classification scale (A, B, C, M, X) is defined against.
  - Real time solar wind (json/rtsw/rtsw_wind_1m.json and
    rtsw_mag_1m.json): proton speed and density, plus the interplanetary
    magnetic field vector, both at 1 minute cadence.

Day 15 finding, worth remembering: this project's tech stack notes assumed
the older solar wind endpoints (products/solar-wind/plasma-*.json and
mag-*.json). NOAA deprecated those in April 2026 (Service Change Notice
26-21) and replaced them with the rtsw_wind_1m.json and rtsw_mag_1m.json
endpoints used below, which also renamed several fields (for example,
"density" became "proton_density" and "lon_gsm" became "phi_gsm"). This
was confirmed directly against the live endpoints, not assumed from
memory, which is exactly the caution the tech stack notes called for.

This module only fetches and shapes the data into clean DataFrames. It
does not classify anything as LOW, MODERATE, or HIGH (that is Day 16), and
it does not produce an operational "so what" assessment (that is the
protected Day 17 work). Caching is also deliberately not added here; that
belongs to Day 20 alongside the rest of the app's autorefresh behavior,
and adding it piecemeal today would just have to be redone then.
"""

from __future__ import annotations

import pandas as pd
import requests

KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"

REQUEST_TIMEOUT_S = 10


def _fetch_json(url: str) -> list[dict]:
    """
    Fetch and parse one SWPC JSON endpoint.

    Deliberately lets requests.exceptions and JSON decode errors propagate
    rather than swallowing them and returning an empty result. An
    operational dashboard that goes quietly blank on a network hiccup is
    worse than one that visibly errors, since a blank "LOW" reading and a
    genuinely low reading look identical to an operator.
    """
    response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def get_kp_index() -> pd.DataFrame:
    """
    Planetary Kp index, 1 minute cadence nowcast.

    Returns a DataFrame with columns time_utc, kp_index (integer 0 to 9 as
    reported by NOAA), estimated_kp (the finer grained real valued
    nowcast), and kp (NOAA's short string code, such as "3M"), sorted
    oldest to newest. Use .iloc[-1] for the current reading.
    """
    records = _fetch_json(KP_URL)
    df = pd.DataFrame(records)
    df["time_utc"] = pd.to_datetime(df["time_tag"])
    df = df.sort_values("time_utc").reset_index(drop=True)
    return df[["time_utc", "kp_index", "estimated_kp", "kp"]]


def classify_xray_flare(flux_w_m2: float) -> str:
    """
    Standard GOES X-ray flare class from long channel (0.1 to 0.8 nm) flux
    in watts per square meter.

    Each letter (A, B, C, M, X) covers one decade of flux. The number
    after the letter is how far into that decade the reading falls, so B3
    is three times the flux of B1, and just under a tenth the flux of C1.
    This is the standard scale used across solar physics and space
    weather reporting, not something invented for this project.
    """
    if flux_w_m2 < 1e-8:
        return "below A"
    elif flux_w_m2 < 1e-7:
        letter, floor = "A", 1e-8
    elif flux_w_m2 < 1e-6:
        letter, floor = "B", 1e-7
    elif flux_w_m2 < 1e-5:
        letter, floor = "C", 1e-6
    elif flux_w_m2 < 1e-4:
        letter, floor = "M", 1e-5
    else:
        letter, floor = "X", 1e-4
    magnitude = flux_w_m2 / floor
    return f"{letter}{magnitude:.1f}"


def get_xray_flux() -> pd.DataFrame:
    """
    GOES long channel (0.1 to 0.8 nm) X-ray flux, roughly 1 minute
    cadence, covering the last 6 hours.

    The raw feed interleaves two wavelength bands (0.05 to 0.4 nm, the
    short channel, and 0.1 to 0.8 nm, the long channel) as separate
    records sharing the same timestamp. Only the long channel is kept
    here, since that is the channel the A/B/C/M/X flare scale is defined
    against.

    Returns a DataFrame with columns time_utc, flux_w_m2, and
    flare_class, sorted oldest to newest.
    """
    records = _fetch_json(XRAY_URL)
    df = pd.DataFrame(records)
    df = df[df["energy"] == "0.1-0.8nm"].copy()
    df["time_utc"] = pd.to_datetime(df["time_tag"])
    df = df.sort_values("time_utc").reset_index(drop=True)
    df["flare_class"] = df["flux"].apply(classify_xray_flare)
    return df.rename(columns={"flux": "flux_w_m2"})[
        ["time_utc", "flux_w_m2", "flare_class"]
    ]


def _latest_active_series(records: list[dict], value_columns: list[str]) -> pd.DataFrame:
    """
    Shared helper for the two rtsw_*_1m.json feeds.

    Both report from multiple instruments (currently SOLAR1 as the
    primary source and ACE as backup) at overlapping timestamps, each
    record flagged with an "active" boolean for which source NOAA is
    treating as authoritative at that moment. Keeping only the active
    rows avoids speed, density, or Bz appearing to jump around from two
    different instruments reporting slightly different values for the
    same minute.
    """
    df = pd.DataFrame(records)
    df = df[df["active"] == True].copy()  # noqa: E712 -- explicit beats implicit truthiness here
    df["time_utc"] = pd.to_datetime(df["time_tag"])
    df = df.sort_values("time_utc").drop_duplicates("time_utc").reset_index(drop=True)
    return df[["time_utc", "source"] + value_columns]


def get_solar_wind() -> pd.DataFrame:
    """
    Real time solar wind: proton speed and density, plus the
    interplanetary magnetic field.

    Bz (the north/south component of the field, in the GSM coordinate
    frame) is the single most operationally important number here. A
    sustained southward (negative) Bz is what actually couples solar wind
    energy into Earth's magnetosphere and drives geomagnetic storms; a
    fast, dense solar wind with northward Bz is comparatively
    uneventful. This mirrors a control surface deflection enabling
    coupling into a coupled system, rather than raw dynamic pressure
    alone doing the work.

    Merges the separate plasma (rtsw_wind_1m) and magnetometer
    (rtsw_mag_1m) feeds on time_utc, since they come from independent
    instruments on the same spacecraft reporting on roughly the same
    1 minute cadence, but not guaranteed to align to the exact second.

    Returns a DataFrame with columns time_utc, proton_speed_km_s,
    proton_density_n_cm3, bz_gsm_nt, and bt_nt, sorted oldest to newest.
    """
    wind_records = _fetch_json(WIND_URL)
    mag_records = _fetch_json(MAG_URL)

    wind_df = _latest_active_series(wind_records, ["proton_speed", "proton_density"])
    mag_df = _latest_active_series(mag_records, ["bz_gsm", "bt"])

    merged = pd.merge_asof(
        wind_df.sort_values("time_utc"),
        mag_df.sort_values("time_utc"),
        on="time_utc",
        direction="nearest",
        tolerance=pd.Timedelta("2min"),
        suffixes=("", "_mag"),
    )
    return merged.rename(columns={
        "proton_speed": "proton_speed_km_s",
        "proton_density": "proton_density_n_cm3",
        "bz_gsm": "bz_gsm_nt",
        "bt": "bt_nt",
    })[["time_utc", "proton_speed_km_s", "proton_density_n_cm3", "bz_gsm_nt", "bt_nt"]]