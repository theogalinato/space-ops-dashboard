"""
Offline sanity test for space_weather.py's pure parsing and classification
logic, using hardcoded sample records captured directly from the live
SWPC feeds on 2026-08-13. Same reasoning as test_satellite_data.py: this
sandbox cannot reach services.swpc.noaa.gov, so the network calls
(_fetch_json, and therefore get_kp_index/get_xray_flux/get_solar_wind
themselves) still need a live run on your end. This validates the
functions that do NOT touch the network: classify_xray_flare and
_latest_active_series.
"""

from space_weather import classify_xray_flare, _latest_active_series

print("=== classify_xray_flare ===")
# Thresholds and one representative value per class, plus a below-A case.
cases = {
    5e-9: "below A",
    5e-8: "A5.0",
    3.4e-7: "B3.4",
    2.1e-6: "C2.1",
    5.3e-5: "M5.3",
    1.8e-4: "X1.8",
}
for flux, expected in cases.items():
    result = classify_xray_flare(flux)
    assert result == expected, f"flux={flux:.1e}: expected {expected}, got {result}"
    print(f"  flux={flux:.1e} W/m^2 -> {result}  [PASS]")
print("  ALL PASS\n")

print("=== _latest_active_series ===")
# Real shape captured from rtsw_mag_1m.json: two sources (SOLAR1 primary,
# ACE backup) reporting at overlapping timestamps, only one active at a
# time, and not guaranteed to arrive in strict time order.
sample_mag_records = [
    {"time_tag": "2026-08-13T20:53:00", "active": True, "source": "SOLAR1", "bt": 3.99, "bz_gsm": -2.15},
    {"time_tag": "2026-08-13T20:53:00", "active": False, "source": "ACE", "bt": 4.86, "bz_gsm": -2.38},
    {"time_tag": "2026-08-13T20:52:00", "active": True, "source": "SOLAR1", "bt": 4.83, "bz_gsm": -2.81},
]
result_df = _latest_active_series(sample_mag_records, ["bt", "bz_gsm"])
print(result_df.to_string(index=False))

assert len(result_df) == 2, f"expected 2 active-only rows, got {len(result_df)}"
assert (result_df["source"] == "SOLAR1").all(), "an inactive ACE row leaked through"
assert list(result_df["time_utc"]) == sorted(result_df["time_utc"]), "rows are not sorted oldest to newest"
print("  PASS: inactive source dropped, sorted oldest to newest\n")

print("ALL CHECKS PASSED")