"""
Offline sanity test for space_weather_status.py.

Unlike test_space_weather.py, everything in this module is pure logic --
no network calls at all -- so this test covers the whole module, not just
part of it.
"""

from space_weather_status import classify_geomagnetic_status, classify_radio_blackout_status

print("=== classify_geomagnetic_status ===")
# One case comfortably inside each band, plus both boundaries exactly at
# the threshold (G-scale boundaries are defined as >=, not >).
kp_cases = {
    2.33: "LOW",
    4.99: "LOW",
    5.00: "MODERATE",
    6.50: "MODERATE",
    7.00: "HIGH",
    8.83: "HIGH",
}
for kp, expected in kp_cases.items():
    result = classify_geomagnetic_status(kp)
    assert result.level == expected, f"kp={kp}: expected {expected}, got {result.level}"
    print(f"  kp={kp:.2f} -> {result.level:9s} ({result.basis})")
print("  ALL PASS\n")

print("=== classify_radio_blackout_status ===")
# One case per letter class, including the two below-M classes that both
# resolve to LOW, plus the "below A" case classify_xray_flare can return.
flare_cases = {
    "below A": "LOW",
    "A5.0": "LOW",
    "B3.4": "LOW",
    "C2.1": "LOW",
    "M1.0": "MODERATE",
    "M5.3": "MODERATE",
    "X1.8": "HIGH",
    "X9.9": "HIGH",
}
for flare_class, expected in flare_cases.items():
    result = classify_radio_blackout_status(flare_class)
    assert result.level == expected, f"flare_class={flare_class}: expected {expected}, got {result.level}"
    print(f"  flare_class={flare_class:8s} -> {result.level:9s} ({result.basis})")
print("  ALL PASS\n")

print("ALL CHECKS PASSED")