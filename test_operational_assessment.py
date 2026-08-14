"""
Offline sanity test for operational_assessment.py.

Pure lookups -- no network calls -- so, like test_space_weather_status.py,
this covers the whole module. Checks that every status level for both
hazards resolves to a complete assessment (all three domain fields
populated) and carries the right hazard/level tags back.
"""

from space_weather_status import STATUS_LEVELS, StatusResult
from operational_assessment import assess_geomagnetic_impact, assess_radio_blackout_impact

print("=== assess_geomagnetic_impact ===")
for level in STATUS_LEVELS:
    result = assess_geomagnetic_impact(StatusResult(level, "test basis"))
    assert result.hazard == "Geomagnetic activity", f"{level}: wrong hazard tag {result.hazard!r}"
    assert result.level == level, f"expected level {level}, got {result.level}"
    for field_name, value in [("gnss", result.gnss), ("hf_radio", result.hf_radio), ("sat_ops", result.sat_ops)]:
        assert isinstance(value, str) and len(value) > 0, f"{level}: {field_name} is empty"
    print(f"  {level:9s} gnss:     {result.gnss}")
    print(f"  {'':9s} hf_radio: {result.hf_radio}")
    print(f"  {'':9s} sat_ops:  {result.sat_ops}")
print("  ALL PASS\n")

print("=== assess_radio_blackout_impact ===")
for level in STATUS_LEVELS:
    result = assess_radio_blackout_impact(StatusResult(level, "test basis"))
    assert result.hazard == "Radio blackout risk", f"{level}: wrong hazard tag {result.hazard!r}"
    assert result.level == level, f"expected level {level}, got {result.level}"
    for field_name, value in [("gnss", result.gnss), ("hf_radio", result.hf_radio), ("sat_ops", result.sat_ops)]:
        assert isinstance(value, str) and len(value) > 0, f"{level}: {field_name} is empty"
    print(f"  {level:9s} gnss:     {result.gnss}")
    print(f"  {'':9s} hf_radio: {result.hf_radio}")
    print(f"  {'':9s} sat_ops:  {result.sat_ops}")
print("  ALL PASS\n")

print("ALL CHECKS PASSED")