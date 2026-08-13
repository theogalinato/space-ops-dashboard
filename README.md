## Progress Log

Continuing my Space Operations Dashboard project (see project instructions).

Days 1–9 complete and committed/pushed:
* Day 1: iss_position.py — live single-satellite tracking (ISS)
* Day 2–3: load_satellites.py — bulk load CelesTrak 'visual' group (157 sats),
  dataframe with lat/lon/altitude/inclination_deg/period_min/speed_km_s
* Day 4: plot_map.py — Plotly scattergeo map, RADARSAT-2 highlighted
* Day 5: ground_track.py — RADARSAT-2 ground track, next 100 min
* Day 6: refactored into satellite_data.py — shared module with
  load_tle_group, get_satellite_by_catnr, compute_subpoints,
  compute_ground_track, compute_orbital_params
* Day 7: data/canadian_assets.csv — curated catalogue (RADARSAT-2, RCM-1/2/3,
  Sapphire, NEOSSat, SCISAT-1, Anik F2/F3) with catnr/cospar_id/operator/
  category/orbit_regime/altitude_km/inclination_deg/launch_date/status/purpose
* Day 8: app.py — first Streamlit app, satellite selectbox, live map via
  st.plotly_chart, orbital-param metrics
* Day 9: app.py extended — search/filter, altitude + orbit-regime
  classification (LEO/MEO/GEO), TLE-age staleness warning (>3 days flagged)

Real findings worth remembering:
- RADARSAT-2 (CATNR 32382) is NOT in CelesTrak's "visual" group. Confirmed
  via testing: RCM and Sapphire are ALSO likely absent from "visual" (not
  yet verified for all three, but RADARSAT-2's absence + typical CelesTrak
  curation makes it likely). get_satellite_by_catnr() has a fallback for
  this: search a provided group first, else direct CATNR query.
- RCM catalog numbers, confirmed via get_satellite_by_catnr(): RCM-1=44322,
  RCM-2=44324, RCM-3=44323. NOT sequential with launch order — all three
  launched together on one Falcon 9, catalog order != deployment order.
- Watch for stale TLE propagation: SGP4 accuracy degrades fast past ~a
  few days from TLE epoch. Now surfaced in the UI as a warning (Day 9).
- .gitignore covers venv/, __pycache__/, *.tle, *.html, gp.php (TLE cache
  files and Plotly HTML outputs are regenerable, never committed).

KNOWN GAP going into today: the "visual" group app currently can't display
Canadian defence/EO assets (RADARSAT-2, RCM, Sapphire) because they're not
members of "visual." The catalogue CSV exists but isn't wired into the app
yet — that's Day 12's job (integrate catalogue as filterable table), not
today's. Flagging so it's not a surprise.

Starting Day 10 now: Passes part 1 — learn find_events; compute rise/
culminate/set times for one satellite over Ottawa. This is topocentric
geometry (observer-relative), a new Skyfield concept vs. the geocentric
subpoint work so far. Let's start with just one satellite (probably
RADARSAT-2, given the Canadian focus) over Ottawa before generalizing to
all five cities on Day 11.