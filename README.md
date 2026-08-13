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
* Day 10: pass_prediction.py — topocentric rise/culminate/set for
  RADARSAT-2 over Ottawa via Skyfield find_events + altaz. Verified
  against live TLE: pass spacing matched ~100.7 min SSO period, duration
  scaled with max elevation as expected.
* Day 11: passes.py — generalized to five Canadian cities (Ottawa,
  Halifax, Edmonton, Vancouver, Yellowknife). Wired into app.py as a
  "Next Passes" section (city selectbox + N-passes slider + table).
  Fixed a bug where app.py had its own separate Timescale object from
  satellite_data.py's — now imports the shared `ts`.
* Day 12: catalogue.py — integrated the Day 7 Canadian asset catalogue
  CSV (9 satellites: RADARSAT-2, RCM-1/2/3, Sapphire, NEOSSat, SCISAT-1,
  Anik F2/F3) as both a filterable table AND live-trackable satellites
  (merged into the main selectbox/map/passes pool). Optimization: bulk
  fetch via CelesTrak's "active" group before falling back to individual
  CATNR queries.

Real findings worth remembering:
- RADARSAT-2 + all 3 RCM satellites are NOT in CelesTrak's "active"
  group (confirmed via cache file check: catnr_32382.tle, catnr_44322/
  44323/44324.tle all had to be individually fetched). Sapphire, NEOSSat,
  SCISAT-1, and both Aniks WERE in "active" (bulk-fetched, no individual
  cache files needed). So the bulk-fetch optimization caught 5/9
  satellites, cut network calls from 9 individual fetches down to
  1 bulk + 4 individual fallbacks.
- Data-quality bug caught and fixed: canadian_assets.csv had inconsistent
  category tags ("Sci/Defence" on NEOSSat vs "Science" on SCISAT-1) --
  same concept, different strings. Fixed the CSV to "Science/Defence".
  Also had to fix the category filter logic itself: it was treating
  compound tags like "EO/Defence" as one opaque string instead of
  splitting on "/" and matching either component -- this was silently
  hiding RCM-1/2/3 from an "EO"-only filter, which is exactly backwards
  for what the catalogue is supposed to demonstrate.
- Current files: satellite_data.py (TLE loading/orbital params, shared
  ts), passes.py (five-city pass prediction), catalogue.py (catalogue
  load + bulk/individual TLE fetch + merge into trackable list), app.py
  (Streamlit UI tying all three together: search/select, map, passes,
  catalogue table).

Starting Day 13 now: buffer/debug + layout cleanup (sidebar, tabs,
captions). Known cleanup items going in:
- app.py still has leftover "ADD #1/#2/#3" instructional comments from
  earlier sessions -- these read as scaffolding, not documentation, and
  should go before this looks like a finished file.
- Everything is currently in one long vertical scroll (search -> metrics
  -> map -> catalogue table -> passes). Worth deciding whether
  sidebar/tabs actually improve this or just add complexity for its own
  sake -- want your read on whether that's worth doing today or whether
  today should just be a straight bug hunt instead.