# Space Operations Dashboard — v1.0

A public-data space domain awareness tool: a Streamlit dashboard that turns live satellite tracking, pass prediction, space weather, a Canadian space-asset catalogue, and reduced conjunction screening into a single operator-facing view of "what is happening in the space domain right now, and how can someone reading this understand it quickly."

Built as a 27-day solo project by a mechanical engineering student (University of New Brunswick, near graduation) to demonstrate space-operations thinking — data to system to environment to operational impact — not just coding ability.

## What this is, and isn't

This is a **public-data educational prototype**, built entirely from free, no-auth data sources (CelesTrak for orbital elements, NOAA SWPC for space weather). It is not a real military, intelligence, or operational CAF/3 Canadian Space Division system.

Two features carry real, stated limitations rather than implicit ones: the conjunction screening tab is a simplified screening heuristic on public TLE data, not a certified collision-probability (Pc) system — public TLEs carry no usable covariance, so a real Pc calculation isn't possible from this data, full stop. The space weather assessment is a simplified educational tool built from NOAA's own published thresholds, not a real space-weather warning system. Both are disclaimed on-screen, not just in this file.

## Features

- **Live satellite tracking** — TLEs loaded and propagated with Skyfield/SGP4, plotted on a 2D map or a true-altitude 3D globe (now with real coastline outlines), with search/select and live altitude/inclination/period/speed.
- **Next-pass prediction** — rise/culminate/set times, duration, and max elevation over five Canadian sites (Ottawa, Halifax, Edmonton, Vancouver, Yellowknife), with a static-visibility fallback for GEO satellites.
- **Canadian asset catalogue** — a curated, filterable table of Canadian space assets (RADARSAT-2, the RADARSAT Constellation Mission, Sapphire, NEOSSat, SCISAT, Anik/Telesat comms, and more).
- **Space weather + operational assessment** — NOAA SWPC data (Kp index, GOES X-ray flux, solar wind) reduced to a LOW/MODERATE/HIGH status with a plain-language "so what" for GNSS, HF radio, and satellite operations.
- **Reduced conjunction screening** — screen any Canadian catalogue satellite against the full tracked population, with a worst-case risk banner and a sortable results table.

## What's new since the MVP milestone

- A true 3D globe (ECI→ECEF conversion, `scatter3d`/`Surface`) replacing the flat map as the primary view, now with locally-bundled Natural Earth coastline outlines so it reads as a recognizable planet instead of a plain sphere — no CDN dependency for this feature.
- Real-time refresh via Streamlit fragments, with data cadence and display cadence handled independently per data source (TLEs cached 6h, space weather 60s, positions recomputed every 10s with no caching).
- A dark/light theme with the native System/Light/Dark switcher, colors validated with a colorblind + contrast checker rather than chosen by eye.
- A sidebar layout consolidating satellite selection and live metrics as persistent controls across every tab.

## Honest limitations, carried forward rather than hidden

- Conjunction screening uses a grid-sampled search over a time window, not a true continuous time-of-closest-approach — a fast, close crossing between two objects could in principle fall entirely between two samples at the default step size. This is disclosed in the tab itself, and the step size is user-adjustable so the limitation is demonstrable, not just asserted.
- The 2D map's land/ocean basemap depends on a third-party CDN (`cdn.plot.ly`) that isn't reachable from the development sandbox this project was built in, so its dark-mode styling was reasoned through rather than screenshot-verified during development — verify it looks right if you're relying on it.
- Solar wind speed/density/Bz are shown as raw data with no severity classification, since NOAA's own scale for solar-wind-driven hazards needs a measurement (>=10 MeV proton flux) this project doesn't ingest — no invented thresholds.

See `README.md` for the full day-by-day build log, design reasoning, and the complete list of open items.

## Tech stack

`skyfield` `pandas` `plotly` `streamlit` `requests` `numpy` — Skyfield handles all orbital mechanics (TLE parsing, SGP4 propagation, pass prediction); Plotly renders both the 2D map and 3D globe; Streamlit provides the dashboard shell, caching, and live-refresh scheduling.

## Screenshots

<!-- Add your screenshots here once captured, e.g.: -->
<!-- ![3D Globe with coastlines](docs/screenshots/v1.0_globe_3d_dark.png) -->
<!-- ![Conjunction Screening](docs/screenshots/v1.0_conjunction.png) -->