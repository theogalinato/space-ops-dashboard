## Progress Log

- **Day 1:** Live single-satellite tracking (ISS) — `iss_position.py`
- **Day 2:** Bulk TLE load, propagate CelesTrak `visual` group into a Pandas dataframe — `load_satellites.py`
- **Day 3:** Added altitude, inclination, period, speed columns — validated against orbital physics
- **Day 4:** First map — Plotly `scattergeo` of all subpoints, RADARSAT-2 highlighted — `plot_map.py`
  - Note: `fig.show()` hangs in this environment (local server issue); using `fig.write_html()` instead. Will switch to `st.plotly_chart()` once Streamlit is in place (Day 8).