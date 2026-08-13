from skyfield.api import load, wgs84
import pandas as pd

ts = load.timescale()

# GROUP=visual: ~100 bright/trackable objects — good size for pipeline testing
url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle'
satellites = load.tle_file(url, filename='visual.tle', reload=False)
print(f'Loaded {len(satellites)} satellites')

t = ts.now()

rows = []
for sat in satellites:
    try:
        geocentric = sat.at(t)
        subpoint = wgs84.subpoint(geocentric)
        rows.append({
            'name': sat.name,
            'norad_id': sat.model.satnum,
            'lat_deg': subpoint.latitude.degrees,
            'lon_deg': subpoint.longitude.degrees,
            'alt_km': subpoint.elevation.km,
        })
    except Exception as e:
        print(f'Skipped {sat.name}: {e}')

df = pd.DataFrame(rows)
print(df.shape)
print(df.head(10))
print(df['alt_km'].describe())