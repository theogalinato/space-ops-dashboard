from skyfield.api import load, wgs84
import pandas as pd
import math

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
        
        vx, vy, vz = geocentric.velocity.km_per_s
        speed_km_s = (vx**2 + vy**2 + vz**2) ** 0.5
        
        period_min = (2 * math.pi) / sat.model.no_kozai
        
        rows.append({
            'name': sat.name,
            'norad_id': sat.model.satnum,
            'lat_deg': subpoint.latitude.degrees,
            'lon_deg': subpoint.longitude.degrees,
            'alt_km': subpoint.elevation.km,
            'inclination_deg': math.degrees(sat.model.inclo),
            'period_min': period_min,
            'speed_km_s': speed_km_s,
        })
    except Exception as e:
        print(f'Skipped {sat.name}: {e}')

df = pd.DataFrame(rows)
print(df.shape)
print(df[['name', 'lat_deg', 'inclination_deg']].head(10))
print(df['alt_km'].describe())
