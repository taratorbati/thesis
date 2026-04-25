"""
HOW TO RE-DOWNLOAD NASA POWER DATA WITH T2M_MIN
================================================

Option 1: Direct URL (paste in your browser)
─────────────────────────────────────────────
https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,GWETROOT,GWETTOP,ALLSKY_SFC_SW_DWN,RH2M,WS2M&community=AG&longitude=48.847&latitude=38.298&start=20200101&end=20260412&format=CSV

Note: I added T2M_MIN, RH2M (relative humidity), and WS2M (wind speed).
      RH2M and WS2M would let you use the Penman-Monteith equation later
      if your committee requests it — it costs nothing to download them now.

Option 2: Python script (run on your own machine)
──────────────────────────────────────────────────
"""

import requests

url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,GWETROOT,GWETTOP,ALLSKY_SFC_SW_DWN,RH2M,WS2M",
    "community": "AG",
    "longitude": 48.847,
    "latitude": 38.298,
    "start": "20200101",
    "end": "20260412",
    "format": "CSV",
}

print("Downloading from NASA POWER...")
response = requests.get(url, params=params, timeout=120)

if response.status_code == 200:
    filename = "POWER_Daily_38.298N_48.847E_with_T2M_MIN.csv"
    with open(filename, "w") as f:
        f.write(response.text)
    print(f"Saved to {filename}")
    print(f"File size: {len(response.text)} bytes")
else:
    print(f"Error: HTTP {response.status_code}")
    print(response.text[:500])

"""
Option 3: NASA POWER web interface
───────────────────────────────────
1. Go to https://power.larc.nasa.gov/data-access-viewer/
2. Select "POWER Daily" → "Agroclimatology"
3. Enter coordinates: 38.298, 48.847
4. Date range: 2020-01-01 to 2026-04-12
5. Select parameters:
   - T2M, T2M_MAX, T2M_MIN
   - PRECTOTCORR
   - GWETROOT, GWETTOP
   - ALLSKY_SFC_SW_DWN
   - RH2M, WS2M  (bonus — for Penman-Monteith if needed later)
6. Download as CSV

After downloading, upload the new file here and I'll continue the analysis.
"""
