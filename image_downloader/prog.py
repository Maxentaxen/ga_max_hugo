import requests
import os
import time
from datetime import datetime, timedelta

WMS_URL = "https://view.eumetsat.int/geoserver/ows"
LAYER = "msg_fes:ir108"
# Start från första möjliga bilddatum (baserat på historisk data)
START_TIME = datetime(2022, 5, 20, 6, 0)  # justerat startdatum
END_TIME = datetime(2025, 10, 2, 0, 0)     # slutpunkt (exkluderande)
TIME_STEP = timedelta(minutes=15)
count = int(60056) # ändra till senasts bild som sparats innan start

def fetch_image_for_time(dt: datetime):
    """Försök hämta bilden för den exakta tiden dt."""
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": LAYER,
        "CRS": "EPSG:4326",
        "BBOX": "48,-10,74,35",
        "WIDTH": "512",
        "HEIGHT": "512",
        "FORMAT": "image/png",
        "TIME": dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    try:
        r = requests.get(WMS_URL, params=params, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Fel vid hämtning {dt}: {e}")
        return False

    if not r.headers.get("Content-Type", "").startswith("image/"):
        # Spara felinnehåll för debugging
        errfn = f"error_{dt.strftime('%Y%m%d_%H%M')}.html"
        with open(errfn, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"❌ Ingen bild tillgänglig vid {dt}, sparade fel till {errfn}")
        return False

    # Spara bilden i mappstruktur per år/månad/dag/timme
    folder = os.path.join(
        str(dt.year),
        f"{dt.month:02d}",
        f"{dt.day:02d}",
        f"{dt.hour:02d}"
    )
    global count
    count += 1
    os.makedirs(folder, exist_ok=True)
    filename = f"seviri_ir108_europe_{dt.strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(folder, filename)
    with open(filepath, "wb") as f:
        f.write(r.content)
    print(f"✅ Sparad bild nr:{count} för {dt} → {filepath}")
    return True

def crawl_all():
    dt = START_TIME
    count = 0
    while dt < END_TIME:
        success = fetch_image_for_time(dt)
        count += 1
        # valfritt: om framgång, ta kort paus för att undvika överbelastning
        time.sleep(0.1)  # 0.1 s paus; justera efter serverns kapacitet

        dt += TIME_STEP

    print(f"🚀 Klart! Försökt {count} tider från {START_TIME} till {END_TIME}")

if __name__ == "__main__":
    crawl_all()
