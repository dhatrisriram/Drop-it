"""
download_videos.py
──────────────────
Downloads the 5 serve videos provided in the assignment
into a local ./videos/ directory.
"""

import os
import urllib.request

VIDEOS = {
    "serve_01.mp4": "https://prod-assets.drop-it.app/cdb9cc9e-8a70-4752-a149-80664cded093.mp4",
    "serve_02.mp4": "https://prod-assets.drop-it.app/a484e70e-a1fd-42dd-9d2f-5d9f7b232a63.mp4",
    "serve_03.mp4": "https://prod-assets.drop-it.app/75a0b737-c6d1-4669-8bcf-8b6a8c09f97e.mp4",
    "serve_04.mp4": "https://prod-assets.drop-it.app/1d746ce2-15f1-4944-be96-bb899d5bc2b6.mp4",
    "serve_05.mp4": "https://prod-assets.drop-it.app/3d2c52c0-a110-4dfb-84fa-8738b64a2046.mp4",
}

# Mimic a real browser so the CDN doesn't block the request
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer":    "https://drop-it.app/",
    "Accept":     "*/*",
}

def download_all(out_dir: str = "videos"):
    os.makedirs(out_dir, exist_ok=True)
    for filename, url in VIDEOS.items():
        dest = os.path.join(out_dir, filename)
        if os.path.exists(dest):
            print(f"  ✓  Already exists: {dest}")
            continue
        print(f"  ↓  Downloading {filename} ...", end=" ", flush=True)
        try:
            req  = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            size = os.path.getsize(dest) / (1024 * 1024)
            print(f"done ({size:.1f} MB)")
        except Exception as e:
            print(f"FAILED: {e}")
            print(f"       → Try opening the URL directly in your browser and saving manually:"
                  f"\n         {url}")

if __name__ == "__main__":
    download_all()
    print("\nAll videos saved to ./videos/")
    print("Run detection with:")
    print("  cd videos && python ../detect_contact.py --all --annotate")