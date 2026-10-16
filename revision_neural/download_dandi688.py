"""Download the centre-out (CO) sessions of one subject of DANDI 000688 into
dandi000688_<sub>_cache/ (resumable curl). Asset list: dandi000688_assets.json
(api.dandiarchive.org/api/dandisets/000688/versions/draft/assets/?page_size=1000).

Usage: python download_dandi688.py C M T J
Monkey C: the 15 largest centre-out sessions (as in the paper); M, T, J: all centre-out sessions.
Writes <cache>/DONE when every file of the subject has the expected size."""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
assets = json.loads((ROOT / "dandi000688_assets.json").read_text())["results"]

for sub in sys.argv[1:]:
    cache = ROOT / f"dandi000688_{sub}_cache"; cache.mkdir(exist_ok=True)
    rows = sorted([a for a in assets if a["path"].startswith(f"sub-{sub}/") and "_ses-CO-" in a["path"]],
                  key=lambda a: -a["size"])
    if sub == "C":
        rows = rows[:15]
    print(f"{sub}: {len(rows)} CO sessions, {sum(a['size'] for a in rows)/1e9:.2f} GB", flush=True)
    for a in rows:
        dest = cache / Path(a["path"]).name
        if dest.exists() and dest.stat().st_size == a["size"]:
            continue
        url = f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
        for attempt in range(8):
            t0 = time.time()
            subprocess.run(["curl", "-sL", "-C", "-", "--retry", "5", "--retry-delay", "30", "--max-time", "3600",
                            "-o", str(dest), url])
            if dest.exists() and dest.stat().st_size == a["size"]:
                print(f"  {dest.name} {a['size']/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
                break
            print(f"  {dest.name}: size {dest.stat().st_size if dest.exists() else 0} != {a['size']}, retry {attempt+1}", flush=True)
            time.sleep(60)
        else:
            print(f"  FAILED {dest.name}", flush=True)
    ok = all((cache / Path(a["path"]).name).exists() and (cache / Path(a["path"]).name).stat().st_size == a["size"] for a in rows)
    if ok:
        (cache / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M"))
    print(f"{sub}: {'complete' if ok else 'INCOMPLETE'}", flush=True)
