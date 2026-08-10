"""
Download the public motor-cortex datasets required to reproduce Figure 1
(burst fraction encodes reach direction more selectively than tonic spike rate).

Both datasets are openly available from the DANDI Archive:
  - MC_Maze  (dandiset 000128, monkey Jenkins, Stanford; delayed centre-out reaching)
  - MC_RTT   (dandiset 000129, monkey Indy, UCSF;   self-paced random-target reaching)

Files are saved next to this script, under the names the analysis scripts expect:
mc_maze_train.nwb and mc_rtt_train.nwb. Each is several hundred MB / tens of
MB; the download runs once and is then cached.
"""
from pathlib import Path

from dandi.dandiapi import DandiAPIClient

HERE = Path(__file__).resolve().parent

DATASETS = [
    # (dandiset, asset path within dandiset, local filename)
    ('000128', 'sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb',
     'mc_maze_train.nwb'),
    ('000129', 'sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb',
     'mc_rtt_train.nwb'),
]


def fetch(dandiset, asset_path, out):
    out = str(HERE / out)          # the analysis scripts chdir here
    if Path(out).exists():
        print(f'  [cached] {out}')
        return
    print(f'Downloading dandiset {dandiset}: {asset_path}')
    with DandiAPIClient() as client:
        ds = client.get_dandiset(dandiset, 'draft')
        try:
            asset = ds.get_asset_by_path(asset_path)
        except Exception:
            # fall back: list train assets so the user can pick the right one
            print(f'  Asset path not found. Available "train" assets in {dandiset}:')
            for a in ds.get_assets():
                if 'train' in a.path:
                    print('   ', a.path)
            raise
        asset.download(out)
    print(f'  saved {out}')


if __name__ == '__main__':
    for d, path, out in DATASETS:
        fetch(d, path, out)
    print('\nAll datasets ready. Now run:  python burst_fraction_combined.py')
