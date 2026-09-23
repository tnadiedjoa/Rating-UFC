"""
Refresh the two Kaggle snapshots the pipeline is built on.

- ``neelagiriaditya/ufc-datasets-1994-2025`` (CC0): a regularly updated mirror
  of ufcstats.com with fight totals, round-by-round stats, results and
  fighter profiles.
- ``mdabbert/ultimate-ufc-dataset`` (CC BY 4.0): betting odds and official
  rankings at fight time, from 2010 onwards.

A snapshot of the files is versioned in ``data/raw`` so the project runs
without a Kaggle account. Refreshing needs the Kaggle API credentials
(``~/.kaggle/kaggle.json``); without them the snapshot is used as is.
"""

import shutil
import tempfile
from pathlib import Path

from ufc_rating.config import ODDS_CSV, UFCSTATS_CSV, UFCSTATS_ROUNDS_CSV

# source -> (Kaggle dataset, {file in the dataset: local copy})
SOURCES = {
    "ufcstats": ("neelagiriaditya/ufc-datasets-1994-2025",
                 {"master.csv": UFCSTATS_CSV, "round.csv": UFCSTATS_ROUNDS_CSV}),
    "odds": ("mdabbert/ultimate-ufc-dataset", {"ufc-master.csv": ODDS_CSV}),
}


def download_sources() -> dict:
    """
    Download the latest version of each source over the versioned snapshot.
    Returns {source: True/False}; never raises, so the pipeline can go on
    with the snapshot when Kaggle is unreachable.
    """
    status = {name: False for name in SOURCES}
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:  # missing package, missing credentials, network
        print(f"  Kaggle unavailable ({type(exc).__name__}: {exc}); using the versioned snapshot.")
        return status

    for name, (dataset, files) in SOURCES.items():
        try:
            with tempfile.TemporaryDirectory() as tmp:
                api.dataset_download_files(dataset, path=tmp, unzip=True, quiet=True)
                missing = [f for f in files if not (Path(tmp) / f).exists()]
                if missing:   # copy all files or none, so the snapshot stays consistent
                    raise FileNotFoundError(f"{missing} not in the download")
                for filename, target in files.items():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(Path(tmp) / filename, target)
            status[name] = True
            print(f"  {dataset}: refreshed")
        except Exception as exc:
            print(f"  {dataset}: download failed ({exc}); keeping the snapshot.")
    return status
