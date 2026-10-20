"""Downloads the raw public data into `data/raw/` (gitignored).

- ChileCompra open data: one ZIP per month with every purchase-order line sent through
  Mercado Público, from `https://transparenciachc.blob.core.windows.net/oc-da/<year>-<month>.zip`
  (no key needed, about 110 MB per month in 2017).
- Kaggle "Quality Prediction in a Mining Process" (`edumagalhaes/quality-prediction-in-a-mining-process`),
  through the Kaggle API (needs `~/.kaggle/kaggle.json`).

    python -m src.sources
"""
from __future__ import annotations

import os
import shutil
import urllib.request

from src.setup_data import FLOTATION_CSV, MONTHS, RAW_DIR

OC_URL = "https://transparenciachc.blob.core.windows.net/oc-da/{year}-{month}.zip"
KAGGLE_DATASET = "edumagalhaes/quality-prediction-in-a-mining-process"
USER_AGENT = "chile-mining-ops-agent (github.com/Rxyxs)"


def _download(url: str, path: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = path + ".part"
    with urllib.request.urlopen(request, timeout=300) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    with open(tmp, "rb") as check:
        if check.read(2) != b"PK":
            os.remove(tmp)
            raise RuntimeError(f"{url} did not return a ZIP")
    os.replace(tmp, path)


def download_all(raw_dir: str = RAW_DIR) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    for year, month in MONTHS:
        path = os.path.join(raw_dir, f"oc_{year}-{month}.zip")
        if not os.path.exists(path):
            _download(OC_URL.format(year=year, month=month), path)
            print(f"downloaded {os.path.basename(path)}")
    if not os.path.exists(os.path.join(raw_dir, os.path.basename(FLOTATION_CSV))):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=raw_dir, unzip=True)
        print("downloaded the flotation plant dataset")


if __name__ == "__main__":
    download_all()
