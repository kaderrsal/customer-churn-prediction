#!/usr/bin/env python3
"""
Download a Kaggle dataset using the Kaggle CLI. Configure your Kaggle API token
as described in https://www.kaggle.com/docs/api.

Usage:
  python scripts/fetch_kaggle.py --slug "<owner/dataset-slug>" --out data/raw

Example:
  python scripts/fetch_kaggle.py --slug "new-york-city/nyc-airbnb-open-data" --out data/raw
"""
import os
import argparse
import shutil
import subprocess


def check_kaggle_config():
    home = os.path.expanduser("~")
    kaggle_path = os.path.join(home, ".kaggle", "kaggle.json")
    return os.path.exists(kaggle_path)


def download_dataset(slug: str, out: str):
    os.makedirs(out, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", out, "--unzip"]
    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        print("Download finished.")
    except subprocess.CalledProcessError as e:
        print("kaggle CLI returned non-zero exit:", e)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, help="Kaggle dataset slug owner/dataset-name")
    parser.add_argument("--out", default="data/raw", help="Output folder")
    args = parser.parse_args()

    if not check_kaggle_config():
        print("Kaggle API token not found. See https://www.kaggle.com/docs/api to set up ~/.kaggle/kaggle.json")
        return

    download_dataset(args.slug, args.out)


if __name__ == '__main__':
    main()
