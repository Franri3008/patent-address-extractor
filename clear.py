#!/usr/bin/env python3
"""Delete output folder contents, preserving cached BigQuery files by default."""

import argparse
import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def clear_output(force: bool = False) -> None:
    if not OUTPUT_DIR.exists():
        print("Nothing to clear — output/ does not exist.")
        return

    for item in OUTPUT_DIR.iterdir():
        if item.name == ".gitkeep":
            continue

        if not force and item.is_file() and item.name.startswith("raw_") and item.suffix == ".csv":
            print(f"  kept   {item.name}  (cached BigQuery file)")
            continue

        if item.is_dir():
            shutil.rmtree(item)
            print(f"  deleted {item.name}/")
        else:
            item.unlink()
            print(f"  deleted {item.name}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear the output folder.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Also delete cached BigQuery files (raw_*.csv).",
    )
    args = parser.parse_args()
    clear_output(force=args.force)
