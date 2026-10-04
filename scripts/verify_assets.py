#!/usr/bin/env python3
"""Check all dataset pairs, labels and checkpoint SHA256 checksums."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/UAM")
    parser.add_argument("--checkpoints", type=Path, default=root / "checkpoints")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.data / "manifest.jsonl").read_text().splitlines()
    ]
    for row in rows:
        base = args.data / row["split"]
        image = base / "images" / (row["name"] + ".jpg")
        mask = base / "masks" / (row["name"] + ".png")
        if sha256(image) != row["image_sha256"] or sha256(mask) != row["mask_sha256"]:
            raise ValueError(f"Checksum mismatch: {row['split']}/{row['name']}")
        label = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
        if label is None or label.ndim != 2 or label.max() > 2:
            raise ValueError(f"Invalid index mask: {mask}")
        if hashlib.sha256(label.tobytes()).hexdigest() != row["labels_sha256"]:
            raise ValueError(f"Label checksum mismatch: {mask}")
    for split, count in [("train", 4298), ("test", 1075)]:
        names = json.loads((args.data / "splits" / (split + ".json")).read_text())
        if len(names) != count or len(set(names)) != count:
            raise ValueError(f"Unexpected {split} split")
        if set(names) != {r["name"] for r in rows if r["split"] == split}:
            raise ValueError(f"Manifest/split mismatch: {split}")
    for item in json.loads((args.checkpoints / "manifest.json").read_text()):
        if sha256(args.checkpoints / item["file"]) != item["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {item['file']}")
    print(f"Verified {len(rows)} dataset pairs and all five checkpoint files.")


if __name__ == "__main__":
    main()
