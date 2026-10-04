#!/usr/bin/env python3
"""Train the sparse and dense models from the provided initialization weights."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "training/from_initialization")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume existing last.pth checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print training commands without running them")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("Workers must be non-negative")
    output = args.output.resolve()
    for scheme in ("sparse", "dense"):
        folder = output / scheme
        command = [
            sys.executable, "-u", str(ROOT / "scripts/train.py"),
            "--scheme", scheme,
            "--epochs", "200",
            "--batch-size", "10",
            "--seed", "1",
            "--workers", str(args.workers),
            "--progress-every", "50",
            "--device", args.device,
            "--output", str(folder),
        ]
        checkpoint = folder / "last.pth"
        if args.resume and checkpoint.is_file():
            command.extend(["--resume", str(checkpoint)])
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
