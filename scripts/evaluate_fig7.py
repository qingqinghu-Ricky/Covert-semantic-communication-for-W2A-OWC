#!/usr/bin/env python3
"""Evaluate all three Fig. 7 schemes and three additional Willie positions."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# KDE workers must not each launch a full BLAS thread pool.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import numpy as np
    import torch
    from tgcn.data import loader
    from tgcn.metrics import SegmentationMetrics, kl_pair, position_kl_pair
    from tgcn.model import LognormalChannel, load_model
    from tgcn.runtime import ROOT, configure, write_csv, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fig7.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data/UAM")
    parser.add_argument(
        "--sparse-checkpoint", type=Path, default=ROOT / "checkpoints/sparse_ema.pth"
    )
    parser.add_argument(
        "--dense-checkpoint", type=Path, default=ROOT / "checkpoints/dense_ema.pth"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/fig7")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--kl-workers", type=int, default=8)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Smoke run; does not reproduce the full test set",
    )
    parser.add_argument(
        "--noise-powers",
        type=float,
        nargs="+",
        help="Override the seven configured points",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-kl", action="store_true", help="Evaluate Fig. 7(a,b) only"
    )
    parser.add_argument(
        "--legacy-drop-last",
        action="store_true",
        help="Use the source main.py's 1070-image protocol",
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.kl_workers < 1
        or (args.max_batches is not None and args.max_batches < 1)
    ):
        parser.error("Batch size, worker count and max batches must be positive")
    start = time.time()
    cfg = json.loads(args.config.read_text())
    device = configure(args.seed, args.device)
    models = [
        load_model(args.sparse_checkpoint, device),
        load_model(args.dense_checkpoint, device),
    ]
    test = loader(
        args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        drop_last=args.legacy_drop_last,
    )
    cache = []
    print("Caching semantic latents once for all noise powers...", flush=True)
    with torch.inference_mode():
        for index, batch in enumerate(test):
            if args.max_batches is not None and index >= args.max_batches:
                break
            image = batch["image"].to(device)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                latents = [model.latent(image).cpu() for model in models]
            cache.append((latents, batch["mask"]))
            if (index + 1) % 20 == 0:
                print(f"Cached {index + 1}/{len(test)} batches", flush=True)
    if not cache:
        raise ValueError("No test samples")
    names = ("sparse", "dense", "energy_matched")
    rows, position_rows, diagnostics = [], [], []
    powers = args.noise_powers or cfg["noise_power_dbm"]
    channel = LognormalChannel().to(device)
    position_channels = [
        LognormalChannel(p["mu"], p["sigma"]).to(device)
        for p in cfg["willie_positions"]
    ]
    pool = (
        None
        if args.skip_kl
        else ProcessPoolExecutor(args.kl_workers, mp_context=mp.get_context("spawn"))
    )
    try:
        for power in powers:
            snr_value = -power
            stats = {name: SegmentationMetrics() for name in names}
            kl = {name: [] for name in names}
            pos_kl = [[] for _ in position_channels]
            energy = {name: [] for name in names}
            active = {name: [] for name in names}
            matching_error = 0.0
            print(f"Evaluating noise power {power:g} dBm", flush=True)
            with torch.inference_mode():
                for batch_index, (cached_latents, targets) in enumerate(cache):
                    sparse_latent, dense_latent = [x.to(device) for x in cached_latents]
                    target = targets.to(device)
                    snr = torch.full((target.shape[0], 1), snr_value, device=device)
                    with torch.amp.autocast(
                        "cuda", enabled=args.amp and device.type == "cuda"
                    ):
                        sparse, _ = models[0].sparse(sparse_latent, snr)
                        dense = (
                            dense_latent
                            / dense_latent.square()
                            .mean(-1, keepdim=True)
                            .clamp_min(1e-8)
                            .sqrt()
                        )
                        target_energy = sparse.square().sum(-1, keepdim=True)
                        dense_energy = dense.square().sum(-1, keepdim=True)
                        matched = (
                            dense
                            * (
                                target_energy.clamp_min(1e-8)
                                / dense_energy.clamp_min(1e-8)
                            ).sqrt()
                        )
                    transmitted = [sparse.float(), dense.float(), matched.float()]
                    relative = (
                        transmitted[2].square().sum(-1)
                        - transmitted[0].square().sum(-1)
                    ).abs()
                    relative /= transmitted[0].square().sum(-1).clamp_min(1e-8)
                    matching_error = max(matching_error, relative.max().item())
                    jobs = []
                    for name, vector, model in zip(
                        names, transmitted, [models[0], models[1], models[1]]
                    ):
                        received = channel(vector, snr)
                        with torch.amp.autocast(
                            "cuda", enabled=args.amp and device.type == "cuda"
                        ):
                            prediction = model.segmentDecoder(received).argmax(dim=1)
                        stats[name].update(prediction, target)
                        energy[name].extend(vector.square().sum(-1).cpu().tolist())
                        active[name].extend((vector != 0).sum(-1).cpu().tolist())
                        if pool is not None:
                            zero = channel(torch.zeros_like(vector), snr)
                            # The two source experiments sampled independent Bob/Willie noise.
                            observed = channel(vector, snr)
                            jobs.append(
                                (
                                    kl[name],
                                    kl_pair,
                                    list(
                                        zip(zero.cpu().numpy(), observed.cpu().numpy())
                                    ),
                                )
                            )
                    if pool is not None:
                        for index, observer in enumerate(position_channels):
                            observed = observer(transmitted[0], snr)
                            zero = observer(torch.zeros_like(transmitted[0]), snr)
                            jobs.append(
                                (
                                    pos_kl[index],
                                    position_kl_pair,
                                    list(
                                        zip(zero.cpu().numpy(), observed.cpu().numpy())
                                    ),
                                )
                            )
                        futures = [
                            (out, [pool.submit(fn, pair) for pair in pairs])
                            for out, fn, pairs in jobs
                        ]
                        for out, tasks in futures:
                            out.extend(task.result() for task in tasks)
                    if (batch_index + 1) % 20 == 0:
                        print(f"  {batch_index + 1}/{len(cache)} batches", flush=True)
            if matching_error > 0.002:
                raise AssertionError(f"Energy matching failed: {matching_error}")
            for name in names:
                row = {
                    "scheme": name,
                    "noise_power_dbm": power,
                    "snr_db": snr_value,
                    "images": len(active[name]),
                    **stats[name].compute(),
                    "active_symbols": float(np.mean(active[name])),
                    "sparsity_ratio": 1 - float(np.mean(active[name])) / 2048,
                    "tx_energy_mean": float(np.mean(energy[name])),
                    "energy_match_max_relative_error": matching_error,
                }
                if kl[name]:
                    row.update(
                        kl_mean=float(np.mean([x["value"] for x in kl[name]])),
                        kl_std=float(np.std([x["value"] for x in kl[name]])),
                        kl_samples=len(kl[name]),
                    )
                rows.append(row)
                diagnostics.append(
                    {
                        "noise_power_dbm": power,
                        "scheme": name,
                        "integration_warnings": sum(x["warnings"] for x in kl[name]),
                    }
                )
            for position, values in zip(cfg["willie_positions"], pos_kl):
                if values:
                    position_rows.append(
                        {
                            "position": position["position"],
                            "noise_power_dbm": power,
                            "mu": position["mu"],
                            "sigma": position["sigma"],
                            "kl_mean": float(np.mean([x["value"] for x in values])),
                            "kl_std": float(np.std([x["value"] for x in values])),
                            "kl_samples": len(values),
                        }
                    )
            write_csv(args.output / "fig7_abc.csv", rows)
            if position_rows:
                write_csv(args.output / "fig7_d.csv", position_rows)
            print(
                f"  sparse mIoU={rows[-3]['miou']:.6f}, sparsity={rows[-3]['sparsity_ratio']:.6f}",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    metadata = {
        "seed": args.seed,
        "device": str(device),
        "amp": args.amp,
        "batch_size": args.batch_size,
        "legacy_drop_last": args.legacy_drop_last,
        "smoke_run": args.max_batches is not None,
        "skip_kl": args.skip_kl,
        "noise_power_dbm": powers,
        "channel_eps": 1e-12,
        "elapsed_seconds": time.time() - start,
        "torch_version": torch.__version__,
        "diagnostics": diagnostics,
    }
    write_json(args.output / "run.json", metadata)
    print(
        f"Saved evaluation to {args.output} ({metadata['elapsed_seconds']:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
