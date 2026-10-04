#!/usr/bin/env python3
"""Retrain the source sparse or lambda=0 model from the supplied initialization."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    from tgcn.data import UAMDataset, loader
    from tgcn.metrics import SegmentationMetrics
    from tgcn.model import SemanticModel, LognormalChannel, update_ema
    from tgcn.runtime import ROOT, configure, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fig7.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data/UAM")
    parser.add_argument("--initialization", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output", type=Path, help="Defaults to training/<scheme>")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N training batches; 0 disables batch messages",
    )
    parser.add_argument("--max-train-batches", type=int, help="Smoke training only")
    parser.add_argument("--max-eval-batches", type=int, help="Smoke validation only")
    parser.add_argument(
        "--regularizer", choices=["source", "reciprocal_noise"], default="source"
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Generate a new stratified holdout from train; 0 preserves source test monitoring",
    )
    parser.add_argument(
        "--resume", type=Path, help="Resume a training/last.pth from this package"
    )
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())["training"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    if epochs < 1 or args.batch_size < 2 or not 0 <= args.validation_fraction < 1:
        parser.error(
            "Epochs must be positive; batch size >=2 (BatchNorm); validation fraction in [0,1)"
        )
    if any(
        v is not None and v < 1 for v in [args.max_train_batches, args.max_eval_batches]
    ):
        parser.error("Smoke limits must be positive")
    args.output = args.output or ROOT / "training" / args.scheme
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pth").exists() and args.resume is None:
        raise FileExistsError(
            "Training output already exists; use --resume or a new --output"
        )
    device = configure(args.seed, args.device)
    encoder_weights = str(args.initialization / "resnet18_imagenet.pth")
    model = SemanticModel(pretrained_encoder=encoder_weights).to(device)
    ema = SemanticModel(pretrained_encoder=encoder_weights).to(device)
    ema.load_state_dict(model.state_dict())
    for net, filename in [
        (model, "segmenter_init.pth"),
        (ema, "segmenter_init_ema.pth"),
    ]:
        state = torch.load(
            args.initialization / filename, map_location="cpu", weights_only=True
        )
        net.segmentNet.load_state_dict(state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    training = loader(
        args.data,
        "train",
        args.batch_size,
        training=True,
        workers=args.workers,
        drop_last=True,
    )
    validation = loader(
        args.data, "test", args.batch_size, workers=args.workers, drop_last=True
    )
    split_info = {
        "policy": "source_test_monitoring",
        "train": len(training.dataset),
        "validation_source": "test",
        "evaluation_drop_last": True,
    }
    if args.validation_fraction:
        # This is a newly generated holdout, not a recovered paper split.
        import cv2
        from collections import defaultdict

        full = UAMDataset(args.data, "train", training=True)
        groups = defaultdict(list)
        for index, name in enumerate(full.names):
            mask = cv2.imread(
                str(args.data / "train/masks" / f"{name}.png"), cv2.IMREAD_UNCHANGED
            )
            groups[tuple(int(c) for c in np.unique(mask) if c != 0)].append(index)
        rng = np.random.default_rng(args.seed)
        allocations = {
            k: int(len(v) * args.validation_fraction) for k, v in groups.items()
        }
        remaining = round(len(full) * args.validation_fraction) - sum(
            allocations.values()
        )
        ranked = sorted(
            groups,
            key=lambda k: len(groups[k]) * args.validation_fraction - allocations[k],
            reverse=True,
        )
        for k in ranked[:remaining]:
            allocations[k] += 1
        valid_indices = []
        for k, v in groups.items():
            rng.shuffle(v)
            valid_indices.extend(v[: allocations[k]])
        valid_set = set(valid_indices)
        train_indices = [i for i in range(len(full)) if i not in valid_set]
        training = DataLoader(
            Subset(full, train_indices),
            args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        validation = DataLoader(
            Subset(UAMDataset(args.data, "train"), sorted(valid_set)),
            args.batch_size,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        split_info = {
            "policy": "new_stratified_holdout",
            "train": len(train_indices),
            "validation": len(valid_indices),
            "train_names": [full.names[i] for i in train_indices],
            "validation_names": [full.names[i] for i in sorted(valid_set)],
        }
    write_json(args.output / "split.json", split_info)
    recipe = {
        "scheme": args.scheme,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "regularizer": args.regularizer,
        "validation_fraction": args.validation_fraction,
        "smoke_run": args.max_train_batches is not None
        or args.max_eval_batches is not None,
        "training_channel_eps": 0.0,
        **cfg,
    }
    recipe["epochs"] = epochs
    recipe["sparse_regularizer"] = (
        "snr_db" if args.regularizer == "source" else "reciprocal_noise"
    )
    recipe["validation_split"] = split_info["policy"]
    write_json(args.output / "recipe.json", recipe)
    start_epoch = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if (
            state["recipe"]["scheme"] != args.scheme
            or state["recipe"]["epochs"] != epochs
        ):
            raise ValueError("Resume requires the same scheme and total epoch count")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        random.setstate(state["python_rng"])
        # A process can stop after writing history but before committing last.pth.
        # Discard only uncommitted epochs so resumed reports have one row per epoch.
        history_path = args.output / "history.jsonl"
        if history_path.exists():
            committed = [
                json.loads(line) for line in history_path.read_text().splitlines()
            ]
            committed = [row for row in committed if row["epoch"] <= start_epoch]
            history_path.write_text(
                "".join(json.dumps(row) + "\n" for row in committed)
            )
    channel = LognormalChannel(eps=0.0).to(device)
    weight = cfg["lambda_sparse"] if args.scheme == "sparse" else cfg["lambda_dense"]
    for epoch in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        model.train()
        ema.train()
        model.segmentNet.eval()
        ema.segmentNet.eval()
        losses = []
        for index, batch in enumerate(training):
            if args.max_train_batches is not None and index >= args.max_train_batches:
                break
            snr_value = random.uniform(*cfg["snr_db_range"])
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            snr = torch.full((images.size(0), 1), snr_value, device=device)
            logits, _, mask = model(images, snr, channel, training=True)
            multiplier = snr if args.regularizer == "source" else 10 ** (snr / 10)
            sparse_loss = (multiplier * mask.sum(1, keepdim=True)).mean() / (224 * 224)
            loss = F.cross_entropy(logits, masks) + weight * sparse_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema(ema, model, cfg["ema_decay"])
            losses.append(loss.item())
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "stage": "train",
                            "scheme": args.scheme,
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "batch": index + 1,
                            "batches": len(training),
                            "mean_loss": float(np.mean(losses)),
                            "elapsed_seconds": time.monotonic() - epoch_started,
                        }
                    ),
                    flush=True,
                )
        if not losses:
            raise ValueError("Training loader produced no batches")
        scheduler.step()
        ema.eval()
        stats = SegmentationMetrics()
        validation_images = 0
        validation_active_symbols = 0
        with torch.inference_mode():
            for index, batch in enumerate(validation):
                if args.max_eval_batches is not None and index >= args.max_eval_batches:
                    break
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                snr = torch.full((images.size(0), 1), snr_value, device=device)
                logits, transmitted, _ = ema(images, snr, channel)
                stats.update(logits.argmax(1), masks)
                validation_images += images.size(0)
                validation_active_symbols += (transmitted != 0).sum().item()
        summary = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_snr_db": snr_value,
            **stats.compute(),
            "validation_images": validation_images,
            "validation_active_symbols": validation_active_symbols / validation_images,
            "validation_sparsity_ratio": 1
            - validation_active_symbols / (validation_images * 2048),
            "train_batches": len(losses),
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        with (args.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        # Source best_acc.pth was saved at the final epoch, not selected by mIoU.
        torch.save(ema.state_dict(), args.output / "ema.pth.tmp")
        (args.output / "ema.pth.tmp").replace(args.output / "ema.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "recipe": recipe,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else [],
                "python_rng": random.getstate(),
            },
            args.output / "last.pth.tmp",
        )
        (args.output / "last.pth.tmp").replace(args.output / "last.pth")
        print(json.dumps(summary), flush=True)
    print(f"Saved final EMA model to {args.output / 'ema.pth'}", flush=True)


if __name__ == "__main__":
    main()
