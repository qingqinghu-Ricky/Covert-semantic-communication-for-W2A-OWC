"""Foreground IoU and the source KDE-based D(P0 || P1) estimator."""

import warnings

import numpy as np
from scipy import integrate
from scipy.stats import gaussian_kde


class SegmentationMetrics:
    def __init__(self):
        self.intersection = np.zeros(3, dtype=np.float64)
        self.union = np.zeros(3, dtype=np.float64)
        self.correct = self.pixels = 0

    def update(self, prediction, target):
        self.correct += (prediction == target).sum().item()
        self.pixels += target.numel()
        for c in range(3):
            p, t = prediction == c, target == c
            self.intersection[c] += (p & t).sum().item()
            self.union[c] += (p | t).sum().item()

    def compute(self):
        valid = self.union > 0
        iou = np.divide(
            self.intersection, self.union + 1e-6, out=np.full(3, -1.0), where=valid
        )
        foreground = iou[1:]
        mean = (foreground * (foreground != -1)).sum() / (
            (foreground != -1).sum() + 1e-6
        )
        return {
            "human_iou": float(iou[1]),
            "vehicle_iou": float(iou[2]),
            "miou": float(mean),
            "pixel_accuracy": self.correct / max(self.pixels, 1),
        }


def kl_pair(pair, position_mode=False):
    zero, signal = pair
    zero, signal = np.asarray(zero, dtype=np.float64), np.asarray(
        signal, dtype=np.float64
    )
    p, q = gaussian_kde(zero), gaussian_kde(signal)

    def integrand(x):
        px, qx = max(float(p(x)[0]), 1e-12), max(float(q(x)[0]), 1e-12)
        return px * np.log(px / qx)

    low, high = (
        min(zero.min(), signal.min()) * 0.95,
        max(zero.max(), signal.max()) * 1.05,
    )
    # Position evaluation used limit=100; the energy-matched evaluator used 50.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", integrate.IntegrationWarning)
        value, error = integrate.quad(
            integrand, low, high, limit=100 if position_mode else 50
        )
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite KL estimate")
    return {
        "value": float(max(value, 0.0) if position_mode else value),
        "error": float(error),
        "warnings": len(caught),
    }


def position_kl_pair(pair):
    return kl_pair(pair, position_mode=True)
