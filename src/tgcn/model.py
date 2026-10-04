"""Semantic transmitter/receiver and the source evaluation channel."""

import torch
from torch import nn

from .architecture import (
    ResNet50FPN,
    Semantic_Encoder,
    Semantic_Decoder,
    SparseMaskGenerator,
)

BOB_MU = -13.1548
BOB_SIGMA = 0.330682


def normalize_active(x, eps=1e-8):
    count = (x != 0).sum(dim=-1, keepdim=True).float()
    energy = x.square().sum(dim=-1, keepdim=True) / (count + eps)
    return x / energy.clamp_min(eps).sqrt()


class SemanticModel(nn.Module):
    """Module names are identical to the original ToyNet state dictionary."""

    def __init__(self, pretrained_encoder=False):
        super().__init__()
        self.segmentNet = ResNet50FPN(3)
        self.segmentEncoder = Semantic_Encoder(
            3, 2048, 224, 224, pretrained=pretrained_encoder
        )
        enc = self.segmentEncoder
        self.segmentDecoder = Semantic_Decoder(
            3, 2048, enc.final_c, enc.final_h, enc.final_w, 224, 224
        )
        self.sparse_mask_gen = SparseMaskGenerator(
            2048, img_feat_dim=256, snr_feat_dim=128
        )

    def latent(self, images):
        with torch.no_grad():
            semantics = self.segmentNet(images)
        return self.segmentEncoder(semantics)

    def sparse(self, latent, snr_db, training=False):
        p = self.sparse_mask_gen(latent.detach(), snr_db.float())
        if training:
            mask = (torch.rand_like(p) < p).int() + p - p.detach()
        else:
            mask = (p >= 0.5).to(latent.dtype)
        return normalize_active(latent * mask), mask

    def forward(self, images, snr_db, channel, training=False):
        transmitted, mask = self.sparse(self.latent(images), snr_db, training)
        return self.segmentDecoder(channel(transmitted, snr_db)), transmitted, mask


class LognormalChannel(nn.Module):
    """Shared Bob normalization, as used by the original Willie-position evaluator.

    eps=1e-12 is significant relative to E[h_b^2] and is retained deliberately.
    Training used eps=0. Every latent block has one fading realization.
    """

    def __init__(self, mu=BOB_MU, sigma=BOB_SIGMA, eps=1e-12):
        super().__init__()
        self.mu, self.sigma, self.eps = mu, sigma, eps

    def forward(self, x, snr_db):
        x = x.float()
        mu, sigma, bob_mu, bob_sigma = (
            torch.tensor(v, device=x.device)
            for v in (self.mu, self.sigma, BOB_MU, BOB_SIGMA)
        )
        fading = torch.distributions.LogNormal(mu, sigma).sample((x.shape[0], 1))
        reference = torch.exp(2 * bob_mu + 2 * bob_sigma.square())
        fading = fading / torch.sqrt(reference + self.eps)
        noise_sigma = torch.sqrt(1 / (10 ** (snr_db.float() / 10)))
        return x * fading + torch.randn_like(x) * noise_sigma


def load_model(path, device):
    model = SemanticModel()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def update_ema(ema, model, decay=0.999):
    # Retain the source state_dict traversal, including shared backbone aliases.
    source = model.state_dict()
    for name, tensor in ema.state_dict().items():
        if tensor.is_floating_point():
            tensor.mul_(decay).add_(source[name], alpha=1 - decay)
        else:
            tensor.copy_(source[name])
