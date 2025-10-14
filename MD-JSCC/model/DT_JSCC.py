import torch
import torch.nn as nn
from .modules import Resblock, MaskAttentionSampler
import copy
# ====================================================================== #
# Multi‑decoder JSCC CIFAR‑10                                            #
# ====================================================================== #
class DTJSCC_CIFAR10(nn.Module):
    """
    One shared Encoder + Sampler; one independent decoder head per task_key.
    """
    def __init__(self, in_channels, latent_channels,
                 out_classes, task_keys, num_embeddings=400):
        super().__init__()
        self.latent_d = latent_channels

        # ---------------- shared encoder ----------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels // 8), nn.ReLU(),

            nn.Conv2d(latent_channels // 8, latent_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels // 4), nn.ReLU(),
            nn.MaxPool2d(2),

            Resblock(latent_channels // 4),

            nn.Conv2d(latent_channels // 4, latent_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels // 2), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(latent_channels // 2, latent_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels), nn.ReLU(),
            nn.MaxPool2d(2),

            Resblock(latent_channels),
        )

        # ---------------- sampler -----------------------------------------
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)

        # ---------------- task‑specific decoders --------------------------
        proto = nn.Sequential(
            Resblock(latent_channels), Resblock(latent_channels),
            nn.BatchNorm2d(latent_channels), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(latent_channels, out_classes),
        )
        self.decoders = nn.ModuleDict({k: copy.deepcopy(proto) for k in task_keys})

    # --------------------------------------------------------------------
    def _encode(self, x):
        z = self.encoder(x)                       # (B,C,H,W)
        shp = z.shape
        z = z.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return z, shp

    def _decode(self, feats, shp, task_key):
        b, c, h, w = shp
        feats = feats.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return self.decoders[task_key](feats)

    # --------------------------------------------------------------------
    def forward(self, x, task_key: str, mod=None):
        z, shp = self._encode(x)
        z, dist = self.sampler(z, mod=mod)
        logits = self._decode(z, shp, task_key)
        return logits, dist


# ====================================================================== #
# Multi-decoder JSCC CINIC-10                                            #
# ====================================================================== #
class DTJSCC_CINIC10(nn.Module):
    """
    One shared Encoder + Sampler; one independent decoder head per task_key.
    Architecture mirrors DTJSCC_CIFAR10; default classes = 10.
    """
    def __init__(self, in_channels, latent_channels,
                 out_classes, task_keys, num_embeddings=400):
        super().__init__()
        assert latent_channels % 8 == 0 and latent_channels >= 64, \
            f"latent_channels must be >=64 and divisible by 8, got {latent_channels}"
        assert in_channels in (1, 3), f"in_channels must be 1 or 3, got {in_channels}"
        self.latent_d = latent_channels

        # ---------------- shared encoder ----------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels // 8), nn.ReLU(),

            nn.Conv2d(latent_channels // 8, latent_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels // 4), nn.ReLU(),
            nn.MaxPool2d(2),

            Resblock(latent_channels // 4),

            nn.Conv2d(latent_channels // 4, latent_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels // 2), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(latent_channels // 2, latent_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels), nn.ReLU(),
            nn.MaxPool2d(2),

            Resblock(latent_channels),
        )

        # ---------------- sampler -----------------------------------------
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)

        # ---------------- task-specific decoders --------------------------
        proto = nn.Sequential(
            Resblock(latent_channels), Resblock(latent_channels),
            nn.BatchNorm2d(latent_channels), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(latent_channels, out_classes),
        )
        self.decoders = nn.ModuleDict({k: copy.deepcopy(proto) for k in task_keys})

    def _encode(self, x):
        z = self.encoder(x)
        shp = z.shape
        z = z.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return z, shp

    def _decode(self, feats, shp, task_key):
        b, c, h, w = shp
        feats = feats.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return self.decoders[task_key](feats)

    def forward(self, x, task_key: str, mod=None):
        z, shp = self._encode(x)
        z, dist = self.sampler(z, mod=mod)
        logits = self._decode(z, shp, task_key)
        return logits, dist


# ====================================================================== #
# Multi-decoder JSCC CIFAR-100                                           #
# ====================================================================== #
class DTJSCC_CIFAR100(nn.Module):
    """
    One shared Encoder + Sampler; one independent decoder head per task_key.
    Architecture mirrors DTJSCC_CIFAR10; default classes = 100.
    """
    def __init__(self, in_channels, latent_channels,
                 out_classes, task_keys, num_embeddings=400):
        super().__init__()
        assert latent_channels % 8 == 0 and latent_channels >= 64, \
            f"latent_channels must be >=64 and divisible by 8, got {latent_channels}"
        assert in_channels in (1, 3), f"in_channels must be 1 or 3, got {in_channels}"
        self.latent_d = latent_channels

        # ---------------- shared encoder ----------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels // 8), nn.ReLU(),

            nn.Conv2d(latent_channels // 8, latent_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels // 4), nn.ReLU(),
            nn.MaxPool2d(2),

            Resblock(latent_channels // 4),

            nn.Conv2d(latent_channels // 4, latent_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels // 2), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(latent_channels // 2, latent_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels), nn.ReLU(),
            nn.MaxPool2d(2),

            Resblock(latent_channels),
        )

        # ---------------- sampler -----------------------------------------
        self.sampler = MaskAttentionSampler(latent_channels, num_embeddings)

        # ---------------- task-specific decoders --------------------------
        proto = nn.Sequential(
            Resblock(latent_channels), Resblock(latent_channels),
            nn.BatchNorm2d(latent_channels), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(latent_channels, out_classes),
        )
        self.decoders = nn.ModuleDict({k: copy.deepcopy(proto) for k in task_keys})

    def _encode(self, x):
        z = self.encoder(x)
        shp = z.shape
        z = z.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_d)
        return z, shp

    def _decode(self, feats, shp, task_key):
        b, c, h, w = shp
        feats = feats.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return self.decoders[task_key](feats)

    def forward(self, x, task_key: str, mod=None):
        z, shp = self._encode(x)
        z, dist = self.sampler(z, mod=mod)
        logits = self._decode(z, shp, task_key)
        return logits, dist
