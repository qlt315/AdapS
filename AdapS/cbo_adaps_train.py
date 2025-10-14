# cbo_adaps.py
# Contextual-MAB with pilot features for task-level batch-size selection (AdapS / JSCC)
# - One decision per task: choose a fixed batch size for the whole task
# - Pilot (optional): measure domain-agnostic continuous "fingerprints" as context
# - In-task: shift detection + TTNorm (deferred-apply), latency with per-timeslot fading
# - Per-task metric: average per-image waiting; reward = mean_acc/100 - kappa * avg_violation_per_img

from __future__ import annotations

import argparse
import math
import random
import time
import csv
import os
from typing import Dict, Tuple, List, Optional
import cloudpickle as cp
import numpy as np
import torch
import torch.nn as nn

# === Project imports ===
from JSCC.datasets.dataloader import get_data
from JSCC.utils.modulation import QAM, PSK
from JSCC.utils.accuracy import accuracy
import model.DT_JSCC as JSCC_model

# === Contextual MAB library ===
from cbo import ContextualBayesianOptimization
from cbo_util import UtilityFunction
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def sample_random_batch(loader, bs: int):
    """
    Sample a random batch of size `bs` directly from loader.dataset.
    Assumes dataset returns (tensor_image [C,H,W] in [0,1], int_label).
    """
    assert bs is not None and bs > 0, "batch size must be positive"
    ds = loader.dataset
    n = len(ds)
    idxs = np.random.choice(n, size=bs, replace=(bs > n))
    imgs, labels = [], []
    for i in idxs:
        x, y = ds[int(i)]
        imgs.append(x.unsqueeze(0))
        labels.append(torch.tensor(y, dtype=torch.long))
    imgs = torch.cat(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    return imgs, labels


# ------------------------------ Helpers ----------------------------------

_DATASET_BASE_RE = __import__('re').compile(
    r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=__import__('re').IGNORECASE
)

def _dataset_base(dataset_name: str) -> str:
    m = _DATASET_BASE_RE.match(str(dataset_name).strip())
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).lower()

def _dataset_base_upper(dataset_name: str) -> str:
    return _dataset_base(dataset_name).upper()

def _infer_num_classes(dataset_name: str) -> int:
    base = _dataset_base(dataset_name)
    if base == "cifar100": return 100
    if base in ("cifar10", "cinic10"): return 10
    raise ValueError(f"Unsupported dataset base: {base}")

def _build_model(dataset: str, in_channels: int, latent_d: int,
                 num_classes: int, num_embeddings: int) -> nn.Module:
    base = _dataset_base(dataset)
    if base == "cifar100":
        model = JSCC_model.DTJSCC_CIFAR100(in_channels, latent_d, num_classes,
                                            num_embeddings=num_embeddings)
    elif base == "cifar10":
        model = JSCC_model.DTJSCC_CIFAR10(in_channels, latent_d, num_classes,
                                           num_embeddings=num_embeddings)
    elif base == "cinic10":
        model = JSCC_model.DTJSCC_CINIC10(in_channels, latent_d, num_classes,
                                           num_embeddings=num_embeddings)
    else:
        raise ValueError(f"No model for dataset: {dataset}")
    return model

def _build_modulator(mod: str, num_embeddings: int, snr_db: float,
                     channel: str, ch_args: Dict) -> object:
    return (QAM if mod.lower()=="qam" else PSK)(
        num_embeddings, snr_db, channel, ch_args
    )

# ---------- Domain name helpers (dynamic by dataset_base) ----------

def _candidate_domain_names_for_base(dataset_base: str) -> List[str]:
    """
    Generate candidate dataset names for this base. We will try to load them and keep the ones that exist.
    If your project doesn't have augmented variants for a base, we'll gracefully fall back to [BASE].
    """
    base_up = _dataset_base_upper(dataset_base)   # e.g., 'CIFAR100'
    # Common suffixes used in your CIFAR10 setup; we'll try and skip missing ones.
    suffixes = ["", "_noise", "_fog", "_bright", "_snow", "_rain", "_blur"]
    return [base_up + s for s in suffixes]

def _build_domains_from_loader_keys(loader_keys: List[str], dataset_base: str) -> List[Tuple[str,str]]:
    """
    Build (dataset_name, channel) tuples from the actually loaded datasets.
    - clean domain (exact base) -> AWGN only
    - augmented domains (prefix base_) -> AWGN/Rician/Rayleigh
    Fallback: if only one dataset exists, use AWGN-only on it.
    """
    keys_up = [k.upper() for k in loader_keys]
    base_up = _dataset_base_upper(dataset_base)

    names = [k for k in loader_keys if k.upper().startswith(base_up)]
    doms: List[Tuple[str,str]] = []
    if base_up in [n.upper() for n in names]:
        # clean domain
        clean_name = [n for n in names if n.upper() == base_up][0]
        doms.append((clean_name, 'awgn'))

    for n in names:
        if n.upper() == base_up:
            continue
        doms += [(n, 'awgn'), (n, 'rician'), (n, 'rayleigh')]

    if not doms:
        # Fallback to "whatever exists": choose the first key and set AWGN
        if loader_keys:
            doms = [(loader_keys[0], 'awgn')]
    return doms


# ---------- metrics used by detection/pilot ----------

def _w2_gaussian(mean1: np.ndarray, std1: np.ndarray,
                 mean2: np.ndarray, std2: np.ndarray) -> float:
    """Channel-wise 2-Wasserstein distance between diagonal Gaussians (mean,std per channel)."""
    m1 = torch.as_tensor(mean1, dtype=torch.float32)
    s1 = torch.as_tensor(std1,  dtype=torch.float32)
    m2 = torch.as_tensor(mean2, dtype=torch.float32)
    s2 = torch.as_tensor(std2,  dtype=torch.float32)
    s1 = torch.clamp(s1, min=1e-8)
    s2 = torch.clamp(s2, min=1e-8)
    w2 = torch.sqrt((m1 - m2)**2 + (s1 - s2)**2).mean()
    return float(w2.item())

def _sym_kl_diag_gauss(mean1, std1, mean2, std2, eps=1e-6) -> float:
    """Symmetric KL between diagonal Gaussians, averaged over channels."""
    mu1  = torch.as_tensor(mean1, dtype=torch.float32)
    mu2  = torch.as_tensor(mean2, dtype=torch.float32)
    var1 = torch.as_tensor(std1,  dtype=torch.float32)**2 + eps
    var2 = torch.as_tensor(std2,  dtype=torch.float32)**2 + eps
    d1 = (var1 + (mu1 - mu2)**2) / (2.0 * var2) - 0.5
    d2 = (var2 + (mu1 - mu2)**2) / (2.0 * var1) - 0.5
    return float((d1 + d2).mean().item())


# -------------------------- TTNorm with deferred-apply --------------------------

class TTNorm(nn.Module):
    """
    Deferred-apply Test-Time Normalization:
      - Persistent (mu_t, var_t) initialized from source BN.
      - Modes:
          * "infer_old": use (mu_t, var_t) only; do not update.
          * "adapt":     fuse (mu_b,var_b) with (mu_t,var_t) via λ=B/(B+m), write back; output uses old stats.
          * "normal":    (unused here) fuse-and-use for current batch, then write back.
    """
    def __init__(self, src_bn: nn.BatchNorm2d, m: float = 10.0):
        super().__init__()
        self.gamma = src_bn.weight
        self.beta  = src_bn.bias
        self.register_buffer("mu_t",  src_bn.running_mean.clone())
        self.register_buffer("var_t", src_bn.running_var.clone())
        self.eps = src_bn.eps
        self.m   = float(m)
        self.mode: str = "infer_old"

    def set_mode(self, mode: str):
        assert mode in ("infer_old", "adapt", "normal")
        self.mode = mode

    def forward(self, x: torch.Tensor):
        if self.mode == "infer_old":
            return nn.functional.batch_norm(
                x, self.mu_t, self.var_t, self.gamma, self.beta,
                training=False, momentum=0.0, eps=self.eps
            )

        # compute current-batch stats
        mu_b  = x.mean(dim=[0, 2, 3])
        var_b = x.var (dim=[0, 2, 3], unbiased=False)
        B = max(1.0, float(x.size(0)))
        lam = B / (B + self.m)
        mu_f  = (1.0 - lam) * self.mu_t + lam * mu_b
        var_f = (1.0 - lam) * self.var_t + lam * var_b

        if self.mode == "adapt":
            with torch.no_grad():
                self.mu_t.copy_(mu_f)
                self.var_t.copy_(var_f)
            return nn.functional.batch_norm(
                x, self.mu_t, self.var_t, self.gamma, self.beta,
                training=False, momentum=0.0, eps=self.eps
            )

        # "normal" (unused)
        y = nn.functional.batch_norm(
            x, mu_f, var_f, self.gamma, self.beta,
            training=False, momentum=0.0, eps=self.eps
        )
        with torch.no_grad():
            self.mu_t.copy_(mu_f)
            self.var_t.copy_(var_f)
        return y


def _get_parent_module(root: nn.Module, dotted_name: str) -> nn.Module:
    if '.' not in dotted_name:
        return root
    parent_path = dotted_name.split('.')[:-1]
    parent = root
    for p in parent_path:
        parent = getattr(parent, p)
    return parent

def replace_bn_with_ttn(model: nn.Module, m: float = 10.0):
    """Replace all nn.BatchNorm2d by TTNorm(m); default mode='infer_old'."""
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.BatchNorm2d):
            parent = _get_parent_module(model, name)
            child_name = name.split('.')[-1]
            ttn = TTNorm(mod, m=m).to(mod.weight.device)
            setattr(parent, child_name, ttn)

def set_ttn_mode(model: nn.Module, mode: str):
    for m in model.modules():
        if isinstance(m, TTNorm):
            m.set_mode(mode)


# ------------------------------ Latency model (per-timeslot fading) ------------------------------

def obtain_tx_latency_seconds_timeslot_fading(
    latent_feat: torch.Tensor,
    num_embeddings: int,
    snr_db: float,
    t_slot_s: float,
    bandwidth_hz: float,
    channel_type: str,
    ch_args: Dict,
) -> float:
    """
    Wireless latency estimate with per-timeslot fading (batch-level):
    - Payload bits L_b = (#symbols per img) * bs * log2(U)
    - Each slot i has |h|^2_i drawn i.i.d. by channel_type.
      AWGN: |h|^2 = 1; Rayleigh: Exp(1); Rician(K): normalized (E|h|^2=1).
    - Capacity per slot: C_i = bandwidth * log2(1 + SNR0 * |h|^2_i).
    - Keep drawing slots until accumulated bits >= L_b; return slots * t_slot_s (seconds).
    """
    bs = latent_feat.size(0)
    z = latent_feat.view(bs, -1)
    symbols_per_img = z.size(1)
    payload_bits = symbols_per_img * bs * math.log2(num_embeddings)

    snr0 = 10 ** (snr_db / 10.0)  # linear

    def draw_h2(n: int) -> np.ndarray:
        ct = channel_type.lower()
        if ct == 'awgn':
            return np.ones(n, dtype=np.float64)
        if ct == 'rayleigh':
            return np.random.exponential(scale=1.0, size=n)  # unit-mean
        if ct == 'rician':
            K = float(ch_args.get('K', 3.0))
            mu = math.sqrt(K / (K + 1.0))
            sigma = math.sqrt(1.0 / (2.0 * (K + 1.0)))  # per real component
            re = np.random.normal(loc=mu, scale=sigma, size=n)
            im = np.random.normal(loc=0.0, scale=sigma, size=n)
            return re*re + im*im
        return np.ones(n, dtype=np.float64)

    bits_acc = 0.0
    slots = 0
    max_slots = int(1e9)
    while bits_acc < payload_bits and slots < max_slots:
        h2 = draw_h2(1)[0]
        cap = bandwidth_hz * math.log2(1.0 + snr0 * h2)  # bits/sec
        if cap <= 0:
            cap = 1e3  # 1 kbps floor
        bits_acc += cap * t_slot_s
        slots += 1
    return slots * t_slot_s


# ------------------------------ Pilot (domain-agnostic fingerprints) ------------------------------

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

@torch.no_grad()
def compute_pilot_context(
    model: nn.Module,
    stem_probe: nn.Module,
    loaders: Dict[str, object],
    domain: Tuple[str, str],
    mod_kind: str,
    num_embeddings: int,
    snr_db: float,
    device: torch.device,
    num_classes: int,
    pilot_bs: int = 32,
    pilot_repeats: int = 2,
    img_ref_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Collect cheap, label-free fingerprints on the chosen domain before making the bs decision.
    Returns raw (unnormalized) features:
        - img_kl_ref: Symmetric KL between pilot stem stats and a global "source" reference
        - ch_mean, ch_std: mean/std of |h| over encoded symbols under the current channel
        - pred_entropy: mean predictive entropy H(p), NOT normalized
        - top1_margin: mean (p1 - p2) over samples
        - T_e_probe, T_d_probe: measured encoder/decoder time (average over repeats)
    """
    dataset_name, channel_type = domain
    ch_args = {}
    if channel_type == 'rician':
        ch_args['K'] = 3.0

    img_kl_vals = []
    h_means, h_stds = [], []
    ent_vals, margin_vals = [], []
    Te_vals, Td_vals = [], []

    for _ in range(max(1, pilot_repeats)):
        imgs, _ = sample_random_batch(loaders[dataset_name], bs=pilot_bs)
        imgs = imgs.to(device, non_blocking=True)

        # Stem stats for image shift
        stem_feat = stem_probe(imgs)
        cur_img_mean = stem_feat.mean(dim=(0,2,3)).cpu().numpy()
        cur_img_std  = stem_feat.std (dim=(0,2,3)).cpu().numpy()

        if img_ref_stats is not None:
            m_ref, s_ref = img_ref_stats
            img_kl = _sym_kl_diag_gauss(cur_img_mean, cur_img_std, m_ref, s_ref)
        else:
            img_kl = float(np.mean(cur_img_std**2))  # fallback: use variance magnitude
        img_kl_vals.append(float(img_kl))

        # Encoder timing + features
        t0 = time.perf_counter()
        feat, shp = model.encode(imgs)
        t1 = time.perf_counter()
        Te_vals.append(t1 - t0)

        # Channel fingerprint (no timing here; Tw is measured during real run)
        z_flat = feat.view(feat.size(0), -1)
        mod_det = QAM(num_embeddings, snr_db, channel_type=channel_type, channel_args=ch_args)
        mod_det.channel_estimates = []
        mod_det.apply_channel(z_flat)
        if len(getattr(mod_det, "channel_estimates", [])) > 0:
            est = torch.cat(mod_det.channel_estimates, dim=0)
        else:
            _, h_true = mod_det.channel.apply(z_flat, return_h=True)
            est = h_true
        h_mag = torch.norm(est, dim=1).cpu().numpy()
        h_means.append(float(h_mag.mean()))
        h_stds.append(float(h_mag.std()))

        # Full logits (one pass) for uncertainty/margin
        mod_inf = _build_modulator(mod_kind, num_embeddings, snr_db, channel_type, ch_args)
        feat_noisy, _ = model.sampler(feat, mod=mod_inf)

        t2 = time.perf_counter()
        logits = model.decode(feat_noisy, shp)
        t3 = time.perf_counter()
        Td_vals.append(t3 - t2)

        probs = _softmax_np(logits.detach().cpu().numpy())
        # entropy
        logp = np.log(np.clip(probs, 1e-12, None))
        ent = -np.sum(probs * logp, axis=1)  # in nats
        ent_vals.append(float(ent.mean()))
        # top-1 margin
        part = np.partition(probs, -2, axis=1)
        top2 = part[:, -2]
        top1 = part[:, -1]
        margin_vals.append(float((top1 - top2).mean()))

    return {
        "img_kl_ref": float(np.mean(img_kl_vals)),
        "ch_mean": float(np.mean(h_means)),
        "ch_std": float(np.mean(h_stds)),
        "pred_entropy": float(np.mean(ent_vals)),
        "top1_margin": float(np.mean(margin_vals)),
        "T_e_probe": float(np.mean(Te_vals)),
        "T_d_probe": float(np.mean(Td_vals)),
    }

def normalize_pilot_features(
    raw: Dict[str, float],
    num_classes: int,
    img_kl_max: float,
    ch_mean_max: float,
    ch_std_max: float,
    Te_max: float,
    Td_max: float,
) -> Dict[str, float]:
    """
    Map raw pilot features into [0,1] with simple, robust clipping and scaling.
    You can tune the *_max hyper-parameters from CLI if needed.
    """
    out = {}
    out["img_kl_ref"] = float(np.clip(raw.get("img_kl_ref", 0.0) / max(img_kl_max, 1e-6), 0.0, 1.0))
    out["ch_mean"] = float(np.clip(raw.get("ch_mean", 0.0) / max(ch_mean_max, 1e-6), 0.0, 1.0))
    out["ch_std"]  = float(np.clip(raw.get("ch_std",  0.0) / max(ch_std_max,  1e-6), 0.0, 1.0))
    ent_raw = raw.get("pred_entropy", 0.0)
    ent_norm = ent_raw / max(math.log(max(2, num_classes)), 1e-6)
    out["pred_entropy_norm"] = float(np.clip(ent_norm, 0.0, 1.0))
    out["top1_margin"] = float(np.clip(raw.get("top1_margin", 0.0), 0.0, 1.0))
    out["T_e_probe"] = float(np.clip(raw.get("T_e_probe", 0.0) / max(Te_max, 1e-6), 0.0, 1.0))
    out["T_d_probe"] = float(np.clip(raw.get("T_d_probe", 0.0) / max(Td_max, 1e-6), 0.0, 1.0))
    return out


# ------------------------------ One task runner (fixed bs) ------------------------------

def _L_img_avg_per_image(B: int, F: float, T_e: float, T_w: float, T_d: float) -> float:
    """
    Average per-image end-to-end waiting for a batch of size B:
      L_img(B) = (B-1)/(2F) + (T_e + T_w + T_d)/B
    - First term: average in-batch queueing (penalizes large B).
    - Second term: amortized encode/tx/decode over B samples (penalizes tiny B).
    """
    B = max(1, int(B))
    F = max(1e-6, float(F))
    return (B - 1.0) / (2.0 * F) + (T_e + T_w + T_d) / float(B)

def run_task_fixed_bs(
    task: Dict,
    bs: int,
    device: torch.device,
    model: nn.Module,
    loaders: Dict[str, object],
    in_channels: int,
    latent_d: int,
    num_embeddings: int,
    mod_kind: str,
    snr_db: float,
    t_slot_s: float,
    bandwidth_hz: float,
    shift_conf: Dict,
    tn_m: float,
    fixed_domain: Optional[Tuple[str, str]] = None,
    available_domains: Optional[List[Tuple[str,str]]] = None,
    verbose: bool = True
):
    """
    Execute one whole task with a fixed batch size, returning:
      mean_acc(%) [sample-weighted], det_acc(%),
      avg_wait_per_img(s), avg_violation_per_img(s), steps, and the chosen domain.
    """
    from copy import deepcopy

    I  = int(task['I'])
    F  = int(task['F'])
    W_th = float(task['Td'])  # interpret Td as the per-image waiting threshold

    # Batches in this task
    ep_max_steps = max(1, math.ceil(I / bs))
    rng = random.Random(task.get('seed', 0))

    # ===== Fixed domain per task =====
    if fixed_domain is not None:
        domain_fixed = fixed_domain
    else:
        if not available_domains:
            # Fall back to whatever loaders we have (AWGN)
            any_ds = list(loaders.keys())[0]
            domain_fixed = (any_ds, 'awgn')
        else:
            domain_fixed = rng.choice(available_domains)
    domain_seq = [domain_fixed] * ep_max_steps  # same domain for all batches in this task
    dataset_fixed, channel_fixed = domain_fixed

    # State for the task
    delta_I = I
    total_wait = 0.0           # accumulate L_img(B) * B over the task
    sum_violation = 0.0        # accumulate max(0, L_img - W_th) * B
    correct_total = 0.0
    sample_total = 0
    det_correct = 0
    det_total = 0
    step_idx = 0
    tta_enabled = False
    ref_img_stats = None
    ref_ch_hist = None

    # Detection config aligned to this task's SNR
    shift_conf_local = dict(shift_conf)
    shift_conf_local['snr'] = snr_db
    shift_conf_local['num_embeddings'] = num_embeddings
    shift_conf_local['tn_m'] = tn_m

    # Frozen stem probe (deep copy of the trained stem)
    stem_probe = deepcopy(model.prep[0]).to(device).eval()
    for p in stem_probe.parameters():
        p.requires_grad_(False)

    # Print the chosen domain once at the beginning
    print(f"  >> fixed_domain={dataset_fixed}/{channel_fixed}")

    while step_idx < ep_max_steps:
        dataset_name, channel_type = domain_seq[step_idx]
        if step_idx == 0:
            prev_dataset, prev_channel = dataset_name, channel_type
        else:
            prev_dataset, prev_channel = domain_seq[step_idx - 1]
        gt_any_shift = ((dataset_name != prev_dataset) or (channel_type != prev_channel))

        # Sample batch (clip by remaining images)
        cur_bs = min(bs, delta_I)
        imgs, labels = sample_random_batch(loaders[dataset_name], bs=cur_bs)
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Channel args + modulators (inference vs detection)
        ch_args = {}
        if channel_type == 'rician': ch_args['K'] = 3.0

        # Inference mod
        mod = _build_modulator(mod_kind, num_embeddings, snr_db, channel_type, ch_args)

        # Detection mod: force QAM to collect channel estimates
        mod_det = QAM(shift_conf_local['num_embeddings'],
                      shift_conf_local['snr'],
                      channel_type=channel_type,
                      channel_args=ch_args)

        # ===================== Detection ===================== #
        with torch.no_grad():
            stem_feat = stem_probe(imgs)
            cur_img_mean = stem_feat.mean(dim=(0,2,3)).cpu().numpy()
            cur_img_std  = stem_feat.std (dim=(0,2,3)).cpu().numpy()

        with torch.no_grad():
            z_enc  = model.encoder(imgs)
            z_flat = z_enc.view(z_enc.size(0), -1)

        # Primary path to collect channel estimates
        mod_det.channel_estimates = []
        mod_det.apply_channel(z_flat)
        if len(getattr(mod_det, "channel_estimates", [])) > 0:
            est = torch.cat(mod_det.channel_estimates, dim=0)
        else:
            _, h_true = mod_det.channel.apply(z_flat, return_h=True)
            est = h_true

        h_mag = torch.norm(est, dim=1).cpu().numpy()
        std_eps = h_mag.std() + 1e-6
        h_norm = (h_mag - h_mag.mean()) / std_eps
        cur_ch_hist, _ = np.histogram(h_norm, bins=50, range=(-5,5), density=True)

        if step_idx == 0:
            det_any = False
            img_score = 0.0
            ch_score = 0.0
            ref_img_stats = (cur_img_mean, cur_img_std)
            ref_ch_hist   = cur_ch_hist
        else:
            img_metric = shift_conf_local.get("img_metric", "kl").lower()
            if img_metric == "kl":
                img_score = _sym_kl_diag_gauss(cur_img_mean, cur_img_std,
                                               ref_img_stats[0], ref_img_stats[1])
            elif img_metric == "wasserstein":
                img_score = _w2_gaussian(cur_img_mean, cur_img_std,
                                         ref_img_stats[0], ref_img_stats[1])
            else:
                raise ValueError(f"Unsupported image metric: {img_metric}")
            det_img = (img_score > float(shift_conf_local['thresh_img']))

            ch_metric = shift_conf_local.get("ch_metric", "wasserstein").lower()
            if ch_metric == "wasserstein":
                from scipy.stats import wasserstein_distance
                ch_score = float(wasserstein_distance(cur_ch_hist, ref_ch_hist))
            elif ch_metric == "kl":
                from scipy.stats import entropy
                eps = 1e-8
                ch_score = float(entropy(cur_ch_hist + eps, ref_ch_hist + eps))
            elif ch_metric == "l2":
                ch_score = float(np.linalg.norm(cur_ch_hist - ref_ch_hist))
            else:
                raise ValueError(f"Unsupported ch_metric: {ch_metric}")
            det_ch = (ch_score > float(shift_conf_local['thresh_ch']))
            det_any = bool(det_img or det_ch)

            ref_img_stats = (cur_img_mean, cur_img_std)
            ref_ch_hist   = cur_ch_hist

        # ===================== Inference (current batch uses OLD stats) ===================== #
        use_cuda = (torch.cuda.is_available() and imgs.is_cuda)
        if use_cuda: torch.cuda.synchronize(imgs.device)

        if (step_idx > 0) and (not tta_enabled) and det_any:
            replace_bn_with_ttn(model, m=float(shift_conf_local.get('tn_m', 10.0)))
            set_ttn_mode(model, "infer_old")
            tta_enabled = True

        en0 = time.perf_counter()
        with torch.no_grad():
            if tta_enabled:
                set_ttn_mode(model, "infer_old")
            feat, shp = model.encode(imgs)
        if use_cuda: torch.cuda.synchronize(imgs.device)
        en1 = time.perf_counter()

        with torch.no_grad():
            feat_noisy, _ = model.sampler(feat, mod=mod)

        de0 = time.perf_counter()
        with torch.no_grad():
            logits = model.decode(feat_noisy, shp)
        if use_cuda: torch.cuda.synchronize(imgs.device)
        de1 = time.perf_counter()

        # Measured batch processing times
        T_e = en1 - en0
        T_d = de1 - de0
        T_w = obtain_tx_latency_seconds_timeslot_fading(
            feat, num_embeddings, snr_db, t_slot_s, bandwidth_hz, channel_fixed, ch_args
        )

        # --- Average per-image waiting for this batch ---
        L_img = _L_img_avg_per_image(cur_bs, float(F), float(T_e), float(T_w), float(T_d))
        over = max(0.0, L_img - W_th)

        # Accuracy
        top1, _ = accuracy(logits, labels, (1, 3))
        v_b = float(top1.item())

        # Accounting (weight by number of images in this batch)
        total_wait += L_img * cur_bs
        sum_violation += over * cur_bs

        correct_total += (v_b / 100.0) * cur_bs
        sample_total  += cur_bs

        step_idx += 1
        if step_idx >= 1:
            det_total += 1
            det_correct += int(det_any == bool(gt_any_shift))

        if verbose:
            print(
                f"  step={step_idx:03d}  bs={cur_bs:<4d}  "
                f"domain={dataset_fixed}/{channel_fixed:<12s}  "
                f"img_score={img_score:.5f}(th={shift_conf_local['thresh_img']})  "
                f"ch_score={ch_score:.6f}(th={shift_conf_local['thresh_ch']})  "
                f"det_any={int(det_any)}  "
                f"T_e={T_e:.4f}s  T_w={T_w:.4f}s  T_d={T_d:.4f}s  "
                f"L_img={L_img:.4f}s  over={over:.4f}s"
            )

        delta_I -= cur_bs

    # Task-level aggregates (per-image)
    mean_acc = 100.0 * (correct_total / max(1, sample_total))
    det_acc = (det_correct / max(1, det_total)) * 100.0
    avg_wait_per_img = total_wait / max(1, I)
    avg_violation_per_img = sum_violation / max(1, I)

    return {
        "mean_acc": float(mean_acc),
        "det_acc": float(det_acc),
        "avg_wait_per_img": float(avg_wait_per_img),
        "avg_violation_per_img": float(avg_violation_per_img),
        "sum_violation": float(sum_violation),
        "steps": int(ep_max_steps),
        "domain": f"{dataset_fixed}/{channel_fixed}",
    }


# ------------------------------ Data/domain preparation ------------------------------

def ensure_domain_loaders(dataset_base: str) -> Dict[str, object]:
    """
    Prepare DataLoaders for all possible domains of the given base.
    We only need .dataset; bs=1, workers=0 is fine. Missing domains will be skipped with a warning.
    """
    candidate_names = _candidate_domain_names_for_base(dataset_base)
    loaders = {}
    for ds in candidate_names:
        try:
            loaders[ds] = get_data(ds, batch_size=1, n_worker=0, train=False)
        except Exception as e:
            print(f"[WARN] skip domain '{ds}': {e}")
    if not loaders:
        # As a last resort, try the bare base name
        base_up = _dataset_base_upper(dataset_base)
        loaders[base_up] = get_data(base_up, batch_size=1, n_worker=0, train=False)
    return loaders


# ------------------------------ Task/context utils ------------------------------

def normalize_core_context(task: Dict, I_max: int, F_max: int, Td_max: int,
                           snr_min: float, snr_max: float) -> Dict[str, float]:
    """Basic normalized context from explicit task parameters (Td is used as W_th)."""
    I  = float(task['I'])
    F  = float(task['F'])
    Td = float(task['Td'])  # per-image waiting threshold
    snr_db = float(task['snr_db'])
    return {
        'I_norm':  float(np.clip(I / I_max, 0.0, 1.0)),
        'F_norm':  float(np.clip(F / F_max, 0.0, 1.0)),
        'Td_norm': float(np.clip(Td / Td_max, 0.0, 1.0)),
        'snr_norm': float(np.clip((snr_db - snr_min) / (snr_max - snr_min + 1e-8), 0.0, 1.0)),
    }

def sample_task(rng: random.Random, args, t_idx: int,
                snr_min_r: float, snr_max_r: float,
                all_domains: List[Tuple[str,str]]) -> Tuple[Dict, Tuple[str,str]]:
    """Sample one synthetic task with fixed SNR and fixed domain."""
    task_snr = float(rng.uniform(snr_min_r, snr_max_r))
    task_domain = rng.choice(all_domains)
    task = {
        'I': int(rng.randint(args.I_min_r, args.I_max_r)),
        'F': int(rng.randint(args.F_min_r, args.F_max_r)),
        # Treat Td as per-image waiting threshold W_th
        'Td': float(rng.uniform(args.Td_min_r, args.Td_max_r)),
        'snr_db': task_snr,
        'seed': int(args.seed + t_idx),
    }
    return task, task_domain

def compute_reward(summary: Dict, kappa: float) -> float:
    """Task-level reward: mean_acc/100 - kappa * avg_violation_per_img."""
    return float(summary['mean_acc'] / 100.0) - float(kappa) * float(summary['avg_violation_per_img'])


# ------------------------------ Global "source" reference for pilot ------------------------------

@torch.no_grad()
def compute_global_img_ref(stem_probe: nn.Module, loaders: Dict[str, object],
                           device: torch.device, ref_domain: str,
                           ref_bs: int = 256, repeats: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a global "source" image-stat reference from a clean domain (e.g., CIFAR10 or CIFAR100).
    Used by pilot to measure symmetric-KL distance.
    """
    if ref_domain not in loaders:
        # Fallback: pick the first available loader
        ref_domain = list(loaders.keys())[0]
        print(f"[WARN] ref_domain not found in loaders, fallback to '{ref_domain}'")

    means, stds = [], []
    for _ in range(max(1, repeats)):
        imgs, _ = sample_random_batch(loaders[ref_domain], bs=ref_bs)
        imgs = imgs.to(device, non_blocking=True)
        feat = stem_probe(imgs)
        means.append(feat.mean(dim=(0,2,3)).cpu().numpy())
        stds.append (feat.std (dim=(0,2,3)).cpu().numpy())
    m = np.mean(np.stack(means, axis=0), axis=0)
    s = np.mean(np.stack(stds,  axis=0), axis=0)
    return m, s


# ------------------------------ Train-time chooser ------------------------------

def _choose_action_train(optimizer, ctx: Dict[str, float], utility, bs_candidates: np.ndarray,
                         t_idx: int, args, x2bs: Dict[float, int]) -> int:
    """
    Train-time chooser with (i) warmup rotation, (ii) epsilon exploration,
    and (iii) nearest-neighbor nudge when the greedy pick is an extreme.
    Suggestion may return key 'a' (log2(bs)) or 'bs' (legacy).
    """
    # Greedy pick from BO
    act = optimizer.suggest(ctx, utility)
    if 'a' in act:
        best = int(x2bs[float(act['a'])])
    else:
        best = int(act['bs'])

    # Epsilon exploration
    if random.random() < float(args.eps_explore):
        return int(random.choice(list(bs_candidates)))

    # Neighbor when extreme
    if (best == int(bs_candidates.min()) or best == int(bs_candidates.max())):
        if random.random() < float(args.neighbor_when_extreme):
            i = int(np.where(bs_candidates == best)[0][0])
            j = max(0, min(len(bs_candidates) - 1, i + (1 if best == int(bs_candidates.min()) else -1)))
            return int(bs_candidates[j])

    return best


# ------------------------------ Main (training only) ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Contextual MAB for task-level batch-size selection (AdapS/JSCC).")

    # Train size / seed
    parser.add_argument("--num_train_tasks", type=int, default=500, help="Number of training tasks for MAB.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for reproducibility.")

    # Model / data
    parser.add_argument("--dataset_base", type=str, default="CINIC10",
                        choices=["CIFAR10","CIFAR100","CINIC10"])
    parser.add_argument("--ckpt", type=str, default="JT-JSCC/trained_models/cinic10_best.pt",
                        help="Path to JSCC checkpoint (.pt).")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_d", type=int, default=512)
    parser.add_argument("--num_embeddings", type=int, default=16)
    parser.add_argument("--mod", type=str, default="psk", choices=["psk","qam"])

    # SNR sampling per task (fixed within task)
    parser.add_argument("--snr_min_r", type=float, default=0.0, help="Min SNR(dB) for task sampling.")
    parser.add_argument("--snr_max_r", type=float, default=20.0, help="Max SNR(dB) for task sampling.")

    # Shift detection config (in-task)
    parser.add_argument("--img_metric", type=str, default="kl", choices=["kl","wasserstein"])
    parser.add_argument("--ch_metric", type=str, default="wasserstein", choices=["wasserstein","kl","l2"])
    parser.add_argument("--thresh_img", type=float, default=0.8)
    parser.add_argument("--thresh_ch", type=float, default=0.1)
    parser.add_argument("--tn_m", type=float, default=10.0, help="TTNorm momentum param m.")

    # Wireless latency params
    parser.add_argument("--t_slot_s", type=float, default=1e-3)
    parser.add_argument("--bandwidth_hz", type=float, default=2e6)

    # Reward
    parser.add_argument("--kappa", type=float, default=1.5)

    # Action space
    parser.add_argument("--actions", type=str, default="2,4,8,16,32,64,128,256",
                        help="Comma-separated candidate batch sizes.")

    # Context normalization ranges (for explicit task params)
    parser.add_argument("--I_max", type=int, default=1024)
    parser.add_argument("--F_max", type=int, default=60)
    parser.add_argument("--Td_max", type=int, default=50)   # used to normalize W_th
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)

    # Task ranges (for synthetic task generator)
    parser.add_argument("--I_min_r", type=int, default=512)
    parser.add_argument("--I_max_r", type=int, default=1024)
    parser.add_argument("--F_min_r", type=int, default=30)
    parser.add_argument("--F_max_r", type=int, default=60)
    # Td_* is interpreted as per-image waiting threshold W_th (kept for CLI compatibility)
    parser.add_argument("--Td_min_r", type=float, default=0.3)
    parser.add_argument("--Td_max_r", type=float, default=1.0)

    # Pilot options (MAB context enrichment)
    parser.add_argument("--pilot_ctx", action="store_true", default=True,
                        help="Enable pilot-based continuous fingerprints in context.")
    parser.add_argument("--pilot_bs", type=int, default=64)
    parser.add_argument("--pilot_repeats", type=int, default=2)
    parser.add_argument("--pilot_img_kl_max", type=float, default=3.0)
    parser.add_argument("--pilot_ch_mean_max", type=float, default=2.0)
    parser.add_argument("--pilot_ch_std_max", type=float, default=1.5)
    parser.add_argument("--pilot_Te_max", type=float, default=0.2)
    parser.add_argument("--pilot_Td_max", type=float, default=0.2)

    # Logging
    parser.add_argument("--log_csv", type=str, default="", help="Append per-task TRAIN summary to this CSV.")
    parser.add_argument("--verbose", action="store_true", help="Print per-batch logs.")

    # Persistence
    parser.add_argument("--save_mab", type=str, default="AdapS/trained_models/cinic10_mab_adaps.pkl",
                        help="Path to save the trained MAB policy (cloudpickle).")

    # Anti-collapse & smoothing
    parser.add_argument("--eps_explore", type=float, default=0.15,
                        help="Min exploration rate during training.")
    parser.add_argument("--neighbor_when_extreme", type=float, default=0.50,
                        help="If the greedy action is extreme, probability to try its nearest neighbor instead.")
    parser.add_argument("--warmup_random", type=int, default=400,
                        help="Number of warmup tasks (we'll rotate actions in warmup).")
    parser.add_argument("--action_length_scale", type=float, default=3.0,
                        help="Length-scale multiplier on the GP action dimension to smooth action responses.")

    args = parser.parse_args()

    # RNG
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Build model & load weights
    num_classes = _infer_num_classes(args.dataset_base)
    model = _build_model(args.dataset_base, args.in_channels, args.latent_d,
                         num_classes, args.num_embeddings).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if args.ckpt:
        try:
            ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
        except TypeError:
            ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt["model_states"])
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"[Init] Loaded JSCC weights from '{args.ckpt}'.")
    else:
        print("[Init] WARNING: No checkpoint provided. Model uses random weights.")

    # Domain data loaders (dynamic by base; skip missing)
    loaders = ensure_domain_loaders(args.dataset_base)

    # Build domain tuples from actually loaded keys
    all_domains = _build_domains_from_loader_keys(list(loaders.keys()), args.dataset_base)
    if not all_domains:
        raise RuntimeError("No valid domains could be constructed. Check your dataloaders.")

    # Frozen stem probe (shared by pilot & in-task detector)
    from copy import deepcopy
    stem_probe = deepcopy(model.prep[0]).to(device).eval()
    for p in stem_probe.parameters():
        p.requires_grad_(False)

    # Global "source" image reference for pilot KL (clean base domain)
    base_clean = _dataset_base_upper(args.dataset_base)
    img_ref_stats = compute_global_img_ref(stem_probe, loaders, device, ref_domain=base_clean,
                                           ref_bs=256, repeats=2)

    # Shift detection base config (SNR will be overwritten per task)
    shift_conf = {
        'img_metric': args.img_metric,
        'ch_metric':  args.ch_metric,
        'thresh_img': args.thresh_img,
        'thresh_ch':  args.thresh_ch,
        'num_embeddings': args.num_embeddings,
        'snr': 0.0,  # placeholder; per-task snr_db will be set inside runner
        'batches_per_domain': 8,  # kept for compatibility
        'ch_est_var': 0.0,
        'tn_m': args.tn_m,
    }

    # Action space (real system actions) and log2 encoding for GP
    bs_candidates = np.array([int(x) for x in args.actions.split(",") if x.strip() != ""], dtype=int)
    bs2x = {int(b): float(np.log2(b)) for b in bs_candidates}
    x2bs = {v: k for k, v in bs2x.items()}
    disc_actions = {'a': np.array([bs2x[int(b)] for b in bs_candidates], dtype=float)}  # 'a' == log2(bs)

    # Context spec: base + optional pilot features
    ctx_spec = {
        'I_norm':'', 'F_norm':'', 'Td_norm':'', 'snr_norm':''
    }
    if args.pilot_ctx:
        ctx_spec.update({
            'img_kl_ref':'', 'ch_mean':'', 'ch_std':'',
            'pred_entropy_norm':'', 'top1_margin':'',
            'T_e_probe':'', 'T_d_probe':''
        })

    # GP kernel
    action_dim = 1
    context_dim = len(ctx_spec)
    length_scale = np.ones(context_dim + action_dim, dtype=float)
    length_scale[-1] = args.action_length_scale  # smoother along action (log2) dimension
    kernel = WhiteKernel(noise_level=1.0) + Matern(nu=1.5, length_scale=length_scale)

    optimizer = ContextualBayesianOptimization(
        all_actions_dict=disc_actions,
        contexts=ctx_spec,
        kernel=kernel,
        noise=1e-3,      # larger observation noise
        init_random=5
    )

    # ---------- Training ----------
    train_stats = []

    # Optional CSV logger for train
    if args.log_csv:
        new_file = not os.path.exists(args.log_csv)
        ftr = open(args.log_csv, "a", newline="")
        csvw_tr = csv.writer(ftr)
        if new_file:
            header = ["phase","task_idx","I","F","W_th","snr_db","domain","chosen_bs",
                      "mean_acc","det_acc","avg_wait_per_img","avg_violation_per_img","steps","reward"]
            if args.pilot_ctx:
                header = header[:7] + ["img_kl_ref","ch_mean","ch_std","pred_entropy_norm","top1_margin",
                                       "T_e_probe","T_d_probe"] + header[7:]
            csvw_tr.writerow(header)
    else:
        ftr = None
        csvw_tr = None

    # Warmup rotation sequence (length = args.warmup_random)
    warm_N = max(0, int(args.warmup_random))
    if warm_N > 0:
        cycles = int(math.ceil(warm_N / len(bs_candidates)))
        warm_seq = (list(bs_candidates) * cycles)[:warm_N]
    else:
        warm_seq = []

    rng_tr = random.Random(args.seed)
    for t_idx in range(args.num_train_tasks):
        task, task_domain = sample_task(rng_tr, args, t_idx, args.snr_min_r, args.snr_max_r, all_domains)

        # Build context
        ctx = normalize_core_context(task, args.I_max, args.F_max, args.Td_max, args.snr_min, args.snr_max)

        pilot_norm = {}
        if args.pilot_ctx:
            pilot_raw = compute_pilot_context(
                model=model, stem_probe=stem_probe, loaders=loaders, domain=task_domain,
                mod_kind=args.mod, num_embeddings=args.num_embeddings, snr_db=task['snr_db'],
                device=device, num_classes=num_classes,
                pilot_bs=args.pilot_bs, pilot_repeats=args.pilot_repeats,
                img_ref_stats=img_ref_stats
            )
            pilot_norm = normalize_pilot_features(
                pilot_raw, num_classes=num_classes,
                img_kl_max=args.pilot_img_kl_max,
                ch_mean_max=args.pilot_ch_mean_max,
                ch_std_max=args.pilot_ch_std_max,
                Te_max=args.pilot_Te_max,
                Td_max=args.pilot_Td_max
            )
            ctx.update(pilot_norm)

        # UCB beta schedule: large -> small
        beta0, beta_min = 8.0, 1.5
        frac = t_idx / max(1, args.num_train_tasks)
        beta = max(beta_min, beta0 * (1.0 - frac))
        utility = UtilityFunction(kind="ucb", beta_kind='const', beta_const=beta)

        # Choose action
        if t_idx < len(warm_seq):
            bs = int(warm_seq[t_idx])  # deterministic rotation in warmup
        else:
            bs = _choose_action_train(optimizer, ctx, utility, bs_candidates, t_idx, args, x2bs)

        # Print per-task header
        print(f"\n[TRAIN Task {t_idx}] I={task['I']} F={task['F']} W_th={task['Td']:.3f}s  snr_db={task['snr_db']:.2f}  "
              f"| choose bs={bs} | domain={task_domain[0]}/{task_domain[1]}")

        summary = run_task_fixed_bs(
            task=task,
            bs=bs,
            device=device,
            model=model,
            loaders=loaders,
            in_channels=args.in_channels,
            latent_d=args.latent_d,
            num_embeddings=args.num_embeddings,
            mod_kind=args.mod,
            snr_db=task['snr_db'],
            t_slot_s=args.t_slot_s,
            bandwidth_hz=args.bandwidth_hz,
            shift_conf=shift_conf,
            tn_m=args.tn_m,
            fixed_domain=task_domain,
            available_domains=all_domains,
            verbose=args.verbose
        )

        reward_raw = compute_reward(summary, args.kappa)

        print(
            f"[TRAIN Task {t_idx}] steps={summary['steps']} bs={bs} "
            f"| reward={reward_raw:.6f}  "
            f"mean_acc={summary['mean_acc']:.2f}%  det_acc={summary['det_acc']:.2f}%  "
            f"avg_wait/img={summary['avg_wait_per_img']:.6f}s  "
            f"avg_violation/img={summary['avg_violation_per_img']:.6f}s"
        )

        # Register with BO (use log2 action value)
        optimizer.register(ctx, {'a': bs2x[int(bs)]}, reward_raw)
        train_stats.append(reward_raw)

        if csvw_tr is not None:
            row = ["train", t_idx, task['I'], task['F'], f"{task['Td']:.6f}", f"{task['snr_db']:.3f}",
                   f"{task_domain[0]}/{task_domain[1]}", bs]
            if args.pilot_ctx:
                row += [f"{pilot_norm.get('img_kl_ref',0):.6f}", f"{pilot_norm.get('ch_mean',0):.6f}",
                        f"{pilot_norm.get('ch_std',0):.6f}", f"{pilot_norm.get('pred_entropy_norm',0):.6f}",
                        f"{pilot_norm.get('top1_margin',0):.6f}", f"{pilot_norm.get('T_e_probe',0):.6f}",
                        f"{pilot_norm.get('T_d_probe',0):.6f}"]
            row += [f"{summary['mean_acc']:.4f}", f"{summary['det_acc']:.4f}",
                    f"{summary['avg_wait_per_img']:.8f}",
                    f"{summary['avg_violation_per_img']:.8f}",
                    summary['steps'], f"{reward_raw:.8f}"]
            csvw_tr.writerow(row)
            ftr.flush()

    if 'ftr' in locals() and ftr is not None:
        ftr.close()
    print(f"\n== TRAIN complete: avg reward = {np.mean(train_stats):.6f} over {len(train_stats)} tasks ==")

    if args.save_mab:
        try:
            meta = {
                "actions": list(map(int, bs_candidates.tolist())),
                "context_keys": list(ctx_spec.keys()),
                "kappa": float(args.kappa),
                "utility": {"kind": "ucb", "beta_kind": "const"},
                "norm_ranges": {
                    "I_max": args.I_max, "F_max": args.F_max, "Td_max": args.Td_max,
                    "snr_min": args.snr_min, "snr_max": args.snr_max
                },
                "encoding": "log2(bs)",
                "dataset_base": _dataset_base_upper(args.dataset_base),
                "domains_loaded": list(loaders.keys())
            }
            payload = {"optimizer": optimizer, "meta": meta}
            os.makedirs(os.path.dirname(args.save_mab), exist_ok=True)
            with open(args.save_mab, "wb") as f:
                cp.dump(payload, f)
            print(f"[SAVE] MAB policy (cloudpickle) -> {args.save_mab}")
        except Exception as e:
            print(f"[SAVE] Failed to save MAB to {args.save_mab}: {e}")

    print("\n== Done ==")


if __name__ == "__main__":
    main()
