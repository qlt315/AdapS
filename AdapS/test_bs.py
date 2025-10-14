# test_bs_fixed.py
# Standalone batch-size probe (fixed batch size)
# Flow mirrors the env: random batch sampling, domain sequence + shift detection,
# deferred-apply TTNorm, JSCC encode->modulate->decode, and latency accounting.
# Key fixes vs. previous attempts:
#   * Wireless latency SNR aligned to snr_db: SNR0 = 10**(snr_db/10).
#   * Separate channel kwargs for modulators from latency params (no _log2U leakage).

from __future__ import annotations

import argparse
import math
import random
import time
from copy import deepcopy
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn

# Project imports expected in your repo layout
from JSCC.datasets.dataloader import get_data
from JSCC.utils.modulation import QAM, PSK
from JSCC.utils.accuracy import accuracy
import model.DT_JSCC as JSCC_model


# ------------------------- Tiny sampler (no iterator advance) -------------------------

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


# ------------------------------ Helpers ---------------------------------- #

_DATASET_BASE_RE = __import__('re').compile(
    r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=__import__('re').IGNORECASE
)

def _dataset_base(dataset_name: str) -> str:
    m = _DATASET_BASE_RE.match(str(dataset_name).strip())
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).lower()

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
                     channel: str, ch_args: Dict[str, Any]) -> object:
    return (QAM if mod.lower()=="qam" else PSK)(
        num_embeddings, snr_db, channel, ch_args
    )

def _domain_sequence(batches_per_domain: int,
                     sim_len: int,
                     rng: random.Random) -> List[Tuple[str, str]]:
    """
    Per-batch domain sequence: start with source, then shifts; each domain lasts
    'batches_per_domain' steps. Same scheme as env.
    """
    source_domain = ('CIFAR10', 'awgn')
    shift_domains = [
        ('CIFAR10_noise',  'awgn'), ('CIFAR10_noise',  'rician'), ('CIFAR10_noise',  'rayleigh'),
        ('CIFAR10_fog',    'awgn'), ('CIFAR10_fog',    'rician'), ('CIFAR10_fog',    'rayleigh'),
        ('CIFAR10_bright', 'awgn'), ('CIFAR10_bright', 'rician'), ('CIFAR10_bright', 'rayleigh'),
        ('CIFAR10_snow',   'awgn'), ('CIFAR10_snow',   'rician'), ('CIFAR10_snow',   'rayleigh'),
        ('CIFAR10_rain',   'awgn'), ('CIFAR10_rain',   'rician'), ('CIFAR10_rain',   'rayleigh'),
        ('CIFAR10_blur',   'awgn'), ('CIFAR10_blur',   'rician'), ('CIFAR10_blur',   'rayleigh'),
    ]
    seq: List[Tuple[str, str]] = []
    while len(seq) < sim_len:
        if not seq:
            seq.extend([source_domain]*batches_per_domain)
        else:
            seq.extend([rng.choice(shift_domains)]*batches_per_domain)
    return seq[:sim_len]


# ---------- metrics used by detection script ----------

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

        # "normal" (not used)
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


# ---------- wireless latency with per-timeslot fading (SNR aligned to snr_db) ---------- #

def obtain_tx_latency(latent_feat: torch.Tensor,
                      channel_type: str,
                      t_slot_s: float,
                      bandwidth_hz: float,
                      snr_db: float,
                      log2U: float,
                      K: Optional[float] = None) -> float:
    """
    Wireless latency with per-timeslot fading; SNR aligned to snr_db.

    - Payload bits L_b = (#symbols per img) * bs * log2(U)
    - Each slot i has |h|^2_i drawn i.i.d. by channel_type.
      AWGN: |h|^2 = 1; Rayleigh: Exp(1); Rician(K): normalized with E|h|^2=1.
    - Capacity per slot: C_i = rho * log2(1 + SNR0 * |h|^2_i), SNR0 = 10**(snr_db/10).
    - Draw slots until accumulated bits >= L_b; T_w = n_slots * t_slot_s.
    """
    bs = latent_feat.size(0)
    z = latent_feat.view(bs, -1)
    symbols_per_img = z.size(1)
    payload_bits = symbols_per_img * bs * log2U

    # Linear SNR aligned to snr_db
    snr0 = 10 ** (snr_db / 10.0)

    def draw_h2(n: int) -> np.ndarray:
        ch = channel_type.lower()
        if ch == 'awgn':
            return np.ones(n, dtype=np.float64)
        if ch == 'rayleigh':
            return np.random.exponential(scale=1.0, size=n)  # unit-mean
        if ch == 'rician':
            KK = 3.0 if K is None else float(K)
            mu = math.sqrt(KK / (KK + 1.0))
            sigma = math.sqrt(1.0 / (2.0 * (KK + 1.0)))
            re = np.random.normal(loc=mu, scale=sigma, size=n)
            im = np.random.normal(loc=0.0, scale=sigma, size=n)
            return re*re + im*im
        return np.ones(n, dtype=np.float64)

    bits_acc = 0.0
    slots = 0
    max_slots = int(1e9)  # safety cap
    while bits_acc < payload_bits and slots < max_slots:
        h2 = draw_h2(1)[0]
        cap = bandwidth_hz * math.log2(1.0 + snr0 * h2)  # bits/sec
        if cap <= 0:
            cap = 1e3
        bits_acc += cap * t_slot_s
        slots += 1
    return slots * t_slot_s


# ------------------------------ Core pipeline (no episodes) ------------------------------ #

def run_fixed_bs_task(
    *,
    batch_size: int,
    I: int,
    F: int,
    Td: float,
    dataset_base: str = "CIFAR10",
    in_channels: int = 3,
    latent_d: int = 512,
    num_embeddings: int = 16,
    mod: str = "psk",
    snr_db: float = 8.0,
    t_slot_s: float = 1e-3,
    bandwidth_hz: float = 2e6,
    tn_m: float = 10.0,
    batches_per_domain: int = 8,
    img_metric: str = "kl",
    ch_metric: str = "wasserstein",
    thresh_img: float = 0.8,
    thresh_ch: float = 0.1,
    seed: Optional[int] = None,
    ckpt: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process a single "task" with a fixed batch size until I images are consumed.
    Mirrors the env step-by-step flow, but without any RL/episodes.
    """

    # RNG + device
    rng = random.Random(seed)
    np.random.seed(seed if seed is not None else 0)
    torch.manual_seed(seed if seed is not None else 0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available() and device.type == 'cuda'

    # Build model
    num_classes = _infer_num_classes(dataset_base)
    model = _build_model(dataset_base, in_channels, latent_d, num_classes, num_embeddings).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)

    # Load checkpoint if provided
    if ckpt:
        try:
            ckpt_obj = torch.load(ckpt, map_location='cpu')
            model.load_state_dict(ckpt_obj["model_states"])
            model.to(device).eval()
            for p in model.parameters(): p.requires_grad_(False)
            print(f"[Info] Loaded JSCC weights from '{ckpt}'.")
        except Exception as e:
            print(f"[Warn] Failed to load checkpoint '{ckpt}': {e}")

    # Frozen image-stats probe (deep copy of prep[0])
    stem_probe = deepcopy(model.prep[0]).to(device).eval()
    for p in stem_probe.parameters(): p.requires_grad_(False)

    # DataLoaders for all possible domains (we only need .dataset)
    loaders: Dict[str, Any] = {}
    for ds in {'CIFAR10','CIFAR10_noise','CIFAR10_fog','CIFAR10_bright','CIFAR10_snow','CIFAR10_rain','CIFAR10_blur'}:
        loaders[ds] = get_data(ds, batch_size=1, n_worker=0, train=False)

    # Domain sequence (length = ceil(I / batch_size))
    max_steps = max(1, math.ceil(I / float(batch_size)))
    domain_seq = _domain_sequence(batches_per_domain, max_steps, rng)

    # Accumulators
    delta_I = I
    elapsed_T = 0.0
    acc_correct_total = 0.0
    sample_total = 0
    det_correct = 0
    det_total = 0
    tta_enabled = True
    ref_img_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ref_ch_hist: Optional[np.ndarray] = None

    # Convenience
    log2U = math.log2(num_embeddings)

    if verbose:
        print(f"[Task] I={I}, F={F}, Td={Td:.3f}s | bs={batch_size} | snr_db={snr_db} | TTA={tta_enabled}")

    step_idx = 0
    sum_violation = 0.0

    while delta_I > 0:
        bs = min(batch_size, delta_I)

        # Current/previous domain (for GT shift)
        dataset_name, channel_type = domain_seq[step_idx]
        if step_idx == 0:
            prev_dataset, prev_channel = dataset_name, channel_type
        else:
            prev_dataset, prev_channel = domain_seq[step_idx - 1]
        gt_any_shift = ((dataset_name != prev_dataset) or (channel_type != prev_channel))

        # Sample data
        imgs, labels = sample_random_batch(loaders[dataset_name], bs=bs)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Channel kwargs for modulators ONLY (no log2U leakage)
        ch_args_mod: Dict[str, Any] = {}
        K_val: Optional[float] = None
        if channel_type == 'rician':
            K_val = 3.0
            ch_args_mod['K'] = K_val

        # Inference modulator (respect config)
        modulator = _build_modulator(mod, num_embeddings, snr_db, channel_type, ch_args_mod)
        # Detection modulator: force QAM and match detector code path
        mod_det = QAM(num_embeddings, snr_db, channel_type=channel_type, channel_args=ch_args_mod)

        # ============== Shift detection (aligned with env) ==============
        with torch.no_grad():
            stem_feat = stem_probe(imgs)
            cur_img_mean = stem_feat.mean(dim=(0, 2, 3)).cpu().numpy()
            cur_img_std  = stem_feat.std (dim=(0, 2, 3)).cpu().numpy()

        with torch.no_grad():
            z_enc  = model.encoder(imgs)
            z_flat = z_enc.view(z_enc.size(0), -1)

        # Primary path: collect channel_estimates
        mod_det.channel_estimates = []
        mod_det.apply_channel(z_flat)

        # Fallback
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
            ref_img_stats = (cur_img_mean, cur_img_std)
            ref_ch_hist   = cur_ch_hist
            det_any = False
            img_score = ch_score = 0.0
        else:
            # image metric
            if img_metric.lower() == "kl":
                img_score = _sym_kl_diag_gauss(cur_img_mean, cur_img_std,
                                               ref_img_stats[0], ref_img_stats[1])
            elif img_metric.lower() == "wasserstein":
                img_score = _w2_gaussian(cur_img_mean, cur_img_std,
                                         ref_img_stats[0], ref_img_stats[1])
            else:
                raise ValueError(f"Unsupported image metric: {img_metric}")
            det_img = (img_score > float(thresh_img))

            # channel metric
            if ch_metric.lower() == "wasserstein":
                from scipy.stats import wasserstein_distance
                ch_score = float(wasserstein_distance(cur_ch_hist, ref_ch_hist))
            elif ch_metric.lower() == "kl":
                from scipy.stats import entropy
                eps = 1e-8
                ch_score = float(entropy(cur_ch_hist + eps, ref_ch_hist + eps))
            elif ch_metric.lower() == "l2":
                ch_score = float(np.linalg.norm(cur_ch_hist - ref_ch_hist))
            else:
                raise ValueError(f"Unsupported ch_metric: {ch_metric}")
            det_ch = (ch_score > float(thresh_ch))

            det_any = bool(det_img or det_ch)
            ref_img_stats = (cur_img_mean, cur_img_std)
            ref_ch_hist   = cur_ch_hist

        # ============== Inference (current batch uses OLD stats) ==============
        if use_cuda:
            torch.cuda.synchronize(device)

        # If a shift is detected and TTA not enabled yet: switch BN->TTNorm
        if (step_idx > 0) and (not tta_enabled) and det_any:
            replace_bn_with_ttn(model, m=float(tn_m))
            set_ttn_mode(model, "infer_old")
            tta_enabled = True

        en0 = time.perf_counter()
        with torch.no_grad():
            if tta_enabled:
                set_ttn_mode(model, "infer_old")
            feat, shp = model.encode(imgs)
        if use_cuda: torch.cuda.synchronize(device)
        en1 = time.perf_counter()

        with torch.no_grad():
            feat_noisy, _ = model.sampler(feat, mod=modulator)

        de0 = time.perf_counter()
        with torch.no_grad():
            logits = model.decode(feat_noisy, shp)
        if use_cuda: torch.cuda.synchronize(device)
        de1 = time.perf_counter()

        # --- measured batch processing times ---
        T_e = en1 - en0  # encoder (batch)
        T_d = de1 - de0  # decoder (batch)
        T_w = obtain_tx_latency(
            feat, channel_type,
            t_slot_s=t_slot_s, bandwidth_hz=bandwidth_hz, snr_db=snr_db,
            log2U=log2U, K=K_val
        )

        # === Batch-blocking latency model ===
        g = 1.0 / float(F)               # generation time per image
        T_a = bs * g + T_e + T_w + T_d   # per-image avg latency in this batch (== batch makespan)
        # ====================================

        top1, _ = accuracy(logits, labels, (1, 3))
        v_b = float(top1.item())  # percentage (0-100)

        # Deadline share for this batch & violation
        over = max(0.0, T_a - (Td * (bs / max(1, I))))
        sum_violation += over

        # ===================== Adapt phase (write stats for NEXT batch) ===================== #
        if tta_enabled:
            set_ttn_mode(model, "adapt")
            with torch.no_grad():
                _ = model.encode(imgs)
            set_ttn_mode(model, "infer_old")

        # Accounting
        delta_I -= bs
        elapsed_T += T_a

        # Weighted accuracy accumulation (by sample count)
        acc_correct_total += (v_b / 100.0) * bs
        sample_total  += bs

        step_idx += 1

        if step_idx >= 1:
            det_total += 1
            det_correct += int(det_any == bool(gt_any_shift))

        if verbose:
            print(
                f"  step={step_idx:03d}  bs={bs:<4d}  domain={dataset_name}/{channel_type:<8s}  "
                f"img_score={img_score:.5f}(th={thresh_img})  ch_score={ch_score:.5f}(th={thresh_ch})  "
                f"det_any={int(det_any)}  T_e={T_e:.4f}s  T_w={T_w:.4f}s  T_d={T_d:.4f}s  "
                f"T_a={T_a:.4f}s  acc1={v_b:.2f}%  over={over:.4f}s  remain_I={delta_I:<5d}"
            )

    mean_acc = 100.0 * (acc_correct_total / max(1, sample_total))
    det_acc = (det_correct / max(1, det_total)) * 100.0 if det_total > 0 else 0.0
    avg_latency_per_img = elapsed_T / float(I)

    summary = {
        "total_steps": step_idx,
        "mean_acc_percent": mean_acc,
        "det_acc_percent": det_acc,
        "total_latency_sec": elapsed_T,
        "avg_latency_per_img_sec": avg_latency_per_img,
        "sum_violation_sec": sum_violation,
        "I": I,
        "F": F,
        "Td": Td,
        "batch_size": batch_size,
        "snr_db": snr_db,
        "tta_used": tta_enabled,
    }

    print("\n== Task summary (fixed batch size) ==")
    print(
        f"steps={summary['total_steps']} | mean_acc={summary['mean_acc_percent']:.2f}% "
        f"| det_acc={summary['det_acc_percent']:.2f}% | total_latency={summary['total_latency_sec']:.4f}s "
        f"| avg_latency/img={summary['avg_latency_per_img_sec']:.6f}s | sum_violation={summary['sum_violation_sec']:.6f}s "
        f"| I={I}"
    )
    return summary


# ------------------------------------ CLI ------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Fixed-batch-size task probe (standalone).")
    parser.add_argument("--bs", type=int, default=128, help="Fixed batch size.")
    parser.add_argument("--I", type=int, default=2048, help="Total images in the task.")
    parser.add_argument("--F", type=int, default=90, help="Image generation rate (Hz).")
    parser.add_argument("--Td", type=float, default=172.0, help="Task deadline (seconds).")

    parser.add_argument("--dataset", type=str, default="CIFAR10", help="Dataset base (CIFAR10/CIFAR100/CINIC10).")
    parser.add_argument("--latent_d", type=int, default=512, help="Latent dimension.")
    parser.add_argument("--num_embeddings", type=int, default=16, help="Codebook size U (symbols).")
    parser.add_argument("--mod", type=str, default="psk", choices=["psk","qam"], help="Modulation kind for inference.")
    parser.add_argument("--snr_db", type=float, default=8.0, help="SNR in dB (used both by modulator and latency).")

    parser.add_argument("--t_slot_s", type=float, default=1e-3, help="Wireless slot duration (s).")
    parser.add_argument("--bandwidth_hz", type=float, default=2e6, help="Wireless bandwidth (Hz).")

    # Shift detection / TTNorm
    parser.add_argument("--batches_per_domain", type=int, default=8, help="Batches per domain segment in sequence.")
    parser.add_argument("--img_metric", type=str, default="kl", choices=["kl","wasserstein"])
    parser.add_argument("--ch_metric", type=str, default="wasserstein", choices=["wasserstein","kl","l2"])
    parser.add_argument("--thresh_img", type=float, default=0.8)
    parser.add_argument("--thresh_ch", type=float, default=0.1)
    parser.add_argument("--tn_m", type=float, default=10.0, help="TTNorm fusion inertia (m).")

    parser.add_argument("--ckpt", type=str, default="JT-JSCC/trained_models/cifar10_best.pt",
                        help="Path to JSCC checkpoint (.pt). If omitted, random weights are used.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-step logging.")

    args = parser.parse_args()

    run_fixed_bs_task(
        batch_size=args.bs,
        I=args.I,
        F=args.F,
        Td=args.Td,
        dataset_base=args.dataset,
        latent_d=args.latent_d,
        num_embeddings=args.num_embeddings,
        mod=args.mod,
        snr_db=args.snr_db,
        t_slot_s=args.t_slot_s,
        bandwidth_hz=args.bandwidth_hz,
        tn_m=args.tn_m,
        batches_per_domain=args.batches_per_domain,
        img_metric=args.img_metric,
        ch_metric=args.ch_metric,
        thresh_img=args.thresh_img,
        thresh_ch=args.thresh_ch,
        seed=args.seed,
        ckpt=args.ckpt,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
