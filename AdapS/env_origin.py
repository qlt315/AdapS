# Gym-style environment for DRL-based batch-size scheduling (AdapS §4.3/§4.4)
# - State  s_b  = [ΔI_b, F, ΔT_d_b, shift_prev]  (normalized)
# - Action a_b  ∈ B̃ (powers-of-two batch sizes)
# - Reward r_b  = v_b  - κ_r * ( (T_d * (bs / I)) - T_a_b )
# - Inference: JSCC model + modulation (PSK/QAM)
# - Latency:   T_a_b = T_g + T_e + T_w + T_d  (T_e/T_d measured; T_w via Shannon using |ĥ|^2)
# - Shift detection (in-env): image shift via stem-feature W2; channel shift via KL on |ĥ| histogram.
#
# Run:
#   python env_adaps.py --ckpt <path_to_jscc_ckpt> --episodes 3 --verbose [--ch_est_var 0.01]
#
# Notes:
#   * Completely self-contained (no external shift_detec.py needed).
#   * No TTA: model is frozen, eval-only.

from __future__ import annotations

import argparse
import math
import random
import time
import numpy as np
from typing import Dict, Tuple, List, Optional

# Gym / Gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces

import torch
import torch.nn as nn

# Project imports (adjust to your layout)
from JSCC.datasets.dataloader import get_data
from JSCC.utils.modulation import QAM, PSK
from JSCC.utils.accuracy import accuracy
import model.DT_JSCC as JSCC_model


# ------------------------- Local data sampling ------------------------- #

def sample_random_batch(loader, bs: int):
    """
    Sample a random batch of size `bs` directly from loader.dataset
    without advancing the loader iterator. Works for datasets that
    return (tensor_image [C,H,W] in [0,1], int_label).
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
                     channel: str, ch_args: Dict) -> object:
    return (QAM if mod.lower()=="qam" else PSK)(
        num_embeddings, snr_db, channel, ch_args
    )

def _domain_sequence(batches_per_domain: int,
                     sim_len: int,
                     rng: random.Random) -> List[Tuple[str, str]]:
    """Build per-batch domain sequence: start with source, then random shifts."""
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

# ---------- tiny stats/metrics (image W2 & channel KL) ----------

def _stem_stats(model, imgs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std over channels in the stem feature space: [B,C,H,W] -> (C,), (C,)"""
    with torch.no_grad():
        stem = model.prep[0](imgs)  # first stem block, consistent with detector space
        mean = stem.mean(dim=(0,2,3)).cpu().numpy()
        std  = stem.std (dim=(0,2,3)).cpu().numpy()
    return mean, std

def _w2_gaussian(mean1: np.ndarray, std1: np.ndarray,
                 mean2: np.ndarray, std2: np.ndarray) -> float:
    """Channel-wise 2-Wasserstein distance between diagonal Gaussians."""
    m1 = torch.from_numpy(mean1)
    s1 = torch.from_numpy(std1)
    m2 = torch.from_numpy(mean2)
    s2 = torch.from_numpy(std2)
    w2_sq = ((m1 - m2)**2 + (s1 - s2)**2).sum()
    return float(torch.sqrt(w2_sq).item())

def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL(p||q) for histograms (both already normalized)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

def _hist_from_hhat(h_hat_cpu: torch.Tensor) -> np.ndarray:
    """Normalized histogram from |ĥ| magnitudes."""
    if h_hat_cpu.numel() == 0:
        return np.ones(50, dtype=np.float32) / 50.0
    with torch.no_grad():
        h_mag = torch.norm(h_hat_cpu, dim=1).cpu().numpy()
        h_norm = (h_mag - h_mag.mean()) / (h_mag.std() + 1e-6)
        hist, _ = np.histogram(h_norm, bins=50, range=(-5, 5), density=True)
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s <= 0:
        return np.ones(50, dtype=np.float32) / 50.0
    return hist / s


# ------------------------------ Environment ------------------------------ #

class AdapSBatchSizeEnv(gym.Env):
    """
    DRL environment for batch-size scheduling per AdapS §4.3/§4.4.
    One episode = one task (I, F, T_d). At each step, the agent chooses the batch size.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 # --- Task sampling (per-episode) ---
                 I_min: int = 1024, I_max: int = 2048,
                 F_min: int = 5,    F_max: int = 30,
                 Td_min: float = 2.0, Td_max: float = 5.0,   # seconds
                 # --- Model / data ---
                 dataset_base: str = "CIFAR10",
                 in_channels: int = 3, latent_d: int = 512,
                 num_embeddings: int = 16,
                 mod: str = "psk", channel_default: str = "rayleigh",
                 snr_db: float = 8.0,
                 # --- Shift detection config ---
                 shift_conf: Optional[Dict] = None,
                 # --- Batch-size action set ---
                 action_batch_sizes: Optional[List[int]] = None,
                 # --- Wireless latency model params (T_w only) ---
                 t_slot_s:   float = 1e-3,   # time-slot length tc [s]
                 bandwidth_hz: float = 10e6, # rho [Hz]
                 tx_power_w: float = 0.1,    # p [W]
                 noise_psd_w_per_hz: float = 10**(-174/10) * 1e-3,  # -174 dBm/Hz -> W/Hz
                 # --- Reward weight ---
                 kappa_r: float = 1.0,
                 # --- Device / seed ---
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None
                 ):
        super().__init__()

        # RNG
        self.rng = random.Random(seed)

        # Device
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._use_cuda = torch.cuda.is_available() and self.device.type == 'cuda'

        # Action set (powers of two by default)
        self.batch_sizes = action_batch_sizes or [2,4,8,16,32,64,128,256]
        self.action_space = spaces.Discrete(len(self.batch_sizes))

        # Observation: [ΔI/I_max, F/F_max, ΔT_d/Td_max, shift_prev] ∈ [0,1]^4
        self._obs_low  = np.zeros(4, dtype=np.float32)
        self._obs_high = np.ones (4, dtype=np.float32)
        self.observation_space = spaces.Box(low=self._obs_low, high=self._obs_high, dtype=np.float32)

        # Task sampling ranges
        self.I_min, self.I_max = I_min, I_max
        self.F_min, self.F_max = F_min, F_max
        self.Td_min, self.Td_max = Td_min, Td_max

        # Model / modulation defaults
        self.dataset_base = dataset_base
        self.in_channels  = in_channels
        self.latent_d     = latent_d
        self.num_embeddings = num_embeddings
        self.num_classes  = _infer_num_classes(dataset_base)
        self.mod_kind     = mod.lower()
        self.channel_default = channel_default
        self.snr_db       = snr_db

        # Shift detection config
        self.shift_conf = {
            'img_metric': 'wasserstein',
            'ch_metric': 'kl',
            'thresh_img': 0.05,
            'thresh_ch':  0.05,
            'num_embeddings': self.num_embeddings,
            'snr': self.snr_db,
            'batches_per_domain': 3,
            # channel-estimation variance for pilot-style estimator
            'ch_est_var': 0.0,
        }
        if shift_conf:
            self.shift_conf.update(shift_conf)

        # Latency model (wireless)
        self.tc   = t_slot_s
        self.rho  = bandwidth_hz
        self.p_tx = tx_power_w
        self.N0   = noise_psd_w_per_hz

        # Reward
        self.kappa_r = kappa_r

        # Build model (weights loaded via make_env_and_load); freeze (no TTA)
        self.model = _build_model(self.dataset_base, self.in_channels,
                                  self.latent_d, self.num_classes, self.num_embeddings).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        torch.set_grad_enabled(False)  # global safety: eval-only
        self._tta_enabled = False  # explicit flag for clarity

        # DataLoaders for domains (we only need .dataset; bs=1, workers=0 is fine)
        self._loaders: Dict[str, object] = {}
        self._ensure_domain_loaders()

        # Internal episode state
        self._reset_ep_state()

    # ---------------------------- Gym API ---------------------------- #

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Sample a new task (I, F, T_d) uniformly
        self.I  = int(self.rng.randint(self.I_min, self.I_max))
        self.F  = int(self.rng.randint(self.F_min, self.F_max))
        self.Td = float(self.rng.uniform(self.Td_min, self.Td_max))

        # Upper bound on #steps for the episode
        self.ep_max_steps = max(1, math.ceil(self.I / min(self.batch_sizes)))

        # Domain sequence for this episode
        self.domain_seq = _domain_sequence(self.shift_conf['batches_per_domain'],
                                           self.ep_max_steps, self.rng)

        # Reset per-episode stats
        self._reset_ep_state()

        obs = self._build_obs()
        info = {"I": self.I, "F": self.F, "Td": self.Td, "tta": self._tta_enabled}
        return obs, info

    def step(self, action: int):
        assert self._step_idx < self.ep_max_steps, "Episode already done, call reset()."
        bs = self.batch_sizes[int(action)]

        # Clip bs by remaining images
        bs = min(bs, self.delta_I)

        # Current/previous domain (for ground-truth shift)
        dataset_name, channel_type = self.domain_seq[self._step_idx]
        if self._step_idx == 0:
            prev_dataset, prev_channel = dataset_name, channel_type
        else:
            prev_dataset, prev_channel = self.domain_seq[self._step_idx - 1]
        gt_img_shift = (dataset_name != prev_dataset)
        gt_ch_shift  = (channel_type != prev_channel)
        gt_any_shift = (gt_img_shift or gt_ch_shift)

        # Sample a batch
        imgs, labels = sample_random_batch(self._loaders[dataset_name], bs=bs)
        imgs   = imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Channel args
        ch_args = {}
        if channel_type == 'rician': ch_args['K'] = 3.0
        elif channel_type == 'nakagami': ch_args['m'] = 2.0

        mod = _build_modulator(self.mod_kind, self.num_embeddings, self.snr_db, channel_type, ch_args)

        # === Local pilot-style channel estimate (with optional error) ===
        feat_probe, _ = self.model.encode(imgs)            # (B, C, H, W)
        z_flat = feat_probe.view(feat_probe.size(0), -1)   # (B, N)
        h_hat_cpu, _ = self._estimate_channel_from_latent(z_flat, mod,
                                                          est_var=float(self.shift_conf.get('ch_est_var', 0.0)))
        # Build current channel histogram
        cur_ch_hist = _hist_from_hhat(h_hat_cpu)

        # === Image & Channel shift detection (in-env) ===
        cur_img_mean, cur_img_std = _stem_stats(self.model, imgs)
        if self._ref_img_stats is None:
            self._ref_img_stats = (cur_img_mean, cur_img_std)
        if self._ref_ch_hist is None:
            self._ref_ch_hist = cur_ch_hist

        # W2 in stem feature space
        w2_img = _w2_gaussian(cur_img_mean, cur_img_std,
                              self._ref_img_stats[0], self._ref_img_stats[1])
        # KL between channel histograms
        kl_ch  = _kl_div(cur_ch_hist, self._ref_ch_hist)

        det_img = (w2_img > float(self.shift_conf['thresh_img']))
        det_ch  = (kl_ch  > float(self.shift_conf['thresh_ch']))
        det_any = bool(det_img or det_ch)

        ok_img = (det_img == gt_img_shift)
        ok_ch  = (det_ch  == gt_ch_shift)
        ok_any = (det_any == gt_any_shift)

        # Update references for next batch
        self._ref_img_stats = (cur_img_mean, cur_img_std)
        self._ref_ch_hist   = cur_ch_hist

        # ---------------- Inference & timing (measured) ----------------
        if self._use_cuda: torch.cuda.synchronize(self.device)

        # Encode
        en0 = time.perf_counter()
        feat, shp = self.model.encode(imgs)          # includes prep
        if self._use_cuda: torch.cuda.synchronize(self.device)
        en1 = time.perf_counter()

        # Pass through channel inside the model (for actual inference)
        feat_noisy, _ = self.model.sampler(feat, mod=mod)

        # Decode
        de0 = time.perf_counter()
        logits = self.model.decode(feat_noisy, shp)
        if self._use_cuda: torch.cuda.synchronize(self.device)
        de1 = time.perf_counter()

        T_e = en1 - en0                 # measured encoder time (s)
        T_d = de1 - de0                 # measured decoder time (s)
        T_g = bs / float(self.F)        # synthetic generation time (s); set to 0 if no generator
        T_w = self._tx_latency_seconds(feat, mod)  # wireless latency estimate (s)
        T_a_b = T_g + T_e + T_w + T_d

        # Accuracy (top-1)
        top1, _ = accuracy(logits, labels, (1, 3))
        v_b = float(top1.item())  # percentage [0,100]

        # Reward
        r = v_b - self.kappa_r * ((self.Td * (bs / self.I)) - T_a_b)

        # Bookkeeping
        self.delta_I -= bs
        self.elapsed_T += T_a_b
        self.acc_sum += v_b
        self.acc_count += 1
        self._step_idx += 1

        # Next observation
        obs = self._build_obs(shift_prev=(1.0 if det_any else 0.0))

        # Termination
        terminated = (self.delta_I <= 0) or (self.elapsed_T >= self.Td)
        truncated  = (self._step_idx >= self.ep_max_steps) and not terminated

        info = {
            "batch_size": bs,
            "dataset": dataset_name,
            "channel": channel_type,
            "acc1": v_b,
            "latency": {"T_g": T_g, "T_e": T_e, "T_w": T_w, "T_d": T_d, "T_a": T_a_b},
            "elapsed_T": self.elapsed_T,
            "remaining_I": self.delta_I,
            # detection outputs
            "det_img": bool(det_img),
            "det_ch":  bool(det_ch),
            "det_any": bool(det_any),
            "gt_img_shift": bool(gt_img_shift),
            "gt_ch_shift":  bool(gt_ch_shift),
            "gt_any_shift": bool(gt_any_shift),
            "ok_img": bool(ok_img),
            "ok_ch":  bool(ok_ch),
            "ok_any": bool(ok_any),
            # scores (optional for debugging)
            "w2_img": w2_img,
            "kl_ch":  kl_ch,
        }
        return obs, float(r), bool(terminated), bool(truncated), info

    def render(self):
        return None

    # ------------------------ Internal methods ------------------------ #

    def _reset_ep_state(self):
        self.delta_I: int = 0
        self.elapsed_T: float = 0.0
        self.acc_sum: float = 0.0
        self.acc_count: int = 0
        self._step_idx: int = 0
        # references for shift detection
        self._ref_img_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._ref_ch_hist: Optional[np.ndarray] = None

    def _ensure_domain_loaders(self):
        """Prepare DataLoaders for all possible domains (we only need .dataset; bs=1, workers=0 is fine)."""
        domains = {
            'CIFAR10', 'CIFAR10_noise', 'CIFAR10_fog', 'CIFAR10_bright',
            'CIFAR10_snow', 'CIFAR10_rain', 'CIFAR10_blur',
        }
        for ds in domains:
            self._loaders[ds] = get_data(ds, batch_size=1, n_worker=0, train=False)

    def _build_obs(self, shift_prev: Optional[float]=None) -> np.ndarray:
        # Initialize ΔI and elapsed_T on first build
        if self._step_idx == 0 and self.delta_I == 0:
            self.delta_I = self.I
            self.elapsed_T = 0.0
            self.acc_sum = 0.0
            self.acc_count = 0

        # shift_prev default = 0 at episode start
        shift_prev = (0.0 if shift_prev is None else float(shift_prev))

        # Normalized observation
        obs = np.array([
            float(self.delta_I) / float(self.I_max),
            float(self.F)       / float(self.F_max),
            max(0.0, self.Td - self.elapsed_T) / float(self.Td_max),
            shift_prev
        ], dtype=np.float32)
        return obs

    def _estimate_channel_from_latent(self, z_flat: torch.Tensor, mod, est_var: float = 0.0):
        """
        Pilot-style channel estimation from latent symbols:
            y, h_true = channel(z)
            ĥ = h_true + ε, with ε ~ N(0, est_var)
            y_eq = y / (ĥ + eps)
        Returns:
            h_hat_cpu: (B, 2) CPU tensor for histogram
            y_eq     : equalized symbols (unused by caller)
        """
        eps = 1e-6
        y, h_true = mod.channel.apply(z_flat, return_h=True)
        if est_var > 0.0:
            noise = torch.randn_like(h_true) * math.sqrt(est_var)
            h_hat = h_true + noise
        else:
            h_hat = h_true
        y_eq = y / (h_hat + eps)
        return h_hat.detach().cpu(), y_eq

    def _tx_latency_seconds(self, latent_feat: torch.Tensor, mod) -> float:
        """
        Wireless latency estimate:
        - Bits L_b = (#symbols per img) * bs * log2(U)
        - Capacity C ≈ ρ * E[ log2(1 + SNR * |h|^2) ]
        - T_w = ceil( L_b / (tc * C) ) * tc
        """
        bs = latent_feat.size(0)
        z = latent_feat.view(bs, -1)
        symbols_per_img = z.size(1)
        payload_bits = symbols_per_img * bs * math.log2(self.num_embeddings)

        # Fresh local estimate for |h|^2 (consistent with detector)
        est_var = float(self.shift_conf.get('ch_est_var', 0.0))
        h_hat_cpu, _ = self._estimate_channel_from_latent(z, mod, est_var=est_var)
        if h_hat_cpu.numel() > 0:
            h2 = torch.norm(h_hat_cpu, dim=1).pow(2).mean().item()
        else:
            h2 = 1.0  # fallback

        # Linear SNR: p*|h|^2 / (N0 * rho)
        snr_linear = (self.p_tx * h2) / (self.N0 * self.rho + 1e-12)
        cap_bps = self.rho * math.log2(1.0 + snr_linear)  # bits per second
        if cap_bps <= 0:
            cap_bps = 1e3  # safety clamp

        bits_per_slot = cap_bps * self.tc
        slots = math.ceil(payload_bits / bits_per_slot)
        return slots * self.tc


# ------------------------------ Convenience ------------------------------ #

def make_env_and_load(model_ckpt_path: str,
                      **env_kwargs) -> AdapSBatchSizeEnv:
    """
    Create the environment and load JSCC weights.
    """
    env = AdapSBatchSizeEnv(**env_kwargs)
    if model_ckpt_path:
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        env.model.load_state_dict(ckpt["model_states"])
        env.model.to(env.device).eval()
        for p in env.model.parameters():
            p.requires_grad_(False)
        print(f"[Env] Loaded JSCC weights from '{model_ckpt_path}'. (TTA: disabled)")
    else:
        print("[Env] WARNING: No checkpoint provided. Model uses random weights. (TTA: disabled)")
    return env


def _run_random_rollouts(env, episodes: int, base_seed: int, verbose: bool):
    """Run random-action episodes and print per-step logs + episode summaries."""
    ep_returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        total_r = 0.0
        step_idx = 0
        if verbose:
            print(f"\n[Episode {ep+1}/{episodes}] Task: I={info['I']}, F={info['F']}, Td={info['Td']:.3f}s | TTA={info['tta']}")

        while not done:
            a = env.action_space.sample()
            obs, r, terminated, truncated, step_info = env.step(a)
            total_r += r
            step_idx += 1
            done = terminated or truncated

            # Per-batch print with detection correctness
            print(
                f"  step={step_idx:03d}  a_idx={a}  bs={step_info['batch_size']:<4d}  "
                f"acc1={step_info['acc1']:.2f}%  "
                f"T_e={step_info['latency']['T_e']:.4f}s  "
                f"T_d={step_info['latency']['T_d']:.4f}s  "
                f"T_w={step_info['latency']['T_w']:.4f}s  "
                f"T_a={step_info['latency']['T_a']:.4f}s  "
                f"remain_I={step_info['remaining_I']:<4d}  "
                f"det(img/ch/any)={int(step_info['det_img'])}/{int(step_info['det_ch'])}/{int(step_info['det_any'])}  "
                f"gt(img/ch/any)={int(step_info['gt_img_shift'])}/{int(step_info['gt_ch_shift'])}/{int(step_info['gt_any_shift'])}  "
                f"ok(img/ch/any)={int(step_info['ok_img'])}/{int(step_info['ok_ch'])}/{int(step_info['ok_any'])}"
            )

        mean_acc = (env.acc_sum / max(1, env.acc_count)) if env.acc_count > 0 else 0.0
        print(f"[Episode {ep+1}] return={total_r:.4f}  steps={step_idx}  "
              f"total_latency={env.elapsed_T:.4f}s  mean_acc={mean_acc:.2f}%")

        ep_returns.append(total_r)

    mean_ret = sum(ep_returns) / max(1, len(ep_returns))
    print(f"\n== Random-policy summary ==  episodes={episodes}  mean_return={mean_ret:.4f}")
    return ep_returns


def main():
    parser = argparse.ArgumentParser(
        description="Random-sampling test for AdapS batch-size scheduling env."
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--ckpt", type=str, default="JT-JSCC/trained_models/cifar10_best.pt", help="Path to JSCC checkpoint (.pt).")
    parser.add_argument("--verbose", action="store_true", help="Print per-step logs.")
    # common env knobs
    parser.add_argument("--dataset_base", type=str, default="CIFAR10")
    parser.add_argument("--mod", type=str, default="psk", choices=["psk", "qam"])
    parser.add_argument("--snr_db", type=float, default=8.0)
    parser.add_argument("--action_set", type=str, default="2,4,8,16,32,64,128,256",
                        help="Comma-separated batch sizes for the discrete action space.")
    parser.add_argument("--I_range", type=str, default="1024,2048", help="min,max")
    parser.add_argument("--F_range", type=str, default="5,30", help="min,max")
    parser.add_argument("--Td_range", type=str, default="2.0,5.0", help="min,max (seconds)")
    parser.add_argument("--ch_est_var", type=float, default=0.0, help="Variance for channel estimate noise (ĥ = h + ε).")
    args = parser.parse_args()

    # parse ranges and action set
    I_min, I_max = map(int, args.I_range.split(","))
    F_min, F_max = map(int, args.F_range.split(","))
    Td_min, Td_max = map(float, args.Td_range.split(","))
    action_batch_sizes = [int(x) for x in args.action_set.split(",")]

    env = make_env_and_load(
        model_ckpt_path=args.ckpt,
        dataset_base=args.dataset_base,
        mod=args.mod,
        snr_db=args.snr_db,
        action_batch_sizes=action_batch_sizes,
        I_min=I_min, I_max=I_max,
        F_min=F_min, F_max=F_max,
        Td_min=Td_min, Td_max=Td_max,
        shift_conf={"ch_est_var": args.ch_est_var},
    )

    _ = _run_random_rollouts(env, args.episodes, args.seed, args.verbose)


if __name__ == "__main__":
    main()
