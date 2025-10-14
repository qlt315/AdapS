# env_adaps.py
# Gym-style environment for DRL-based batch-size scheduling (AdapS §4.3/§4.4)
# - State  s_b  = [ΔI_b, F, ΔT_d_b, shift_prev]  (normalized)
# - Action a_b  ∈ B̃ (powers-of-two batch sizes)
# - Reward r_b  = (v_b/100) - κ_r * over,  over = max(0, T_a - Td * (bs/I))
# - Inference: JSCC model + modulation (PSK/QAM)
# - Latency (batch-blocking):
#     * Must generate bs images first, then process the whole batch once.
#     * T_a (avg per-image latency in this batch) = bs*g + T_e + T_w + T_d,  g=1/F
# - Wireless T_w: per-timeslot fading; AWGN vs. Rayleigh/Rician are different.
# - Shift detection matches the standalone detector (frozen probe for image stats).
# - TTA (TTNorm) policy: deferred apply (infer with old stats, then fuse for next batch).

from __future__ import annotations

import argparse
import math
import random
import time
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
from copy import deepcopy

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
                     channel: str, ch_args: Dict) -> object:
    return (QAM if mod.lower()=="qam" else PSK)(
        num_embeddings, snr_db, channel, ch_args
    )

def _domain_sequence(batches_per_domain: int,
                     sim_len: int,
                     rng: random.Random) -> List[Tuple[str, str]]:
    """Per-batch domain sequence: start with source, then shifts; each domain lasts 'batches_per_domain' steps."""
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


# ------------------------------ Environment ------------------------------ #

class AdapSEnv(gym.Env):
    """
    DRL environment for batch-size scheduling per AdapS §4.3/§4.4.
    One episode = one task (I, F, T_d). The agent chooses batch size at each step.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 # --- Task sampling (defaults; can be overridden later) ---
                 I_min: int = 8192, I_max: int = 20000,
                 F_min: int = 60,   F_max: int = 120,
                 Td_min: float = 170, Td_max: float = 180,   # seconds
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
                 t_slot_s:   float = 1e-3,   # timeslot length (s)
                 bandwidth_hz: float = 2e6, # rho [Hz]
                 tx_power_w: float = 0.1,    # p_tx [W]
                 noise_psd_w_per_hz: float = 10**(-174/10) * 1e-3,  # -174 dBm/Hz -> W/Hz
                 # --- Reward weight ---
                 kappa_r: float = 1,
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

        # Observation: [ΔI/I_max, F/F_max, ΔT_d/Td_max, shift_prev]
        self._obs_low  = np.zeros(4, dtype=np.float32)
        self._obs_high = np.ones (4, dtype=np.float32)
        self.observation_space = spaces.Box(low=self._obs_low, high=self._obs_high, dtype=np.float32)

        # -------- Adjustable task ranges (kept on self) --------
        self.I_min, self.I_max = I_min, I_max
        self.F_min, self.F_max = F_min, F_max
        self.Td_min, self.Td_max = Td_min, Td_max

        # Model / modulation defaults (kept on self)
        self.dataset_base = dataset_base
        self.in_channels  = in_channels
        self.latent_d     = latent_d
        self.num_embeddings = num_embeddings
        self.num_classes  = _infer_num_classes(dataset_base)
        self.mod_kind     = mod.lower()
        self.channel_default = channel_default
        self.snr_db       = snr_db

        # Shift detection config kept on self (exactly like detection script)
        self.shift_conf = {
            'img_metric': 'kl',
            'ch_metric':  'wasserstein',
            'thresh_img': 0.8,
            'thresh_ch':  0.1,
            'num_embeddings': self.num_embeddings,
            'snr': self.snr_db,
            'batches_per_domain': 8,
            'ch_est_var': 0.0,
            'tn_m': 10.0,
        }
        if shift_conf:
            self.shift_conf.update(shift_conf)

        # Reward weight kept on self
        self.kappa_r = kappa_r

        # ---------- Wireless link parameters kept on self ----------
        self.tc   = t_slot_s
        self.rho  = bandwidth_hz
        self.p_tx = tx_power_w
        self.N0   = noise_psd_w_per_hz
        # -----------------------------------------------------------

        # Build model (weights will be loaded via load_checkpoint)
        self.model = _build_model(self.dataset_base, self.in_channels,
                                  self.latent_d, self.num_classes, self.num_embeddings).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._tta_enabled = False

        # -------- frozen image-stats probe (deep copy of prep[0]) --------
        self.stem_probe = deepcopy(self.model.prep[0]).to(self.device).eval()
        for p in self.stem_probe.parameters():
            p.requires_grad_(False)

        # DataLoaders for domains (we only need .dataset; bs=1, workers=0 is fine)
        self._loaders: Dict[str, object] = {}
        self._ensure_domain_loaders()

        # Internal episode state
        self._reset_ep_state()

    # ---------------------------- Public helpers on instance ---------------------------- #

    def load_checkpoint(self, model_ckpt_path: str):
        """Load JSCC weights and refresh the frozen stem probe."""
        if not model_ckpt_path:
            print("[Env] WARNING: No checkpoint provided. Model uses random weights.")
            return
        try:
            ckpt = torch.load(model_ckpt_path, map_location='cpu', weights_only=True)
        except TypeError:
            warnings.warn("torch.load(weights_only=True) not available; falling back to default.", RuntimeWarning)
            ckpt = torch.load(model_ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt["model_states"])
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.stem_probe = deepcopy(self.model.prep[0]).to(self.device).eval()
        for p in self.stem_probe.parameters():
            p.requires_grad_(False)
        print(f"[Env] Loaded JSCC weights from '{model_ckpt_path}'.")

    def run_random_rollouts(self, episodes: int, base_seed: int, verbose: bool = True):
        """Run random-action episodes and print per-step logs + episode summaries."""
        ep_returns = []
        for ep in range(episodes):
            # Gym v0 reset: use seed() then reset()
            self.seed(base_seed)
            obs = self.reset()
            done = False
            total_r = 0.0
            step_idx = 0
            if verbose:
                print(f"\n[Episode {ep+1}/{episodes}] Task: I={self.I}, F={self.F}, Td={self.Td:.3f}s | TTA={self._tta_enabled}")

            while not done:
                a = self.action_space.sample()
                obs, r, done, step_info = self.step(a)
                total_r += r
                step_idx += 1

                print(
                    f"  step={step_idx:03d}  a_idx={a:<2d}  bs={step_info['batch_size']:<4d}  "
                    f"domain={step_info['domain']:<20s}  "
                    f"img_score={step_info['w2_img_or_kl_img']:.5f}(th={self.shift_conf['thresh_img']})  "
                    f"ch_score={step_info['kl_or_w2_ch']:.6f}(th={self.shift_conf['thresh_ch']})  "
                    f"det_any={step_info['det_any']}  "
                    f"T_g={step_info['latency']['T_g']:.4f}s  T_e={step_info['latency']['T_e']:.4f}s  "
                    f"T_w={step_info['latency']['T_w']:.4f}s  T_d={step_info['latency']['T_d']:.4f}s  "
                    f"T_a={step_info['latency']['T_a']:.4f}s  "
                    f"acc1={step_info['acc1']:.2f}%  reward={step_info['reward']:.4f}  "
                    f"remain_I={step_info['remaining_I']:<5d}"
                )

            # Weighted-by-samples mean accuracy for the task
            mean_acc = 100.0 * (self.correct_total / max(1, self.sample_total))
            det_acc = (self.det_correct / max(1, self.det_total)) * 100.0
            print(f"[Episode {ep+1}] return={total_r:.4f}  steps={step_idx}  "
                  f"total_latency={self.elapsed_T:.4f}s  mean_acc={mean_acc:.2f}%  "
                  f"det_acc={det_acc:.2f}% ({self.det_correct}/{self.det_total})")

            ep_returns.append(total_r)

        mean_ret = sum(ep_returns) / max(1, len(ep_returns))
        print(f"\n== Random-policy summary ==  episodes={episodes}  mean_return={mean_ret:.4f}")
        return ep_returns

    # ---------------------------- Gym API (instance methods, Gym v0) ---------------------------- #

    def reset(self):
        """
        First call: sample (I, F, Td) and domain_seq; store on self (fixed for this env instance).
        Later calls: reuse the same task/domain_seq so every episode is identical (seed controls it).
        """
        if not hasattr(self, "I") or not hasattr(self, "F") or not hasattr(self, "Td") or not hasattr(self, "domain_seq"):
            # Sample a new task (I, F, T_d)
            self.I = int(self.rng.randint(self.I_min, self.I_max))
            self.F = int(self.rng.randint(self.F_min, self.F_max))
            self.Td = float(self.rng.uniform(self.Td_min, self.Td_max))
            # Steps upper-bound
            self.ep_max_steps = max(1, math.ceil(self.I / min(self.batch_sizes)))
            # Domain sequence for this task (fixed across episodes)
            self.domain_seq = _domain_sequence(self.shift_conf['batches_per_domain'],
                                               self.ep_max_steps, self.rng)
        else:
            self.ep_max_steps = max(1, math.ceil(self.I / min(self.batch_sizes)))

        # Reset per-episode stats
        self._reset_ep_state()
        return self._build_obs()  # Gym v0: only obs

    def step(self, action: int):
        assert self._step_idx < self.ep_max_steps, "Episode already done, call reset()."
        bs = self.batch_sizes[int(action)]
        bs = min(bs, self.delta_I)  # clip by remaining images

        # Current/previous domain (for GT shift)
        dataset_name, channel_type = self.domain_seq[self._step_idx]
        if self._step_idx == 0:
            prev_dataset, prev_channel = dataset_name, channel_type
        else:
            prev_dataset, prev_channel = self.domain_seq[self._step_idx - 1]
        gt_any_shift = ((dataset_name != prev_dataset) or (channel_type != prev_channel))

        # Sample batch
        imgs, labels = sample_random_batch(self._loaders[dataset_name], bs=bs)
        imgs   = imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Channel args + modulators (inference vs detection)
        ch_args = {}
        if channel_type == 'rician': ch_args['K'] = 3.0
        elif channel_type == 'nakagami': ch_args['m'] = 2.0  # (unused here but kept for API symmetry)

        # Inference mod (respect config)
        mod = _build_modulator(self.mod_kind, self.num_embeddings, self.snr_db, channel_type, ch_args)

        # Detection mod: force QAM and match detector code path
        mod_det = QAM(self.shift_conf['num_embeddings'],
                      self.shift_conf['snr'],
                      channel_type=channel_type,
                      channel_args=ch_args)

        # ===================== Detection (aligned with detection script) ===================== #
        with torch.no_grad():
            stem_feat = self.stem_probe(imgs)
            cur_img_mean = stem_feat.mean(dim=(0,2,3)).cpu().numpy()
            cur_img_std  = stem_feat.std (dim=(0,2,3)).cpu().numpy()

        with torch.no_grad():
            z_enc  = self.model.encoder(imgs)
            z_flat = z_enc.view(z_enc.size(0), -1)

        # Primary path: like the detection script
        mod_det.channel_estimates = []
        mod_det.apply_channel(z_flat)

        # Fallback path: if estimates not collected, use low-level channel.apply to fetch h_true
        if len(getattr(mod_det, "channel_estimates", [])) > 0:
            est = torch.cat(mod_det.channel_estimates, dim=0)
        else:
            _, h_true = mod_det.channel.apply(z_flat, return_h=True)
            est = h_true

        h_mag = torch.norm(est, dim=1).cpu().numpy()
        std_eps = h_mag.std() + 1e-6
        h_norm = (h_mag - h_mag.mean()) / std_eps
        cur_ch_hist, _ = np.histogram(h_norm, bins=50, range=(-5,5), density=True)

        if self._step_idx == 0:
            self._ref_img_stats = (cur_img_mean, cur_img_std)
            self._ref_ch_hist   = cur_ch_hist
            det_any = False
            img_score = ch_score = 0.0
        else:
            img_metric = self.shift_conf.get("img_metric", "kl").lower()
            if img_metric == "kl":
                img_score = _sym_kl_diag_gauss(cur_img_mean, cur_img_std,
                                               self._ref_img_stats[0], self._ref_img_stats[1])
            elif img_metric == "wasserstein":
                img_score = _w2_gaussian(cur_img_mean, cur_img_std,
                                         self._ref_img_stats[0], self._ref_img_stats[1])
            else:
                raise ValueError(f"Unsupported image metric: {img_metric}")
            det_img = (img_score > float(self.shift_conf['thresh_img']))

            ch_metric = self.shift_conf.get("ch_metric", "wasserstein").lower()
            if ch_metric == "wasserstein":
                from scipy.stats import wasserstein_distance
                ch_score = float(wasserstein_distance(cur_ch_hist, self._ref_ch_hist))
            elif ch_metric == "kl":
                from scipy.stats import entropy
                eps = 1e-8
                ch_score = float(entropy(cur_ch_hist + eps, self._ref_ch_hist + eps))
            elif ch_metric == "l2":
                ch_score = float(np.linalg.norm(cur_ch_hist - self._ref_ch_hist))
            else:
                raise ValueError(f"Unsupported ch_metric: {ch_metric}")
            det_ch = (ch_score > float(self.shift_conf['thresh_ch']))

            det_any = bool(det_img or det_ch)

            self._ref_img_stats = (cur_img_mean, cur_img_std)
            self._ref_ch_hist   = cur_ch_hist

        # ===================== Inference (current batch uses OLD stats) ===================== #
        if self._use_cuda: torch.cuda.synchronize(self.device)

        if (self._step_idx > 0) and (not self._tta_enabled) and det_any:
            replace_bn_with_ttn(self.model, m=float(self.shift_conf.get('tn_m', 10.0)))
            set_ttn_mode(self.model, "infer_old")
            self._tta_enabled = True

        en0 = time.perf_counter()
        with torch.no_grad():
            if self._tta_enabled:
                set_ttn_mode(self.model, "infer_old")
            feat, shp = self.model.encode(imgs)
        if self._use_cuda: torch.cuda.synchronize(self.device)
        en1 = time.perf_counter()

        with torch.no_grad():
            feat_noisy, _ = self.model.sampler(feat, mod=mod)

        de0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model.decode(feat_noisy, shp)
        if self._use_cuda: torch.cuda.synchronize(self.device)
        de1 = time.perf_counter()

        # --- measured batch processing times ---
        T_e = en1 - en0  # encoder (batch)
        T_d = de1 - de0  # decoder (batch)
        T_w = self.obtain_tx_latency(feat, channel_type, ch_args)  # wireless (batch, per-slot fading)

        # === Batch-blocking latency model ===
        g = 1.0 / float(self.F)               # generation time per image
        T_a = bs * g + T_e + T_w + T_d        # avg per-image latency in this batch == batch makespan

        # ====================================

        top1, _ = accuracy(logits, labels, (1, 3))
        v_b = float(top1.item())

        # Reward: accuracy minus latency budget violation (deadline share for this batch)
        over = max(0.0, T_a - (self.Td * (bs / max(1, self.I))))
        r = (v_b / 100.0) * (bs / self.I) - self.kappa_r * over

        # ===================== Adapt phase (write stats for NEXT batch) ===================== #
        if self._tta_enabled:
            set_ttn_mode(self.model, "adapt")
            with torch.no_grad():
                _ = self.model.encode(imgs)
            set_ttn_mode(self.model, "infer_old")

        # Accounting
        self.delta_I -= bs
        self.elapsed_T += T_a

        # Weighted accuracy accumulation (by sample count)
        correct_this = (v_b / 100.0) * bs
        self.correct_total += correct_this
        self.sample_total  += bs
        self.acc_sum += v_b
        self.acc_count += 1

        self._step_idx += 1

        if self._step_idx >= 1:
            self.det_total += 1
            self.det_correct += int(det_any == bool(gt_any_shift))

        obs = self._build_obs(shift_prev=(1.0 if det_any else 0.0))

        # Termination: ensure we always finish the task
        terminated = (self.delta_I <= 0)
        truncated  = (self._step_idx >= self.ep_max_steps) and not terminated
        done = bool(terminated or truncated)

        info = {
            "batch_size": bs,
            "domain": f"{dataset_name}/{channel_type}",
            "w2_img_or_kl_img": img_score,
            "kl_or_w2_ch": ch_score,
            "det_any": int(det_any),
            "latency": {
                # Keep legacy fields for logs:
                "T_g": bs * g,   # total generation time of this batch
                "T_e": T_e,
                "T_w": T_w,
                "T_d": T_d,
                # NEW: T_a is the per-image average latency under batch-blocking (== batch makespan)
                "T_a": T_a
            },
            "acc1": v_b,
            "reward": float(r),
            "remaining_I": self.delta_I,
        }
        return obs, float(r), done, info

    def render(self):
        return None

    # ------------------------ Internal methods ------------------------ #

    def _reset_ep_state(self):
        self.delta_I: int = 0
        self.elapsed_T: float = 0.0
        self.acc_sum: float = 0.0
        self.acc_count: int = 0
        # weighted-accuracy accumulators
        self.correct_total: float = 0.0
        self.sample_total: int = 0
        self._step_idx: int = 0
        self._ref_img_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._ref_ch_hist: Optional[np.ndarray] = None
        self._tta_enabled: bool = False
        self.det_correct: int = 0
        self.det_total: int = 0

    def _ensure_domain_loaders(self):
        """Prepare DataLoaders for all possible domains (we only need .dataset; bs=1, workers=0 is fine)."""
        domains = {
            'CIFAR10', 'CIFAR10_noise', 'CIFAR10_fog', 'CIFAR10_bright',
            'CIFAR10_snow', 'CIFAR10_rain', 'CIFAR10_blur',
        }
        for ds in domains:
            self._loaders[ds] = get_data(ds, batch_size=1, n_worker=0, train=False)

    def _build_obs(self, shift_prev: Optional[float]=None) -> np.ndarray:
        if self._step_idx == 0 and self.delta_I == 0:
            self.delta_I = self.I
            self.elapsed_T = 0.0
            self.acc_sum = 0.0
            self.acc_count = 0
            self.correct_total = 0.0
            self.sample_total  = 0
        shift_prev = (0.0 if shift_prev is None else float(shift_prev))
        obs = np.array([
            float(self.delta_I) / float(self.I_max),
            float(self.F)       / float(self.F_max),
            max(0.0, self.Td - self.elapsed_T) / float(self.Td_max),
            shift_prev
        ], dtype=np.float32)
        return obs

    # ---------- wireless latency with per-timeslot fading ---------- #
    def obtain_tx_latency(self, latent_feat: torch.Tensor, channel_type: str, ch_args: Dict) -> float:
        """
        Wireless latency estimate with per-timeslot fading (SNR aligned to snr_db):
        - Payload bits L_b = (#symbols per img) * bs * log2(U)
        - Each slot i has |h|^2_i drawn i.i.d. by channel_type.
          AWGN: |h|^2 = 1; Rayleigh: Exp(1); Rician(K): normalized (E|h|^2=1) approx.
        - Capacity per slot: C_i = rho * log2(1 + SNR0 * |h|^2_i), where SNR0 = 10**(snr_db/10).
        - Keep drawing slots until accumulated bits >= L_b; T_w = n_slots * tc.
        """
        bs = latent_feat.size(0)
        z = latent_feat.view(bs, -1)
        symbols_per_img = z.size(1)
        payload_bits = symbols_per_img * bs * math.log2(self.num_embeddings)

        # Use task SNR everywhere (linear), aligned with self.snr_db (in dB).
        snr0 = 10 ** (self.snr_db / 10.0)

        # Helper to draw |h|^2 samples (unit mean for fading channels)
        def draw_h2(n: int) -> np.ndarray:
            if channel_type.lower() == 'awgn':
                return np.ones(n, dtype=np.float64)  # no fading
            if channel_type.lower() == 'rayleigh':
                # |h|^2 ~ Exp(1) (unit-mean)
                return np.random.exponential(scale=1.0, size=n)
            if channel_type.lower() == 'rician':
                # Rician with K (power ratio of LOS to NLOS); normalized so E|h|^2=1
                K = float(ch_args.get('K', 3.0))
                mu = math.sqrt(K / (K + 1.0))
                sigma = math.sqrt(1.0 / (2.0 * (K + 1.0)))  # per real component
                re = np.random.normal(loc=mu, scale=sigma, size=n)
                im = np.random.normal(loc=0.0, scale=sigma, size=n)
                return re*re + im*im
            # default fallback
            return np.ones(n, dtype=np.float64)

        # Accumulate bits slot-by-slot (integer slots; keep original discretization)
        bits_acc = 0.0
        slots = 0
        max_slots = int(1e9)  # safety cap
        while bits_acc < payload_bits and slots < max_slots:
            h2 = draw_h2(1)[0]
            cap = self.rho * math.log2(1.0 + snr0 * h2)  # bits/sec
            if cap <= 0:
                cap = 1e3  # 1 kbps floor to avoid zero-cap
            bits_acc += cap * self.tc
            slots += 1
        return slots * self.tc

    def seed(self, seed: Optional[int] = None):
        """Legacy Gym API: set RNG seeds and return [seed]. New seed => forget current fixed task."""
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # Forget the current fixed task so next reset() will sample a new one
            for attr in ("I", "F", "Td", "domain_seq", "ep_max_steps"):
                if hasattr(self, attr):
                    delattr(self, attr)
        return [seed]


# ------------------------------------ main ------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="AdapS env demo (minimal CLI).")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for env.reset().")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="JT-JSCC/trained_models/cifar10_best.pt",
        help="Path to JSCC checkpoint (.pt). If omitted, random weights are used."
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-step logs.")
    args = parser.parse_args()

    env = AdapSEnv()
    if args.ckpt:
        env.load_checkpoint(args.ckpt)
    env.run_random_rollouts(episodes=args.episodes, base_seed=args.seed, verbose=args.verbose)


if __name__ == "__main__":
    main()
