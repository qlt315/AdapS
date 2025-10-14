import os
import re
import time
import torch
import torch.nn as nn
import torch.optim as optim  # (kept if you later want optimizers)
from typing import Dict

from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy


# ----------------------------- Utilities ---------------------------------- #

_DATASET_BASE_RE = re.compile(r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=re.IGNORECASE)

def infer_num_classes(dataset_name: str) -> int:
    """
    Infer number of classes from dataset string, tolerant to variants like 'CIFAR10_blur'.
    """
    ds = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(ds)
    if not m:
        raise ValueError(f"Unknown dataset for inferring num_classes: {dataset_name}")
    base = m.group(1).lower()
    if base == "cifar100":
        return 100
    if base in ("cifar10", "cinic10"):
        return 10
    raise ValueError(f"Unsupported dataset base: {base}")

def dataset_base_key(dataset_name: str) -> str:
    """
    Return canonical lowercase base key: 'cifar10' / 'cifar100' / 'cinic10'.
    """
    ds = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(ds)
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).lower()

def default_ckpt_filename(dataset_name: str) -> str:
    """
    Build a default checkpoint filename like 'cifar10_best.pt' based on dataset base.
    """
    base = dataset_base_key(dataset_name)
    return f"{base}_best.pt"


# ----------------------- Test-Time Normalization (TTN) --------------------- #

class TTNorm(nn.Module):
    """
    Test-Time Normalization with deferred-apply:
      - Keeps persistent (mu_t, var_t) initialized from the source BN.
      - Modes:
          * "infer_old": normalize using (mu_t, var_t) ONLY (no fusion, no write).
          * "adapt":     compute (mu_b, var_b) on current batch, fuse with (mu_t, var_t)
                        via λ = B/(B+m), and WRITE BACK (mu_t, var_t).
                        The current forward output still uses OLD (mu_t, var_t) (so
                        current batch prediction is unaffected). New stats take effect
                        on the NEXT batch.
          * "normal":    (classic TTN) normalize with fused stats and write back.
                        (Not used in this script.)
    """
    def __init__(self, src_bn: nn.BatchNorm2d, m: float = 10.0):
        super().__init__()
        # reuse scale/shift from the source BN
        self.gamma = src_bn.weight
        self.beta  = src_bn.bias

        # persistent stats
        self.register_buffer("mu_t",  src_bn.running_mean.clone())
        self.register_buffer("var_t", src_bn.running_var.clone())

        self.eps  = src_bn.eps
        self.m    = float(m)
        self.mode = "infer_old"  # default

    def set_mode(self, mode: str):
        assert mode in ("infer_old", "adapt", "normal")
        self.mode = mode

    def forward(self, x: torch.Tensor):
        if self.mode == "infer_old":
            # use old persistent stats; do not update
            return nn.functional.batch_norm(
                x, self.mu_t, self.var_t, self.gamma, self.beta,
                training=False, momentum=0.0, eps=self.eps
            )

        # compute current batch stats
        mu_b  = x.mean(dim=[0, 2, 3])
        var_b = x.var (dim=[0, 2, 3], unbiased=False)
        B = max(1.0, float(x.size(0)))
        lam = B / (B + self.m)
        mu_f  = (1.0 - lam) * self.mu_t + lam * mu_b
        var_f = (1.0 - lam) * self.var_t + lam * var_b

        if self.mode == "adapt":
            # write back for NEXT batch; but current output still uses OLD stats
            with torch.no_grad():
                self.mu_t.copy_(mu_f)
                self.var_t.copy_(var_f)
            return nn.functional.batch_norm(
                x, self.mu_t, self.var_t, self.gamma, self.beta,
                training=False, momentum=0.0, eps=self.eps
            )

        # "normal": normalize with fused stats and write back (not used here)
        y = nn.functional.batch_norm(
            x, mu_f, var_f, self.gamma, self.beta,
            training=False, momentum=0.0, eps=self.eps
        )
        with torch.no_grad():
            self.mu_t.copy_(mu_f)
            self.var_t.copy_(var_f)
        return y


def _get_parent_module(root: nn.Module, dotted_name: str) -> nn.Module:
    """
    Given a dotted module name like 'layer1.0.bn1', return the parent module object.
    If no dot exists, return root.
    """
    if '.' not in dotted_name:
        return root
    parent_path = dotted_name.split('.')[:-1]
    parent = root
    for p in parent_path:
        parent = getattr(parent, p)
    return parent

def replace_with_ttn(model: nn.Module, m: float, update: bool = True):
    """
    Replace all nn.BatchNorm2d modules in `model` with TTNorm(m).
    NOTE: the 'update' flag from the original script is no longer needed because
    deferred-apply is controlled by 'mode' (infer_old/adapt). Kept for API compat.
    """
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


# ----------------------------- Runner Helpers ------------------------------ #

def ttn_setup(model: nn.Module, *, update_stats: bool):
    """
    Keep for compatibility; in this deferred-apply implementation we use
    set_ttn_mode() around each batch. This function only toggles train/eval
    for semantics (has no effect on TTNorm which uses functional BN).
    """
    model.train() if update_stats else model.eval()
    return None  # no optimizer needed for pure TTN evaluation


@torch.inference_mode()
def eval_one_snr(loader, model, mod, device, psnr_db: int):
    """
    Evaluate one SNR point with deferred-apply TTN:
      - For each batch:
          1) Inference pass: set mode="infer_old", run encode->sampler->decode for logits.
          2) Adapt pass: set mode="adapt", run encode->sampler->decode to fuse & write stats
             for BOTH encoder and decoder (current batch outputs are ignored).
      - Returns (acc1, acc3, ms_per_img).
    """
    correct1 = correct3 = samples = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # -------- Inference phase: use OLD stats only (no write-back) --------
        set_ttn_mode(model, "infer_old")
        feat, shp = model.encode(x)
        feat, _   = model.sampler(feat, mod=mod)
        logits    = model.decode(feat, shp)

        # Accuracy
        top1, top3 = accuracy(logits, y, (1, 3))
        bs = x.size(0)
        correct1 += (top1.item() / 100.0) * bs
        correct3 += (top3.item() / 100.0) * bs
        samples  += bs

        # -------- Adapt phase: fuse & write stats for NEXT batch (encoder + decoder) --------
        set_ttn_mode(model, "adapt")
        feat2, shp2 = model.encode(x)               # encoder TTNorm updates
        feat2, _    = model.sampler(feat2, mod=mod) # propagate to decoder domain
        _           = model.decode(feat2, shp2)     # decoder TTNorm updates
        set_ttn_mode(model, "infer_old")

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    acc1 = (correct1 / samples) * 100.0
    acc3 = (correct3 / samples) * 100.0
    ms_per_img = dt / samples * 1e3

    print(f"[SNR {psnr_db:2d} dB]  Acc@1 {acc1:6.2f}  Acc@3 {acc3:6.2f}  {ms_per_img:.2f} ms/img")
    return acc1, acc3, ms_per_img


# --------------------------------- Main ------------------------------------ #

def main(args):
    print(f"\n[Config] Dataset: {args.dataset} | Modulation: {args.mod.upper()} | "
          f"Channel: {args.channel.upper()}\n")

    # Instantiate model by dataset base
    base = dataset_base_key(args.dataset)
    if base == "cifar100":
        model = JSCC_model.DTJSCC_CIFAR100(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif base == "cifar10":
        model = JSCC_model.DTJSCC_CIFAR10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif base == "cinic10":
        model = JSCC_model.DTJSCC_CINIC10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    else:
        raise ValueError(f"No available model for dataset: {args.dataset}, "
                         f"please check model/DT_JSCC.py.")

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_states"])
    model.to(args.device)

    # Replace all BN2d with TTNorm (deferred-apply). Default mode="infer_old".
    replace_with_ttn(model, m=args.tn_m, update=True)

    # Save a clean base state so we can restore between SNRs if desired
    base_state: Dict[str, torch.Tensor] = {k: v.clone() for k, v in model.state_dict().items()}

    # Data loader (test/val split)
    loader = get_data(args.dataset, args.bs, n_worker=0, train=False)

    # Prepare containers
    acc1s, acc3s, ms_list = [], [], []

    # Evaluate across SNR = 0..25 dB
    for psnr in range(26):
        # Build modulation/channel object
        kw = {}
        if args.channel == "rician":
            kw["K"] = args.K
        elif args.channel == "nakagami":
            kw["m"] = args.m

        mod = (QAM if args.mod == "qam" else PSK)(
            args.num_embeddings, psnr, args.channel, kw
        )

        # (Optional semantics) reflect we will update stats across batches
        _ = ttn_setup(model, update_stats=True)

        # Evaluate one SNR point (deferred-apply inside)
        a1, a3, ms = eval_one_snr(loader, model, mod, args.device, psnr)
        acc1s.append(a1)
        acc3s.append(a3)
        ms_list.append(ms)

        # Restore the entire model state (including TTNorm buffers) between SNRs
        model.load_state_dict(base_state, strict=True)

        # Persist results after each SNR
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            {"acc1s": acc1s, "acc3s": acc3s, "ms": ms_list},
            os.path.join(args.save_dir, f"ttn-eval-{base}-{args.channel}.pt")
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("TTN evaluation with deferred-apply per-batch stats")

    # Core config
    p.add_argument("--dataset", default="CIFAR10_noise",
                   help="Dataset name, variants allowed, e.g., CIFAR10_blur, CINIC10_noise")
    p.add_argument("--mod", choices=["psk", "qam"], default="psk")
    p.add_argument("--num_embeddings", type=int, default=16)

    # Model dims
    p.add_argument('--in_channels', type=int, default=3, help='Input image channels')
    p.add_argument('--latent_d', type=int, default=512, help='Latent dimension for JSCC encoder')

    # Paths
    p.add_argument("--model_dir", default="JT-JSCC/trained_models")
    p.add_argument("--name", default="", help="Subfolder under model_dir that holds the checkpoint")
    p.add_argument('--save_dir', type=str, default='AdapS/eval')

    # Runtime
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bs", type=int, default=256, help="Test batch size")

    # Channel params
    p.add_argument("--channel", choices=["awgn", "rayleigh", "rician", "nakagami"],
                   default="rayleigh")
    p.add_argument("--K", type=float, default=3.0, help="Rician K-factor")
    p.add_argument("--m", type=float, default=2.0, help="Nakagami m-parameter")

    # TTNorm-specific
    p.add_argument("--tn_m", type=float, default=10.0,
                   help="Smoothing constant m in lambda = B / (B + m)")
    p.add_argument("--update_stats", action="store_true",
                   help="(Kept for compatibility; not used directly)")

    args = p.parse_args()

    # Derived fields
    args.num_classes = infer_num_classes(args.dataset)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model checkpoint path (defaults to '<base>_best.pt' inside model_dir/name)
    ckpt_name = default_ckpt_filename(args.dataset)
    args.model_path = os.path.join(args.model_dir, args.name, ckpt_name)

    # Run
    main(args)
