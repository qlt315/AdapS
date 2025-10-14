import os
import re
import torch

from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy

# ========================= Regex helpers for dataset =========================
# Match dataset base at the beginning: cifar100 / cifar10 / cinic10,
# followed by a word boundary, underscore, hyphen, or end of string.
_DATASET_BASE_RE = re.compile(r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=re.IGNORECASE)

def detect_base(dataset_name: str) -> str:
    """Return canonical base in UPPER (CIFAR10 / CIFAR100 / CINIC10) via regex."""
    s = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(s)
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).upper()

def infer_num_classes(base_upper: str) -> int:
    """Map base dataset to class count."""
    if base_upper == "CIFAR100":
        return 100
    if base_upper in ("CIFAR10", "CINIC10"):
        return 10
    raise ValueError(f"Unknown dataset base: {base_upper}")

def pick_model_cls(base_upper: str):
    """Select the multi-decoder model class by base dataset."""
    mapping = {
        "CIFAR10":  JSCC_model.DTJSCC_CIFAR10,
        "CIFAR100": JSCC_model.DTJSCC_CIFAR100,
        "CINIC10":  JSCC_model.DTJSCC_CINIC10,
    }
    try:
        return mapping[base_upper]
    except KeyError:
        raise ValueError(f"No available model for dataset base: {base_upper}. "
                         f"Please check model/DT_JSCC.py.")

# =============================== Eval helper =================================
def eval_test(loader, model, mod, task_key, psnr, device):
    """Run evaluation on one PSNR and return (acc1, acc3)."""
    acc1 = acc3 = 0.0
    model.eval()
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outs, _ = model(imgs, task_key=task_key, mod=mod)
            a1, a3 = accuracy(outs, labs, (1, 3))
            acc1 += a1.item()
            acc3 += a3.item()
    acc1 /= len(loader)
    acc3 /= len(loader)
    print(f"[Eval] PSNR={psnr:2d}dB | Acc@1={acc1:.4f} | Acc@3={acc3:.4f}")
    return acc1, acc3

# =============================== Main logic ==================================
def main(args):
    print(
        f"\n[Config] Dataset={args.dataset} | Mod={args.mod.upper()} "
        f"| Channel={args.channel.upper()}\n"
    )

    # ---- Dataset (regex-based) ----
    base = detect_base(args.dataset)                  # 'CIFAR10' / 'CIFAR100' / 'CINIC10'
    num_classes = infer_num_classes(base)
    ModelCls = pick_model_cls(base)

    # ---- Multi-decoder heads (keep consistent with training) ----
    keys = [f"{base}_awgn", f"{base}_noise_awgn", f"{base}_rician"]

    # Requested head from canonical base + channel
    requested_key = f"{base}_{args.channel}"
    if requested_key not in keys:
        # Minimal fallback among our known heads
        requested_key = f"{base}_rician"

    # ---- Build model (IMPORTANT: task_keys must be a LIST) ----
    model = ModelCls(
        in_channels=3,
        latent_channels=args.latent_d,
        out_classes=num_classes,
        task_keys=keys,                 # pass the LIST, not a single string
        num_embeddings=args.num_embeddings
    )

    # ---- Data loader ----
    # Use the original args.dataset string for loader (suffixes like _noise allowed),
    # but the decoder heads use the normalized base.
    n_worker = 0 if os.name == "nt" else 8
    val_loader = get_data(args.dataset, 256, n_worker=n_worker, train=False)

    # ---- Resolve checkpoint path by base (aligns with your training save names) ----
    base_lower = base.lower()
    default_ckpt = os.path.join(args.model_dir, f"{base_lower}_best.pt")
    model_path = args.model_path if getattr(args, "model_path", None) else default_ckpt

    if not os.path.exists(model_path):
        print(f"[Warn] Checkpoint not found at {model_path}. "
              f"Will still attempt to load and likely fail.")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_states"]

    # ---- Optional: remap decoder prefix if checkpoint base != current base ----
    def _remap_decoder_prefix(state_dict, old_base, new_base):
        """Rename 'decoders.<old>_' prefix to 'decoders.<new>_' to match current model."""
        if old_base == new_base:
            return state_dict
        out = {}
        old_prefix = f"decoders.{old_base}_"
        new_prefix = f"decoders.{new_base}_"
        for k, v in state_dict.items():
            if k.startswith(old_prefix):
                out[new_prefix + k[len(old_prefix):]] = v
            else:
                out[k] = v
        return out

    # Detect checkpoint's base from its decoder keys
    old_base_detected = None
    for k in sd.keys():
        if k.startswith("decoders.CIFAR10_"):
            old_base_detected = "CIFAR10"; break
        if k.startswith("decoders.CIFAR100_"):
            old_base_detected = "CIFAR100"; break
        if k.startswith("decoders.CINIC10_"):
            old_base_detected = "CINIC10"; break

    if old_base_detected and old_base_detected != base:
        print(f"[Warn] Remapping decoder prefix {old_base_detected} -> {base} in checkpoint")
        sd = _remap_decoder_prefix(sd, old_base_detected, base)

    # Load with strict=False so unrelated heads/params don't break loading
    incompatible = model.load_state_dict(sd, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing:
        print(f"[Load] Missing keys: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        print(f"[Load] Unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")

    model.to(args.device)

    # ---- Choose an actual task_key present in the model ----
    all_keys = list(getattr(model, "decoders", {}).keys())
    if requested_key in all_keys:
        task_key = requested_key
    elif f"{base}_rician" in all_keys:
        task_key = f"{base}_rician"
    else:
        # Keep your original ultimate fallback text for transparency
        print(f"[Warn] Decoder for {requested_key} not found, using CIFAR10_noise_awgn Decoder")
        task_key = "CIFAR10_noise_awgn"

    # ---- Evaluation loop over PSNR ----
    acc1s, acc3s = [], []
    for psnr in range(0, 26):
        ch_args = {"K": args.K} if args.channel == "rician" else \
                  {"m": args.m} if args.channel == "nakagami" else {}
        mod = (QAM if args.mod == "qam" else PSK)(
            args.num_embeddings, psnr, args.channel, ch_args
        )

        a1, a3 = eval_test(val_loader, model, mod, task_key, psnr, args.device)
        acc1s.append(a1)
        acc3s.append(a3)

    # ---- Save results ----
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"md-eval-{args.dataset}-{args.channel}.pt")
    torch.save(dict(acc1s=acc1s, acc3s=acc3s), out_path)
    print(f"\n[Done] results saved to {out_path}")

# ================================== CLI =====================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Multi-decoder evaluator")

    # basic
    parser.add_argument("--dataset", type=str, default="CINIC10_noise")
    parser.add_argument("--device",  type=str, default="cuda:0")
    parser.add_argument("--model_dir", type=str, default="MD-JSCC/trained_models")
    parser.add_argument("--save_dir",  type=str, default="MD-JSCC/eval")
    parser.add_argument("--model_path", type=str, default="")  # optional override

    # model / modulation
    parser.add_argument("--mod", type=str, default="psk", choices=["psk", "qam"])
    parser.add_argument("--num_embeddings", type=int, default=16)
    parser.add_argument("--latent_d", type=int, default=512)
    parser.add_argument("--num_latent", type=int, default=4)

    # channel
    parser.add_argument("--channel", type=str, default="rician",
                        choices=["awgn", "rayleigh", "rician", "nakagami"])
    parser.add_argument("--K", type=float, default=3.0)
    parser.add_argument("--m", type=float, default=2.0)

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
