import os, torch, torch.optim as optim, torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import re
import model.DT_JSCC as JSCC_model         # multi-decoder version
from model.losses import RIBLoss
from datasets.dataloader import get_data
from engine import train_one_epoch, test
from utils.modulation import QAM, PSK

# ----------------------------- regex helpers ------------------------------ #
# Match dataset base at the beginning: cifar100 / cifar10 / cinic10,
# followed by a word boundary, underscore, hyphen, or end of string.
_DATASET_BASE_RE = re.compile(r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=re.IGNORECASE)

def detect_base(dataset_name: str) -> str:
    """Return canonical base name in UPPER (CIFAR10 / CIFAR100 / CINIC10) via regex."""
    ds = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(ds)
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).upper()

def infer_num_classes(dataset_name: str) -> int:
    """Infer number of classes based on regex-detected dataset base."""
    base = detect_base(dataset_name)
    if base == "CIFAR100":
        return 100
    if base in ("CIFAR10", "CINIC10"):
        return 10
    # Unreachable due to detect_base guard
    raise ValueError(f"Unknown dataset for inferring num_classes: {dataset_name}")

# --------------------------------------------------------------------------- #
def build_mod(mod_type, n_emb, psnr, channel, ch_args):
    """Factory for QAM / PSK modulators."""
    cls = QAM if str(mod_type).lower() == "qam" else PSK
    return cls(n_emb, psnr, channel, ch_args)

# --------------------------------------------------------------------------- #
def main(args):
    # ============ 1) Build task list ======================================
    # Choose tasks (dataset, channel, unique task_key) using regex-detected base.
    base = detect_base(args.dataset)  # "CIFAR10" / "CIFAR100" / "CINIC10"

    tasks = [
        (base, "awgn",   f"{base}_awgn"),
        (f"{base}_noise", "awgn", f"{base}_noise_awgn"),
        (base, "rician", f"{base}_rician"),
    ]
    task_keys = [k for _, _, k in tasks]  # pass to model for head selection

    # ============ 2) Model =================================================
    # Select model class by regex-detected base.
    if base == "CIFAR10":
        model = JSCC_model.DTJSCC_CIFAR10(
            args.in_channels, args.latent_d, args.num_classes,
            task_keys=task_keys, num_embeddings=args.num_embeddings
        )
    elif base == "CIFAR100":
        model = JSCC_model.DTJSCC_CIFAR100(
            args.in_channels, args.latent_d, args.num_classes,
            task_keys=task_keys, num_embeddings=args.num_embeddings
        )
    elif base == "CINIC10":
        model = JSCC_model.DTJSCC_CINIC10(
            args.in_channels, args.latent_d, args.num_classes,
            task_keys=task_keys, num_embeddings=args.num_embeddings
        )
    else:
        # Should not happen due to detect_base
        raise ValueError(f"No available model for dataset: {args.dataset}, please check model/DT_JSCC.py.")
    model.to(args.device)

    # ============ 3) Optimizer / LR schedule ===============================
    optimizer  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    criterion  = RIBLoss(args.lam)

    # ============ 4) DataLoaders per task =================================
    # Each task has its own training loader (by dataset name string),
    # while validation uses the original args.dataset string (may include suffix).
    loaders = {d: get_data(d, args.N, n_worker=8, train=True) for d, _, _ in tasks}
    val_loader = get_data(args.dataset, args.N, n_worker=8, train=False)

    # ============ 5) Logging / checkpoint =================================
    writer = SummaryWriter(os.path.join("MD-JSCC/logs", args.name))
    best_acc, start_epoch = 0.0, 0
    last_ckpt = os.path.join(args.ckpt_dir, f"{detect_base(args.dataset).lower()}_last.pt")

    # ============ 6) Training loop ========================================
    for epoch in range(start_epoch, args.epochs):

        # ----- iterate over tasks (multi-decoder heads) --------------------
        for dname, ch_type, key in tasks:
            ch_args = {"K": args.K} if ch_type == "rician" else \
                      {"m": args.m} if ch_type == "nakagami" else {}
            mod = build_mod(args.mod, args.num_embeddings, args.psnr, ch_type, ch_args)

            train_one_epoch(
                loaders[dname], model, optimizer, criterion,
                writer, epoch, mod=mod, args=args, task_key=key
            )

        scheduler.step()

        # ----- periodic validation ----------------------------------------
        if epoch % 5 == 0:
            val_mod = build_mod(args.mod, args.num_embeddings, args.psnr, 'awgn', {})
            val_task_key = f"{base}_awgn"
            acc = test(
                val_loader, model, criterion, writer,
                epoch, mod=val_mod, args=args, task_key=val_task_key
            )
            if acc > best_acc:
                best_acc = acc
                torch.save(
                    {"epoch": epoch, "model_states": model.state_dict()},
                    os.path.join(args.ckpt_dir, f"{detect_base(args.dataset).lower()}_best.pt")
                )

        # ----- always save last -------------------------------------------
        torch.save(
            {
                "epoch": epoch,
                "model_states": model.state_dict(),
                "optimizer_states": optimizer.state_dict(),
                "scheduler_states": scheduler.state_dict(),
                "best_acc": best_acc
            },
            last_ckpt
        )

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, multiprocessing as mp

    parser = argparse.ArgumentParser("Joint-task Multi-Decoder Trainer")
    # --------- CLI arguments ----------
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--root",    type=str, default="MD-JSCC/trained_models")
    parser.add_argument("--device",  type=str, default="cuda:0")
    parser.add_argument("--mod",     type=str, default="psk")
    parser.add_argument("--num_latent", type=int, default=4)
    parser.add_argument("--latent_d",   type=int, default=512)
    parser.add_argument('--in_channels', type=int, default=3, help='Image input channels')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--N",      type=int, default=256)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--maxnorm",type=float, default=1.)
    parser.add_argument("--num_embeddings", type=int, default=16)
    parser.add_argument("--lam",    type=float, default=0.0)
    parser.add_argument("--psnr",   type=float, default=8.0)
    parser.add_argument("--K", type=float, default=3.0)
    parser.add_argument("--m", type=float, default=2.0)

    args = parser.parse_args()

    # ---- regex-based inferences & names ----
    args.num_classes = infer_num_classes(args.dataset)  # uses regex
    args.device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.ckpt_dir = os.path.join(args.root)

    base_upper = detect_base(args.dataset)   # "CIFAR10" / "CIFAR100" / "CINIC10"
    ds_base = base_upper.lower()             # "cifar10" / "cifar100" / "cinic10"
    args.name = f"md-{ds_base}"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Global TensorBoard step
    args.n_iter = 0

    main(args)
