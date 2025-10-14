# meta_train.py
import os
import re
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import higher

from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK

# ----------------------------- regex helpers ------------------------------ #
# Match base at the beginning: cifar100 / cifar10 / cinic10,
# followed by a word boundary, underscore, hyphen, or end of string.
_DATASET_BASE_RE = re.compile(r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=re.IGNORECASE)

def detect_base(dataset_name: str) -> str:
    """Return canonical base in UPPER (CIFAR10 / CIFAR100 / CINIC10) via regex."""
    ds = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(ds)
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).upper()

def infer_num_classes(dataset_name: str) -> int:
    """Infer number of classes based on regex-detected base."""
    base = detect_base(dataset_name)
    if base == "CIFAR100":
        return 100
    if base in ("CIFAR10", "CINIC10"):
        return 10
    # Unreachable due to detect_base guard
    raise ValueError(f"Unknown dataset for inferring num_classes: {dataset_name}")

# ---------- utility functions ----------
def freeze_running_stats(model):
    """Freeze running_mean and running_var of every BatchNorm layer."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            # Uncomment if you also want to freeze BN affine params:
            # m.weight.requires_grad_(False)
            # m.bias.requires_grad_(False)

def bn_to_gn(model, num_groups=8):
    """Recursively replace BatchNorm2d with GroupNorm."""
    import torch.nn as nn
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(
                num_groups=min(num_groups, child.num_features),
                num_channels=child.num_features,
            )
            setattr(model, name, gn)
        else:
            bn_to_gn(child, num_groups)

# ---------- meta-training ----------
def meta_train(args):
    writer = SummaryWriter(log_dir=os.path.join("Meta-JSCC/logs", args.name))
    best_meta_loss = float("inf")

    # Task list (use regex-detected base)
    base = detect_base(args.dataset)  # "CIFAR10" / "CIFAR100" / "CINIC10"
    tasks = [
        (base, "awgn"),
        (f"{base}_noise", "awgn"),
        (base, "rician"),
    ]

    # Build base model (by regex base)
    if base == 'CINIC10':
        base_model = JSCC_model.DTJSCC_CINIC10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif base == 'CIFAR10':
        base_model = JSCC_model.DTJSCC_CIFAR10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif base == 'CIFAR100':
        base_model = JSCC_model.DTJSCC_CIFAR100(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    else:
        # Should not happen due to detect_base guard
        raise ValueError(f"No available model for dataset: {args.dataset}, please check model/DT_JSCC.py.")

    base_model.train()
    base_model.to(args.device)

    # Load a supervised pre-training checkpoint if provided
    if args.init is not None and os.path.isfile(args.init):
        ckpt = torch.load(args.init, map_location="cpu")
        base_model.load_state_dict(ckpt["model_states"] if "model_states" in ckpt else ckpt)
        print(f"==> Loaded checkpoint from {args.init}")

    # Optional: replace BN with GN
    if args.use_gn:
        bn_to_gn(base_model, num_groups=8)
        base_model.to(args.device)
        print("==> Replaced all BatchNorm with GroupNorm")

    # Optional: freeze running stats of BN
    if args.freeze_bn:
        freeze_running_stats(base_model)
        print("==> Frozen BatchNorm running stats")

    # Outer optimizer
    meta_optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.meta_lr, weight_decay=1e-4)

    # Meta-training loop
    for meta_iter in range(args.meta_iters):
        meta_optimizer.zero_grad()
        total_meta_loss = 0.0
        print(f"\n=== Meta Iteration {meta_iter} ===")

        for task_idx, (dataset_name, channel_type) in enumerate(tasks):
            print(f"\n-- Task {task_idx + 1}/{len(tasks)}: Dataset={dataset_name}, Channel={channel_type}")

            # Build modulator (kept on CPU)
            channel_args = {}
            if channel_type == "rician":
                channel_args["K"] = args.K
            elif channel_type == "nakagami":
                channel_args["m"] = args.m
            mod = QAM(args.num_embeddings, args.psnr, channel_type, channel_args) \
                  if args.mod == "qam" else \
                  PSK(args.num_embeddings, args.psnr, channel_type, channel_args)

            # Dataloaders
            support_loader = get_data(dataset_name, args.inner_bs, n_worker=4)
            query_loader = get_data(dataset_name, args.query_bs, n_worker=4, train=False)

            # Inner optimizer
            inner_opt = torch.optim.SGD(base_model.parameters(), lr=args.inner_lr, momentum=0.9)

            # First-order MAML (track_higher_grads=False)
            with higher.innerloop_ctx(
                base_model,
                inner_opt,
                copy_initial_weights=True,
                track_higher_grads=False,
            ) as (fmodel, diffopt):

                fmodel.train()
                support_iter = iter(support_loader)

                # ---------- inner loop ----------
                for _ in range(args.inner_steps):
                    try:
                        imgs, lbls = next(support_iter)
                    except StopIteration:
                        support_iter = iter(support_loader)
                        imgs, lbls = next(support_iter)

                    imgs, lbls = imgs.to(args.device), lbls.to(args.device)
                    logits, _ = fmodel(imgs, mod=mod)
                    loss_s = F.cross_entropy(logits, lbls)
                    diffopt.step(loss_s)

                # ---------- meta (query) ----------
                fmodel.eval()
                query_iter = iter(query_loader)
                meta_loss, meta_acc = 0.0, 0.0

                for _ in range(args.query_steps):
                    try:
                        q_imgs, q_lbls = next(query_iter)
                    except StopIteration:
                        query_iter = iter(query_loader)
                        q_imgs, q_lbls = next(query_iter)

                    q_imgs, q_lbls = q_imgs.to(args.device), q_lbls.to(args.device)
                    q_logits, _ = fmodel(q_imgs, mod=mod)
                    loss_q = F.cross_entropy(q_logits, q_lbls)
                    meta_loss += loss_q

                    pred = torch.argmax(q_logits, dim=1)
                    meta_acc += (pred == q_lbls).float().mean().item()

                meta_loss /= args.query_steps
                meta_acc /= args.query_steps
                print(f"[Meta loss] {meta_loss.item():.4f}, Acc: {meta_acc*100:.2f}%")

                meta_loss.backward()
                total_meta_loss += meta_loss.item()

        # Update outer parameters
        meta_optimizer.step()
        if meta_iter == 0:
            ema_loss = total_meta_loss
        else:
            ema_loss = 0.2 * total_meta_loss + 0.8 * ema_loss
        writer.add_scalar("meta_loss", total_meta_loss, meta_iter)
        writer.add_scalar("ema_meta_loss", ema_loss, meta_iter)

        print(f"[MetaIter {meta_iter}] Total Loss={total_meta_loss:.4f}, EMA={ema_loss:.4f}")

        # ---------- save checkpoints ----------
        base_lower = detect_base(args.dataset).lower()
        ckpt = {"model_states": base_model.state_dict(),
                "meta_iter": meta_iter,
                "meta_loss": total_meta_loss}

        # Always keep last
        torch.save(ckpt, os.path.join(args.save_path, f"{base_lower}_meta_last.pt"))

        # Save best by meta-loss
        if total_meta_loss < best_meta_loss:
            best_meta_loss = total_meta_loss
            torch.save(ckpt, os.path.join(args.save_path, f"{base_lower}_meta_best.pt"))
            print(f"Saved new best model (loss {best_meta_loss:.4f})")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('-d', '--dataset', type=str, default='CINIC10', help='dataset name')
    # model hyper-parameters
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--latent_d", type=int, default=512)
    parser.add_argument("--num_embeddings", type=int, default=16)

    # channel / modulation
    parser.add_argument("--psnr", type=float, default=8.0)
    parser.add_argument("--mod", type=str, choices=["qam", "psk"], default="psk")
    parser.add_argument("--K", type=float, default=3.0)
    parser.add_argument("--m", type=float, default=2.0)

    # meta / inner-loop hyper-parameters
    parser.add_argument("--meta_iters", type=int, default=100)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--query_steps", type=int, default=4)
    parser.add_argument("--inner_bs", type=int, default=256)
    parser.add_argument("--query_bs", type=int, default=256)

    parser.add_argument("--meta_lr", type=float, default=2e-4)
    parser.add_argument("--inner_lr", type=float, default=1e-2)

    # extra switches
    parser.add_argument(
        "--freeze_bn",
        action="store_true",
        default=True,
        help="Freeze BatchNorm running stats",
    )
    parser.add_argument(
        "--use_gn", action="store_true", default=False, help="Replace BatchNorm with GroupNorm"
    )
    parser.add_argument(
        "--init", type=str, default=f"JSCC/trained_models/CINIC10-awgn/best.pt",
        help="Path to a supervised pre-training checkpoint"
    )

    # paths
    parser.add_argument("--name", type=str, default="meta-cinic100")
    parser.add_argument("--save_path", type=str, default="Meta-JSCC/trained_models")

    args = parser.parse_args()
    args.num_classes = infer_num_classes(args.dataset)  # regex-based
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)



    meta_train(args)
