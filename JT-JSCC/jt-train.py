# joint_train.py  ——  multi‑task supervised training (no meta loop)
import os, torch, torch.optim as optim, torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import argparse, multiprocessing as mp
import model.DT_JSCC as JSCC_model
from model.losses import RIBLoss
from datasets.dataloader import get_data
from engine import train_one_epoch, test
from utils.modulation import QAM, PSK
import re

def infer_num_classes(dataset_name: str) -> int:
    name = dataset_name.upper()
    if name.startswith("CIFAR100"):
        return 100
    if name.startswith("CIFAR10"):
        return 10
    if name.startswith("CINIC10"):
        return 10
    # fallback: keep user-specified value (or raise)
    raise ValueError(f"Unknown dataset for inferring num_classes: {dataset_name}")



# --------------------- build modulator ------------------------------------
def build_mod(mod_type, num_emb, psnr, channel, channel_args):
    cls = QAM if mod_type == "qam" else PSK
    return cls(num_emb, psnr, channel, channel_args)

# --------------------- main ------------------------------------------------
def main(args):
    ds = str(args.dataset).strip().lower()

    if re.match(r'^cifar100(\b|[_-])', ds):
        model = JSCC_model.DTJSCC_CIFAR100(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif re.match(r'^cifar10(\b|[_-])', ds):
        model = JSCC_model.DTJSCC_CIFAR10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif re.match(r'^cinic10(\b|[_-])', ds):
        model = JSCC_model.DTJSCC_CINIC10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    else:
        raise ValueError(
            f"No available model for dataset: {args.dataset}, please check model/DT_JSCC.py."
        )
    model.to(args.device)

    optimizer  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    criterion  = RIBLoss(args.lam)
    writer     = SummaryWriter(os.path.join("JSCC/logs"))

    # --------------- define tasks: (dataset, channel) ----------------------

    # choose tasks by dataset name
    ds = args.dataset.upper()
    if ds.startswith("CIFAR10"):
        base = "CIFAR10"
    elif ds.startswith("CINIC10"):
        base = "CINIC10"
    elif ds.startswith("CIFAR100"):
        base = "CIFAR100"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    tasks = [
        (base, "awgn"),
        (f"{base}_noise", "awgn"),
        (base, "rician"),
    ]

    # build one dataloader per task
    loaders = {d: get_data(d, args.N, n_worker=8, train=True) for d, _ in tasks}
    val_loader = get_data(args.dataset, args.N, n_worker=8, train=False)

    # checkpoint resume -----------------------------------------------------
    start_epoch, best_acc = 0, 0.0
    ckpt_path = os.path.join(path_to_save, f"{args.dataset.lower()}_last.pt")


    # ---------------- training loop ---------------------------------------
    for epoch in range(start_epoch, args.epoches):

        for dataset_name, channel_type in tasks:
            # choose channel‑specific args
            ch_args = {}
            if channel_type == "rician":   ch_args["K"] = args.K
            elif channel_type == "nakagami": ch_args["m"] = args.m

            mod = build_mod(args.mod, args.num_embeddings,
                            args.psnr, channel_type, ch_args)

            train_one_epoch(loaders[dataset_name], model, optimizer,
                            criterion, writer, epoch,
                            mod=mod, args=args)

        scheduler.step()

        # ---- validation ---------------------------------------------------
        if epoch % 5 == 0:
            ref_mod = build_mod(args.mod, args.num_embeddings,
                                args.psnr, args.channel, {})
            acc1 = test(val_loader, model, criterion,
                        writer, epoch, mod=ref_mod, args=args)

            if acc1 > best_acc:
                best_acc = acc1
                torch.save({"epoch": epoch,
                            "model_states": model.state_dict(),
                            "optimizer_states": optimizer.state_dict(),
                            "scheduler_states": scheduler.state_dict(),
                            "best_acc": best_acc},
                           os.path.join(path_to_save, f"{args.dataset.lower()}_best.pt"))

        # ---- always save last --------------------------------------------
        torch.save({"epoch": epoch,
                    "model_states": model.state_dict(),
                    "optimizer_states": optimizer.state_dict(),
                    "scheduler_states": scheduler.state_dict(),
                    "best_acc": best_acc}, ckpt_path)

# ---------------- argument & run ------------------------------------------
if __name__ == "__main__":


    parser = argparse.ArgumentParser("Joint JSCC Trainer")

    parser.add_argument('--dataset', type=str, default='CINIC10')
    parser.add_argument('--root', type=str, default='JT-JSCC/trained_models')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mod', type=str, default='psk')
    parser.add_argument('--num_latent', type=int, default=4)
    parser.add_argument('--latent_d', type=int, default=512)
    parser.add_argument('--in_channels', type=int, default=3, help='The image input channel')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--maxnorm', type=float, default=1.)
    parser.add_argument('--num_embeddings', type=int, default=16)
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--psnr', type=float, default=8.0)
    parser.add_argument('--K', type=float, default=3.0)
    parser.add_argument('--m', type=float, default=2.0)
    parser.add_argument('--channel', type=str, default='awgn',
                        choices=['awgn', 'rayleigh', 'rician', 'nakagami'], help='Channel type')
    args = parser.parse_args()
    args.num_classes = infer_num_classes(args.dataset)
    path_to_save = os.path.join(args.root)
    os.makedirs(path_to_save, exist_ok=True)
    args.n_iter = 0
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
