import os, time, warnings, torch, torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import make_grid

import model.DT_JSCC as JSCC_model
from model.losses          import RIBLoss
from datasets.dataloader   import get_data
from engine                import train_one_epoch, test
from utils.modulation      import QAM, PSK
import multiprocessing as mp
if os.name == "nt":                  # Windows needs explicit context
    mp.set_start_method("spawn", force=True)

# -------------------------------------------------------------------------- #
#  helpers                                                                   #
# -------------------------------------------------------------------------- #
def infer_num_classes(dataset_name: str) -> int:
    n = dataset_name.upper()
    if n.startswith("CIFAR100"): return 100
    if n.startswith("CIFAR10"):  return 10
    if n.startswith("CINIC10"):  return 10
    raise ValueError(f"Unknown dataset: {dataset_name}")

def build_backbone(args):
    """Return a fresh JSCC backbone according to dataset name (no MNIST)."""
    ds = args.dataset.upper()
    num_classes = infer_num_classes(ds)
    kwargs = dict(in_channels=3,
                  latent_channels=args.latent_d,
                  out_classes=num_classes,
                  num_embeddings=args.num_embeddings)
    if ds.startswith("CIFAR100"):
        return JSCC_model.DTJSCC_CIFAR100(**kwargs)
    elif ds.startswith("CINIC10"):
        return JSCC_model.DTJSCC_CINIC10(**kwargs)
    elif ds.startswith("CIFAR10"):
        return JSCC_model.DTJSCC_CIFAR10(**kwargs)
    else:
        raise ValueError(f"No available model for dataset: {args.dataset}")

def make_modulator(psnr: float, chan: str, args):
    """Return QAM / PSK channel object with given PSNR."""
    kw = {}
    if chan == "rician":
        kw["K"] = args.K
    elif chan == "nakagami":
        kw["m"] = args.m
    mod_cls = QAM if args.mod == "qam" else PSK
    return mod_cls(args.num_embeddings, psnr, chan, kw)

def stream_definition(base: str):
    """Dynamic 6-domain curriculum for {base, base_noise} × {awgn, rician, rayleigh}."""
    return [
        ("awgn",     base),
        ("rician",   base),
        ("rayleigh", base),
        ("awgn",     f"{base}_noise"),
        ("rician",   f"{base}_noise"),
        ("rayleigh", f"{base}_noise"),
    ]


# -------------------------------------------------------------------------- #
#  training for one domain                                                   #
# -------------------------------------------------------------------------- #
def run_one_domain(idx, chan, dataset_name, ckpt_in, args):
    """
    • Load ckpt from previous domain (or initial baseline if None)
    • Retrain for N epochs under current (chan, dataset)
    • Save final checkpoint to feed into next domain
    """
    tag = f"{idx:02d}-{chan}-{dataset_name}"
    print(f"\n[{tag}]  starting …")

    # ---------- model & optimizer -----------------------------------------
    model = build_backbone(args).to(args.device)
    opt    = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched  = lr_scheduler.StepLR(opt, step_size=80, gamma=0.5)
    crit   = RIBLoss(args.lam)    # simple CE+KL loss

    # ---------- restore previous weights (if any) -------------------------
    if ckpt_in and os.path.isfile(ckpt_in):
        state = torch.load(ckpt_in, map_location="cpu")
        model.load_state_dict(state["model_states"], strict=False)
        if "optimizer_states" in state:
            try:
                opt.load_state_dict(state["optimizer_states"])
            except Exception:
                pass
        print(f"[{tag}]  loaded ckpt {ckpt_in}")

    # ---------- data ------------------------------------------------------
    nw = 0 if os.name == "nt" else args.num_workers
    train_loader = get_data(dataset_name, args.N,  n_worker=nw)
    val_loader   = get_data(dataset_name, args.N,  n_worker=nw, train=False)

    # tensorboard
    log_dir = os.path.join("RT-JSCC/logs", tag)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_acc = 0.0
    psnr = args.psnr                     # fixed for all domains
    mod  = make_modulator(psnr, chan, args)

    # ---------- timing ----------------------------------------------------
    start_t = time.time()

    for ep in range(args.epochs):
        train_one_epoch(train_loader, model, opt, crit, writer,
                        epoch=ep, mod=mod, args=args)
        sched.step()

        if ep >= 1:     # validate after warm-up
            acc1 = test(val_loader, model, crit, writer, ep, mod, args)
            best_acc = max(best_acc, acc1)

    elapsed = time.time() - start_t
    print(f"[{tag}]  finished in {elapsed/60:.2f} min | best Acc@1={best_acc:.2f}%")

    # ---------- save ckpt -------------------------------------------------
    ckpt_path = os.path.join(args.save_dir, f"stream_{tag}.pt")
    torch.save(dict(model_states=model.state_dict(),
                    optimizer_states=opt.state_dict(),
                    best_acc=best_acc), ckpt_path)
    return ckpt_path, best_acc


# -------------------------------------------------------------------------- #
#  main loop over 6 domains                                                  #
# -------------------------------------------------------------------------- #
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # normalize dataset base name (CIFAR10 / CIFAR100 / CINIC10)
    base = args.dataset.upper()
    if   base.startswith("CIFAR100"): base = "CIFAR100"
    elif base.startswith("CINIC10"):  base = "CINIC10"
    elif base.startswith("CIFAR10"):  base = "CIFAR10"
    else: raise ValueError(f"Unsupported dataset: {args.dataset}")

    # set num_classes from dataset
    args.num_classes = infer_num_classes(base)

    stream = stream_definition(base)

    stream_t0 = time.time()
    ckpt_prev = None
    best_list = []

    for idx, (chan, dset) in enumerate(stream):
        ckpt_prev, best = run_one_domain(idx, chan, dset, ckpt_prev, args)
        best_list.append(best)

    print("\n[Summary]  Acc@1 per domain:", [f"{b:.2f}" for b in best_list])
    print(f"[Total]   stream retraining took {(time.time()-stream_t0)/60:.2f} min")


# -------------------------------------------------------------------------- #
#  CLI                                                                       #
# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, multiprocessing as mp
    P = argparse.ArgumentParser("Stream retraining for JSCC")

    # data / model
    P.add_argument("--dataset", default="CIFAR100")
    P.add_argument("--channels", type=int, default=3)
    P.add_argument("--num_classes", type=int, default=10)   # will be overwritten by infer_num_classes
    P.add_argument("--latent_d", type=int, default=512)
    P.add_argument("--num_latent", type=int, default=4)
    P.add_argument("--num_embeddings", type=int, default=16)
    P.add_argument('--maxnorm', type=float, default=1., help='Max norm')

    # training
    P.add_argument("--epochs", type=int, default=50)
    P.add_argument("--N",      type=int, default=512, help="batch size")
    P.add_argument("--lr",     type=float, default=1e-3)
    P.add_argument("--lam",    type=float, default=0.0)
    P.add_argument("--psnr",   type=float, default=8.0)

    # modulation / channel
    P.add_argument("--mod",     choices=["psk", "qam"], default="psk")
    P.add_argument("--K",       type=float, default=3.0)
    P.add_argument("--m",       type=float, default=2.0)

    # runtime
    P.add_argument("--device", default="cuda:0")
    P.add_argument("--num_workers", type=int, default=max(0, mp.cpu_count()-1))

    # output
    P.add_argument("--save_dir", default="RT-JSCC/stream_ckpts")

    args = P.parse_args()
    args.n_iter = 0
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    main(args)
