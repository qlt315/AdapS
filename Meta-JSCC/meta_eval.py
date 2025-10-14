import os
import time
import warnings
import torch
import torch.nn.functional as F
from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy
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

# ---------- single‑PSNR evaluation ----------
def eval_test(model, mod, args, psnr, adaptation=False):
    device = args.device
    model.eval()

    # ----- inner‑loop adaptation -------------------------------------------
    adapt_time = 0.0
    if adaptation:
        # build a fresh support loader (no shuffle; deterministic order)
        n_worker = 0 if os.name == 'nt' else 2
        support_loader = get_data(
            args.dataset, args.inner_bs, n_worker=n_worker, train=False
        )
        support_iter = iter(support_loader)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.inner_lr, momentum=0.9)

        start = time.time()
        model.train()
        for _ in range(args.adapt_steps):
            imgs, labs = next(support_iter)
            imgs, labs = imgs.to(device), labs.to(device)

            optimizer.zero_grad()
            outs, _ = model(imgs, mod=mod)
            loss = F.cross_entropy(outs, labs)
            loss.backward()
            optimizer.step()
        adapt_time = time.time() - start
        model.eval()                              # back to eval for query

    # ----- query / validation ----------------------------------------------
    n_worker = 0 if os.name == 'nt' else 4
    query_loader = get_data(args.dataset, 256, n_worker=n_worker, train=False)

    correct1 = correct3 = total = 0
    with torch.no_grad():
        for imgs, labs in query_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outs, dist = model(imgs, mod=mod)
            c1, c3 = accuracy(outs, labs, (1, 3))
            bsz = labs.size(0)
            correct1 += c1.item() * bsz / 100
            correct3 += c3.item() * bsz / 100
            total += bsz

    acc1 = correct1 / total * 100
    acc3 = correct3 / total * 100
    print(
        f"[Eval] PSNR={psnr:2d}dB | Acc@1={acc1:6.2f}% | "
        f"Acc@3={acc3:6.2f}% | Adapt {adapt_time*1e3:6.2f} ms"
    )
    return acc1, acc3, dist, adapt_time


# ---------- main entry ----------------------------------------------------
def main(args):
    print(
        f"\n[Config] Dataset={args.dataset} | Mod={args.mod.upper()} "
        f"| Channel={args.channel.upper()} | Adapt={args.adapt}\n"
    )

    # build backbone --------------------------------------------------------
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
    # load checkpoint -------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="You are using `torch.load` with `weights_only=False`",
        )
        ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_states"] if "model_states" in ckpt else ckpt)
    model.to(args.device)

    # back‑up original weights (for per‑PSNR restore) -----------------------
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    PSNRs = list(range(0, 26, 1))
    acc1s, acc3s, adapt_times, dists = [], [], [], []

    for psnr in PSNRs:
        channel_args = {}
        if args.channel == "rician":
            channel_args["K"] = args.K
        elif args.channel == "nakagami":
            channel_args["m"] = args.m

        mod = (QAM if args.mod == "qam" else PSK)(
            args.num_embeddings, psnr, args.channel, channel_args
        )

        a1, a3, dist, t = eval_test(
            model, mod, args, psnr,
            adaptation=args.adapt
        )

        acc1s.append(a1); acc3s.append(a3)
        adapt_times.append(t); dists.append(dist)

        # restore weights for next PSNR ------------------------------------
        model.load_state_dict(orig_state)
        model.to(args.device)

    # save ------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir, f"meta-eval-{args.dataset}-{args.channel}.pt"
    )
    torch.save(
        dict(acc1s=acc1s, acc3s=acc3s, adapt_times=adapt_times, dists=dists),
        save_path,
    )
    if args.adapt and adapt_times:
        total_t = sum(adapt_times)
        mean_t = total_t / len(adapt_times)
        print(f"\n[Adaptation time] total = {total_t * 1e3:.2f} ms  "
              f"| mean = {mean_t * 1e3:.2f} ms")
    print(f"\n[Done] Results saved to {save_path}")


# ---------- CLI -----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Meta Evaluation")

    # model & data
    parser.add_argument("--dataset", type=str, default="CIFAR100_noise")
    parser.add_argument("--mod", type=str, choices=["qam", "psk"], default="psk")
    parser.add_argument("--num_embeddings", type=int, default=16)
    parser.add_argument("--latent_d", type=int, default=512)
    parser.add_argument("--num_latent", type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=3, help='The image input channel')

    # channel
    parser.add_argument("--channel", type=str, default="rician",
                        choices=["awgn", "rayleigh", "rician", "nakagami"])
    parser.add_argument("--K", type=float, default=3.0)
    parser.add_argument("--m", type=float, default=2.0)

    # adaptation hyper‑params
    parser.add_argument("--adapt", action="store_true", default=True)
    parser.add_argument("--adapt_steps", type=int, default=5)
    parser.add_argument("--inner_bs", type=int, default=256)
    parser.add_argument("--inner_lr", type=float, default=1e-2)

    # paths
    parser.add_argument("--model_dir", type=str, default="Meta-JSCC/trained_models")
    parser.add_argument("--model_file", type=str, default="cifar100_meta_best.pt")
    parser.add_argument("--save_dir", type=str, default="Meta-JSCC/eval")

    # runtime
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    args.num_classes = infer_num_classes(args.dataset)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.model_path = os.path.join(args.model_dir, args.model_file)

    main(args)
