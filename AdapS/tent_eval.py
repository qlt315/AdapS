import os, time, re
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy

# ========================= Dataset helpers (regex) ===========================
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

def infer_num_classes(dataset_name: str) -> int:
    """Infer number of classes based on regex-detected dataset base."""
    base = detect_base(dataset_name)
    if base == "CIFAR100":
        return 100
    if base in ("CIFAR10", "CINIC10"):
        return 10
    raise ValueError(f"Unknown dataset for inferring num_classes: {dataset_name}")

# ============================== TENT utilities ===============================
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Standard entropy of softmax outputs:
      H(p) = - sum_i p_i * log(p_i)
    """
    log_probs = torch.log_softmax(logits, dim=1)
    probs     = torch.softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()

def tent_configure_model(model: nn.Module):
    """
    Freeze all parameters except BatchNorm affine (gamma/beta).
    Put model in train mode so BN updates its running stats.
    Return the list of parameters to optimize (BN affine only).
    """
    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # enable BN affine
    bn_affine_params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and m.affine:
            if m.weight is not None:
                m.weight.requires_grad_(True)
                bn_affine_params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad_(True)
                bn_affine_params.append(m.bias)

    # train mode to update BN running stats during TENT
    model.train()
    return bn_affine_params

def tent_setup(model: nn.Module, lr: float = 1e-4, wd: float = 0.0, optim_type: str = "adam"):
    """
    Prepare model & optimizer for TENT. Only BN affine are optimized.
    """
    params = tent_configure_model(model)
    if len(params) == 0:
        # Fallback: no BN with affine found. Keep a dummy optimizer to avoid crashes.
        return optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)
    if optim_type.lower() == "sgd":
        opt = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        opt = optim.Adam(params, lr=lr, weight_decay=wd)
    return opt

# ============================= Eval per-PSNR (TENT) ==========================
def eval_psnr_tent(loader, model, mod, args, psnr) -> tuple[float, float, float]:
    """
    Run TENT adaptation over the loader for a given PSNR.
    - Unsupervised adaptation: minimize prediction entropy (no labels used).
    - Update only BN affine (gamma/beta).
    - Return (acc1, acc3, ms_per_img).
    """
    # fresh optimizer each PSNR (episodic setting)
    optimizer = tent_setup(model, lr=args.tent_lr, wd=args.tent_wd, optim_type=args.tent_optim)

    correct1 = correct3 = samples = 0
    if args.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for x, y in loader:
        x = x.to(args.device)
        y = y.to(args.device)

        # forward (no torch.no_grad — we need gradients for TENT)
        feat, shp = model.encode(x)
        feat, _   = model.sampler(feat, mod=mod)
        logits    = model.decode(feat, shp)

        # TENT loss: entropy minimization
        loss = softmax_entropy(logits)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # metrics for logging (labels are only used for reporting, not for loss)
        top1, top3 = accuracy(logits.detach(), y, (1, 3))
        bs = x.size(0)
        correct1 += top1.item() / 100.0 * bs
        correct3 += top3.item() / 100.0 * bs
        samples  += bs

    if args.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    acc1 = correct1 / samples * 100.0
    acc3 = correct3 / samples * 100.0
    ms_per_img = dt / samples * 1e3
    print(f"[SNR {psnr:2d} dB]  Acc@1 {acc1:6.2f}  Acc@3 {acc3:6.2f}  {ms_per_img:.2f} ms/img")
    return acc1, acc3, ms_per_img

# ================================== Main ====================================
def main(args):
    print(f"\n[Config] Dataset: {args.dataset} | Modulation: {args.mod.upper()} | Channel: {args.channel.upper()}\n")

    # ------------------ build model via regex dataset base -------------------
    ds = str(args.dataset).strip().lower()
    if   re.match(r'^cifar100(\b|[_-])', ds):
        Model = JSCC_model.DTJSCC_CIFAR100
    elif re.match(r'^cifar10(\b|[_-])',  ds):
        Model = JSCC_model.DTJSCC_CIFAR10
    elif re.match(r'^cinic10(\b|[_-])',  ds):
        Model = JSCC_model.DTJSCC_CINIC10
    else:
        raise ValueError(f"No available model for dataset: {args.dataset}, please check model/DT_JSCC.py.")

    model = Model(
        args.in_channels, args.latent_d, args.num_classes,
        num_embeddings=args.num_embeddings
    )

    # --------------------------- load checkpoint -----------------------------
    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_states"])
    model.to(args.device)

    # -------------------------- prepare dataloader ---------------------------
    loader = get_data(args.dataset, args.bs, n_worker=0, train=False)

    # ------------------------- episodic base state ---------------------------
    # keep a copy of initial (source) weights; restore after each PSNR
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ----------------------------- eval over PSNR ----------------------------
    acc1s, acc3s, ms_list = [], [], []
    os.makedirs(args.save_dir, exist_ok=True)

    for psnr in range(26):
        # build channel modulator
        ch_kwargs = {"K": args.K} if args.channel == "rician" else {}
        if args.channel == "nakagami":
            ch_kwargs["m"] = args.m
        mod = (QAM if args.mod == "qam" else PSK)(
            args.num_embeddings, psnr, args.channel, ch_kwargs
        )

        # TENT adaptation & evaluation
        a1, a3, ms_img = eval_psnr_tent(loader, model, mod, args, psnr)
        acc1s.append(a1); acc3s.append(a3); ms_list.append(ms_img)

        # episodic reset to source model
        model.load_state_dict(base_state, strict=True)

        # save after each PSNR (keeps original behavior)
        torch.save(
            {"acc1s": acc1s, "acc3s": acc3s, "ms": ms_list},
            os.path.join(args.save_dir, f"tent-eval-{args.dataset}-{args.channel}.pt")
        )

# =================================== CLI ====================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("TENT eval (test-time entropy minimization)")

    # basic
    p.add_argument("--dataset", default="CINIC10_noise")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bs", type=int, default=256, help="test batch size")
    p.add_argument("--save_dir", type=str, default="AdapS/eval")
    p.add_argument("--model_dir", default="JT-JSCC/trained_models")
    p.add_argument("--name", default="")

    # model
    p.add_argument('--in_channels', type=int, default=3, help='image input channels')
    p.add_argument('--latent_d', type=int, default=512)
    p.add_argument("--num_embeddings", type=int, default=16)
    p.add_argument("--num_latent", type=int, default=4)  # kept for compatibility
    p.add_argument("--mod", choices=["psk","qam"], default="psk")

    # channel
    p.add_argument("--channel", choices=["awgn","rayleigh","rician","nakagami"], default="rician")
    p.add_argument("--K", type=float, default=3.0)
    p.add_argument("--m", type=float, default=2.0)

    # TENT hyper-params
    p.add_argument("--tent_lr", type=float, default=1e-4, help="learning rate for BN affine")
    p.add_argument("--tent_wd", type=float, default=0.0,  help="weight decay for BN affine")
    p.add_argument("--tent_optim", type=str, default="adam", choices=["adam","sgd"])

    args = p.parse_args()
    args.num_classes = infer_num_classes(args.dataset)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # default checkpoint path inferred from dataset base
    base_lower = detect_base(args.dataset).lower()
    if hasattr(args, "name") and args.name:
        args.model_path = os.path.join(args.model_dir, args.name, f"{base_lower}_best.pt")
    else:
        args.model_path = os.path.join(args.model_dir, f"{base_lower}_best.pt")

    main(args)
