import os, re, torch, torch.nn as nn, torch.optim as optim
from typing import Dict, List, Tuple, Optional
from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy

# ============================ Regex helpers ================================ #
_DATASET_BASE_RE = re.compile(r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=re.IGNORECASE)

def detect_base(dataset_name: str) -> str:
    """Return canonical base name in UPPER via regex ('CIFAR10'/'CIFAR100'/'CINIC10')."""
    ds = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(ds)
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).upper()

def infer_num_classes(dataset_name: str) -> int:
    base = detect_base(dataset_name)
    return 100 if base == "CIFAR100" else 10

# ====================== Non-destructive channel gating ===================== #
class FilterMask(nn.Module):
    r"""
    Soft remodeling: gate Conv2d output channels without changing tensor shapes.
    - Forward: y = conv(x) * mask, where mask ∈ {0,1}^{C_out}.
    - Backward: zero gradients for pruned (mask==0) output channels of weight/bias.
    """
    def __init__(self, conv: nn.Conv2d, selected_idx: torch.Tensor):
        super().__init__()
        assert isinstance(conv, nn.Conv2d)
        self.conv = conv
        oc = conv.out_channels

        device = conv.weight.device
        m = torch.zeros(oc, dtype=torch.float32, device=device)
        m[selected_idx.to(device)] = 1.0
        self.register_buffer("mask", m.view(1, oc, 1, 1))

        # Only pass gradients for selected output channels
        def _zero_grad_out_channels(p: torch.Tensor):
            if p.grad is None:
                return
            if p.shape[0] == m.numel():             # first dim is C_out
                nz = (m.view(-1) == 0)
                p.grad[nz] = 0

        self.conv.weight.register_hook(_zero_grad_out_channels)
        if self.conv.bias is not None:
            self.conv.bias.register_hook(_zero_grad_out_channels)

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        return y * self.mask

# ========================== Module traversal utils ========================= #
def _iter_conv_layers(mod: nn.Module):
    """Yield (parent, name, conv) for every Conv2d in the module tree."""
    for name, child in mod.named_children():
        for p, n, c in _iter_conv_layers(child):     # recurse
            yield p, n, c
        if isinstance(child, nn.Conv2d):
            yield mod, name, child

def _collect_conv_bn_pairs(mod: nn.Module) -> List[Tuple[nn.Module, str, nn.Conv2d, str, nn.BatchNorm2d]]:
    """
    Heuristically find (conv -> next bn) pairs by scanning each parent's ordered children.
    Returns list of (parent, conv_name, conv, bn_name, bn).
    """
    pairs = []
    for parent in mod.modules():
        if not hasattr(parent, "_modules"):
            continue
        names = list(parent._modules.keys())
        for i, name in enumerate(names):
            child = parent._modules[name]
            if isinstance(child, nn.Conv2d):
                bn: Optional[nn.BatchNorm2d] = None
                bn_name = ""
                if i + 1 < len(names):
                    nxt = parent._modules[names[i + 1]]
                    if isinstance(nxt, nn.BatchNorm2d):
                        bn = nxt
                        bn_name = names[i + 1]
                if bn is not None:
                    pairs.append((parent, name, child, bn_name, bn))
    return pairs

# ============================ Freeze BN stats ============================== #
@torch.no_grad()
def _freeze_running_stats_bn(model: nn.Module):
    """
    Disable BN running-stat updates while keeping affine (gamma/beta) trainable.
    BN will use the existing running mean/var (no calibration).
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(False)  # stop running_mean/var updates
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)

# ============================ Scoring by BN grads ========================== #
def _one_grad_probe_and_score(
    model: nn.Module,
    imgs: torch.Tensor,
    loss_mode: str,
    labels: Optional[torch.Tensor],
    mod
) -> Dict[str, torch.Tensor]:
    """
    One forward+backward on a SINGLE streaming batch to get BN gamma/beta gradients.
    Return: dict mapping conv->bn pair key to per-channel score (|dγ| + |dβ|).
    """
    model.zero_grad(set_to_none=True)
    logits, _ = model(imgs, mod=mod)
    if loss_mode == "sup_ce":
        assert labels is not None, "labels required for supervised CE"
        loss = nn.functional.cross_entropy(logits, labels)
    elif loss_mode == "im":
        probs = torch.softmax(logits, dim=1)
        cond = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
        marginal = probs.mean(dim=0)
        loss = cond - (marginal * torch.log(marginal.clamp_min(1e-8))).sum()
    else:
        probs = torch.softmax(logits, dim=1)
        loss = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()

    loss.backward()

    conv_bn_pairs = _collect_conv_bn_pairs(model)
    scores: Dict[str, torch.Tensor] = {}
    for parent, conv_name, conv, bn_name, bn in conv_bn_pairs:
        key = f"{id(parent)}.{conv_name}->{bn_name}"
        g = None
        if bn.weight is not None and bn.weight.grad is not None:
            g = bn.weight.grad.detach().abs()
        if bn.bias is not None and bn.bias.grad is not None:
            gb = bn.bias.grad.detach().abs()
            g = gb if g is None else (g + gb)
        if g is None:
            if conv.weight.grad is not None:
                g = conv.weight.grad.detach().abs().flatten(1).mean(dim=1)
            else:
                g = conv.weight.detach().abs().flatten(1).mean(dim=1)
        scores[key] = g
    return scores

def _apply_masks_by_bn_grad(
    model: nn.Module,
    select_ratio: float,
    probe_batch: torch.Tensor,
    probe_labels: Optional[torch.Tensor],
    probe_mod,
    loss_mode: str = "im"
):
    """
    ElasticDNN selection on ONE streaming batch:
      1) Freeze BN running-stat updates (no calibration).
      2) One grad pass to get BN γ/β gradients.
      3) For each (conv->bn) pair, select top-p output channels by |dγ|+|dβ|.
      4) Insert FilterMask to gate channels（未选通道被梯度屏蔽）。
    """
    _freeze_running_stats_bn(model)
    model.train()  # enable grads for conv and BN affine
    scores = _one_grad_probe_and_score(model, probe_batch, loss_mode, probe_labels, probe_mod)

    conv_bn_pairs = _collect_conv_bn_pairs(model)
    for parent, conv_name, conv, _bn_name, _ in conv_bn_pairs:
        key = f"{id(parent)}.{conv_name}->{_bn_name}"
        s = scores[key]                   # [C_out]
        oc = conv.out_channels
        k = max(1, int(round(oc * float(select_ratio))))
        idx = torch.topk(s, k=k, largest=True).indices
        fm = FilterMask(conv, idx.to(conv.weight.device))
        setattr(parent, conv_name, fm)

# ============================== Param filter =============================== #
def selected_params(model: nn.Module):
    """Yield only parameters we update online: selected conv channels + BN affine."""
    for m in model.modules():
        if isinstance(m, FilterMask):
            yield m.conv.weight
            if m.conv.bias is not None:
                yield m.conv.bias
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                yield m.weight
            if m.bias is not None:
                yield m.bias

# ================================ Eval ===================================== #
def eval_one_psnr(val_loader, model, mod, device, psnr):
    model.eval()
    acc1 = acc3 = 0.0
    with torch.no_grad():
        for imgs, labs in val_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            logits, _ = model(imgs, mod=mod)
            a1, a3 = accuracy(logits, labs, (1, 3))
            acc1 += a1.item()
            acc3 += a3.item()
    acc1 /= len(val_loader)
    acc3 /= len(val_loader)
    print(f"[Eval] PSNR={psnr:2d}dB | Acc@1={acc1:.4f} | Acc@3={acc3:.4f}")
    return acc1, acc3

# ================================= Main ==================================== #
def main(args):
    print(f"\n[Config] Dataset: {args.dataset} | Modulation: {args.mod.upper()} | Channel: {args.channel.upper()}\n")
    base = detect_base(args.dataset)
    num_classes = infer_num_classes(args.dataset)

    # Model (single-decoder)
    if base == "CIFAR10":
        Model = JSCC_model.DTJSCC_CIFAR10
    elif base == "CIFAR100":
        Model = JSCC_model.DTJSCC_CIFAR100
    else:
        Model = JSCC_model.DTJSCC_CINIC10

    model = Model(args.in_channels, args.latent_d, num_classes, num_embeddings=args.num_embeddings)
    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_states"])
    model.to(args.device)

    # Data (streaming): do NOT assume we can grab multiple batches up front
    train_loader = get_data(args.dataset, args.bs, n_worker=0, train=True)
    val_loader   = get_data(args.dataset, args.bs, n_worker=0, train=False)

    # ---------------------- 1) ONE-BATCH remodeling (selection) ----------------------
    stream_iter = iter(train_loader)
    first_imgs, first_labels = next(stream_iter)  # only the first, available batch
    first_imgs   = first_imgs.to(args.device)
    first_labels = first_labels.to(args.device) if not args.unsupervised else None

    ch_args = {"K": args.K} if args.channel == "rician" else {"m": args.m} if args.channel == "nakagami" else {}
    probe_mod = (QAM if args.mod == "qam" else PSK)(args.num_embeddings, args.ft_psnr, args.channel, ch_args)

    loss_mode = "sup_ce" if (not args.unsupervised) else ("im" if args.im else "ent")
    _apply_masks_by_bn_grad(
        model,
        select_ratio=args.keep_ratio,
        probe_batch=first_imgs,
        probe_labels=first_labels,
        probe_mod=probe_mod,
        loss_mode=loss_mode
    )
    _freeze_running_stats_bn(model)  # keep BN stats frozen afterwards

    # ---------------------- 2) Streaming adaptation (per-batch) ----------------------
    # For each subsequent batch that arrives, do 'steps_per_batch' gradient steps on that batch only.
    model.train()                    # conv & BN affine trainable; BN stats frozen
    _freeze_running_stats_bn(model)  # double-safety
    opt = optim.Adam(selected_params(model), lr=args.lr)

    # first batch can also be used for adaptation (optional). We’ve already consumed it for selection;
    # if you want to adapt on it as well, uncomment the following small helper:
    def adapt_on_batch(imgs, labels):
        imgs = imgs.to(args.device)
        labels = labels.to(args.device)
        mod = (QAM if args.mod == "qam" else PSK)(args.num_embeddings, args.ft_psnr, args.channel, ch_args)
        for _ in range(args.steps_per_batch):
            opt.zero_grad()
            logits, _ = model(imgs, mod=mod)

            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            opt.step()

    # (Optional) also adapt on the first batch
    if args.adapt_on_first:
        adapt_on_batch(first_imgs, first_labels)

    # Now process the rest of the stream online:
    for imgs, labels in stream_iter:
        adapt_on_batch(imgs, labels)

    # ---------------------- 3) Evaluation after streaming ----------------------
    acc1s, acc3s = [], []
    for psnr in range(0, 26):
        ch_args_eval = {"K": args.K} if args.channel == "rician" else {"m": args.m} if args.channel == "nakagami" else {}
        mod = (QAM if args.mod == "qam" else PSK)(args.num_embeddings, psnr, args.channel, ch_args_eval)
        a1, a3 = eval_one_psnr(val_loader, model, mod, args.device, psnr)
        acc1s.append(a1); acc3s.append(a3)

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"ednn-eval-{args.dataset}-{args.channel}.pt")
    torch.save({"acc1s": acc1s, "acc3s": acc3s}, out_path)
    print(f"\n[Done] Saved: {out_path}")

# ================================== CLI ==================================== #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("EDNN Baseline (JSCC, streaming)")

    # Data / model
    p.add_argument("--dataset", default="CIFAR10_noise")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--latent_d",   type=int, default=512)
    p.add_argument("--num_embeddings", type=int, default=16)

    # Paths
    p.add_argument("--model_path", default="JT-JSCC/trained_models/cifar10_best.pt")
    p.add_argument("--save_dir",   default="AdapS/eval")

    # Channel / modulation
    p.add_argument("--mod", choices=["psk","qam"], default="psk")
    p.add_argument("--channel", choices=["awgn","rayleigh","rician","nakagami"], default="rayleigh")
    p.add_argument("--K", type=float, default=3.0)
    p.add_argument("--m", type=float, default=2.0)

    # Remodeling options
    p.add_argument("--keep_ratio", type=float, default=0.7, help="Per-layer top-p filter ratio for surrogate.")
    p.add_argument("--ft_psnr",  type=int, default=8, help="PSNR anchor used in probe/adaptation.")

    # Streaming adaptation options
    p.add_argument("--unsupervised", action="store_true", help="Use entropy/IM loss without labels.")
    p.add_argument("--im", action="store_true", help="Use Information Maximization (else entropy).")
    p.add_argument("--steps_per_batch", type=int, default=1, help="How many gradient steps for each arriving batch.")
    p.add_argument("--adapt_on_first", action="store_true", help="Also adapt on the first (probe) batch.")

    # Optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=256)

    # Device
    p.add_argument("--device", default="cuda:0")

    args = p.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
