# ------------------------------------------------------------- #
#  Streaming TTNorm eval (deferred-update version)              #
#  - This version ensures: CURRENT batch uses OLD stats;        #
#    NEW fused stats are written back AFTER forward and         #
#    therefore take effect STARTING FROM THE NEXT batch.        #
# ------------------------------------------------------------- #
import os, time, torch, torch.nn as nn
from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy  import accuracy


# ---------------- 1) TTNorm (deferred-apply) ----------------- #
class TTNorm(nn.Module):
    """
    Test-Time Normalization with deferred stat update (TTN):
      - Persistent stats (mu_t, var_t) are used to normalize the CURRENT batch.
      - After producing the output, we fuse CURRENT-BATCH stats into
        (mu_t, var_t), so the fused stats affect ONLY the NEXT batch.
      - Fusion: mu_f = (1-λ) * mu_t + λ * mu_b,  λ = B / (B + m)
                var_f = (1-λ) * var_t + λ * var_b
    """
    def __init__(self, src: nn.BatchNorm2d, m: float = 10.0):
        super().__init__()
        # reuse gamma/beta
        self.gamma, self.beta = src.weight, src.bias
        # persistent stats initialized from source running stats
        self.register_buffer("mu_t",  src.running_mean.clone())
        self.register_buffer("var_t", src.running_var.clone())
        self.eps = src.eps
        self.m   = float(m)

    @torch.no_grad()
    def _adapt_after(self, x: torch.Tensor):
        """Fuse current-batch stats into persistent stats for NEXT batch."""
        mu_b  = x.mean(dim=(0, 2, 3))
        var_b = x.var (dim=(0, 2, 3), unbiased=False)
        B     = float(max(1, x.size(0)))
        lam   = B / (B + self.m)
        mu_f  = (1.0 - lam) * self.mu_t + lam * mu_b
        var_f = (1.0 - lam) * self.var_t + lam * var_b
        self.mu_t.copy_(mu_f)
        self.var_t.copy_(var_f)

    def forward(self, x: torch.Tensor):
        # 1) Use OLD persistent stats for CURRENT inference
        y = nn.functional.batch_norm(
            x, self.mu_t, self.var_t, self.gamma, self.beta,
            training=False, momentum=0.0, eps=self.eps
        )
        # 2) AFTER forward, update persistent stats -> will affect NEXT batch
        self._adapt_after(x)
        return y


def replace_with_ttn(model: nn.Module, m: float):
    """
    Replace every nn.BatchNorm2d in `model` with deferred-apply TTNorm.
    """
    modules = dict(model.named_modules())
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.BatchNorm2d):
            if '.' in name:
                parent_name, child = name.rsplit('.', 1)
                parent = modules[parent_name]
            else:
                parent, child = model, name
            setattr(parent, child, TTNorm(mod, m).to(mod.weight.device))


# ---------------- 2) One epoch (single pass) ------------------ #
@torch.no_grad()
def run_epoch(loader, net, mod, dev):
    """
    Single pass over a loader:
      - TTNorm inside the network updates its persistent stats AFTER each batch,
        so next batch benefits from the fused stats.
      - The current batch prediction always uses the OLD stats.
    """
    correct1 = correct3 = samples = 0
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)

        feat, shp   = net.encode(x)
        feat, _     = net.sampler(feat, mod=mod)
        logits      = net.decode(feat, shp)

        top1, top3  = accuracy(logits, y, (1, 3))
        bs          = x.size(0)
        correct1   += (top1.item() / 100.0) * bs
        correct3   += (top3.item() / 100.0) * bs
        samples    += bs

    if dev.type == 'cuda':
        torch.cuda.synchronize()
    ms_per_img = (time.time() - t0) / max(1, samples) * 1e3
    acc1 = correct1 / max(1, samples) * 100.0
    acc3 = correct3 / max(1, samples) * 100.0
    return acc1, acc3, ms_per_img


# ---------------- 3) Streaming evaluation --------------------- #
def main(args):
    # build & load model by dataset base key
    base = args.dataset.upper()
    if base.startswith("CIFAR10"):
        base, num_cls, Net = "CIFAR10", 10, JSCC_model.DTJSCC_CIFAR10
    elif base.startswith("CINIC10"):
        base, num_cls, Net = "CINIC10", 10, JSCC_model.DTJSCC_CINIC10
    elif base.startswith("CIFAR100"):
        base, num_cls, Net = "CIFAR100", 100, JSCC_model.DTJSCC_CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    net = Net(3, args.lat_d, num_cls, args.num_emb)
    state = torch.load(args.model_path, map_location='cpu')
    # accept either {"model_states": ...} or plain state_dict
    net.load_state_dict(state["model_states"] if "model_states" in state else state)
    net.to(args.device)
    net.eval()

    # replace all BN2d with deferred-apply TTNorm
    replace_with_ttn(net, args.tn_m)

    # streaming domains (example)
    stream = [
        ("awgn",     base),
        ("rician",   base),
        ("rayleigh", base),
        ("awgn",     f"{base}_noise"),
        ("rician",   f"{base}_noise"),
        ("rayleigh", f"{base}_noise"),
    ]

    os.makedirs(args.ckpt_dir, exist_ok=True)
    acc1_all, acc3_all = [], []
    SNR = 0  # fixed PSNR for demonstration (adapt as needed)

    for idx, (ch, ds_name) in enumerate(stream, 1):
        # build channel object
        kw = {}
        if ch == "rician":
            kw["K"] = args.K
        elif ch == "nakagami":
            kw["m"] = args.m
        mod = (QAM if args.mod == 'qam' else PSK)(args.num_emb, SNR, ch, kw)

        # loader for this domain
        loader = get_data(ds_name, args.bs, n_worker=0, train=False)

        # single pass (TTNorm will update persistent stats AFTER each batch)
        a1, a3, ms = run_epoch(loader, net, mod, args.device)
        acc1_all.append(a1); acc3_all.append(a3)

        print(f"[{idx}/{len(stream)}] {ch.upper()} | {ds_name:<16} "
              f"Acc@1 {a1:6.2f}  Acc@3 {a3:6.2f}  {ms:.2f} ms/img")

        # snapshot the adapted model (its TTNorm buffers now reflect the last batch)
        torch.save(net.state_dict(),
                   os.path.join(args.ckpt_dir, f"ttn_after_domain{idx}.pt"))

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({"acc1": acc1_all, "acc3": acc3_all},
               os.path.join(args.save_dir, "ttn_stream_eval_results.pt"))


# ---------------- 4) CLI ------------------------------------- #
if __name__ == "__main__":
    import argparse
    P = argparse.ArgumentParser("Stream-TTN eval (deferred-update TTNorm)")
    P.add_argument("--dataset", default="CIFAR10")
    P.add_argument("--num_emb", type=int, default=16)
    P.add_argument("--lat_d",   type=int, default=512)
    P.add_argument("--num_lat", type=int, default=4)
    P.add_argument("--model_dir", default="JT-JSCC/trained_models")
    P.add_argument("--name",      default="")
    P.add_argument("--device",    default="cuda:0")
    P.add_argument("--bs",        type=int, default=256)
    P.add_argument("--mod", choices=["psk","qam"], default="psk")
    P.add_argument("--K",  type=float, default=3.0)
    P.add_argument("--m",  type=float, default=2.0)
    P.add_argument("--tn_m", type=float, default=10.0,
                   help="Smoothing constant m in λ = B / (B + m)")
    P.add_argument("--save_dir", default="AdapS/eval")
    P.add_argument("--ckpt_dir", default="AdapS/models/stream_ckpts")
    args = P.parse_args()

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # default checkpoint path; accepts either best.pt or *_best.pt
    default_name = "best.pt"
    args.model_path = os.path.join(args.model_dir, args.name, default_name)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
