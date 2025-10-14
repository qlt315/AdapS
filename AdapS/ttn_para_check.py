# ------------------------------------------------------------- #
#  TT-Norm batch-size / interval sweep (buffer-aware version)   #
# ------------------------------------------------------------- #
import os, time, torch, torch.nn as nn
from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy  import accuracy


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

# ------------------- TT-Norm layer --------------------------- #
class TTNorm(nn.Module):
    """Test-Time Normalization (ICLR’23)."""
    def __init__(self, src_bn: nn.BatchNorm2d, m: float = 10.0):
        super().__init__()
        self.gamma, self.beta = src_bn.weight, src_bn.bias
        self.register_buffer("mu_s",  src_bn.running_mean.clone())
        self.register_buffer("var_s", src_bn.running_var.clone())
        self.eps = src_bn.eps
        self.m   = m    # smoothing constant

    def forward(self, x: torch.Tensor):
        mu_b  = x.mean(dim=[0, 2, 3])
        var_b = x.var (dim=[0, 2, 3], unbiased=False)
        lam   = x.size(0) / (x.size(0) + self.m)       # total samples in *this* update
        mu_f  = (1-lam)*self.mu_s + lam*mu_b
        var_f = (1-lam)*self.var_s + lam*var_b
        return nn.functional.batch_norm(x, mu_f, var_f,
                                        self.gamma, self.beta,
                                        training=False, momentum=0,
                                        eps=self.eps)


def swap_to_ttn(net: nn.Module, m: float):
    """Replace every BatchNorm2d by TTNorm (call *after* loading ckpt)."""
    for name, mod in list(net.named_modules()):
        if isinstance(mod, nn.BatchNorm2d):
            p_name, c_name = name.rsplit('.', 1) if '.' in name else ('', name)
            parent = dict(net.named_modules()).get(p_name, net) if p_name else net
            setattr(parent, c_name, TTNorm(mod, m).to(mod.weight.device))


# --------------------- one run (bs,intv) --------------------- #
@torch.no_grad()
def run(loader, net, mod, device, interval):
    """Evaluate one (batch_size, interval) setting."""
    net.eval()
    buf = []                          # sample buffer
    acc1 = acc3 = samples = 0
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()

    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)

        # ---------- inference ----------
        feat, shp = net.encode(imgs)
        feat, _   = net.sampler(feat, mod)
        logits    = net.decode(feat, shp)
        t1, t3    = accuracy(logits, labs, (1, 3))
        bs        = imgs.size(0)
        acc1 += t1.item() / 100 * bs
        acc3 += t3.item() / 100 * bs
        samples += bs

        # ---------- buffer ----------
        buf.append(imgs)
        if len(buf) == interval:                   # time to update stats
            big_batch = torch.cat(buf, dim=0)      # B = bs × interval
            net.train()
            _ = net.encode(big_batch)              # single pass to refresh stats
            net.eval()
            buf.clear()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    ms = (time.time() - t0) / samples * 1e3
    return acc1 / samples * 100, acc3 / samples * 100, ms


# ------------------ helper: build fresh net ------------------ #
def build_net(cfg):
    """Fresh backbone + checkpoint (with original BN)."""
    net = (JSCC_model.DTJSCC_CIFAR10(3, cfg.latent_d, 10, cfg.num_embeddings)
           if cfg.dataset.lower().startswith('cifar10') else
           JSCC_model.DTJSCC_CINIC10(3, cfg.latent_d, 10, cfg.num_embeddings)
           if cfg.dataset.lower().startswith('cinic10') else
           JSCC_model.DTJSCC_CIFAR100(3, cfg.latent_d, 100, cfg.num_embeddings))

    ckpt = torch.load(cfg.model_path, map_location='cpu')
    net.load_state_dict(ckpt["model_states"])
    net.to(cfg.device)
    return net


# ------------------------------ main ------------------------- #
def main(cfg):
    device = cfg.device

    # fixed channel (SNR = 8 dB)
    ch_kw = {"K": cfg.K} if cfg.channel == 'rician' else {}
    if cfg.channel == 'nakagami':
        ch_kw['m'] = cfg.m
    mod = (QAM if cfg.mod == 'qam' else PSK)(cfg.num_embeddings,
                                             cfg.snr, cfg.channel, ch_kw)

    combined_results = {}

    # -------------------------------------------------
    # Sweep 1: batch size, fixed interval=10, m=10
    # -------------------------------------------------
    fixed_intv = 1
    fixed_m    = 10.0
    bs_list = [1, 2, 4, 8, 16, 32, 64, 256]

    print(f"\n>>> Sweep batch size (interval={fixed_intv}, m={fixed_m})")
    bs_results = {}
    for bs in bs_list:
        loader = get_data(cfg.dataset, bs, n_worker=0, train=False, shuffle=True)

        net = build_net(cfg)
        swap_to_ttn(net, fixed_m)

        a1, a3, ms = run(loader, net, mod, device, fixed_intv)
        bs_results[bs] = dict(acc1=a1, acc3=a3, ms=ms)
        print(f"[bs={bs:3d}]  Acc@1 {a1:6.2f}  Acc@3 {a3:6.2f}   {ms:5.2f} ms/img")

        del net
        torch.cuda.empty_cache()

    combined_results["sweep_batch_size"] = bs_results

    # -------------------------------------------------
    # Sweep 2: interval, fixed batch_size=64, m=10
    # -------------------------------------------------
    fixed_bs = 1
    intv_list = [1, 2, 5, 10, 20, 50, 100]

    print(f"\n>>> Sweep interval (batch size={fixed_bs}, m={fixed_m})")
    intv_results = {}
    loader = get_data(cfg.dataset, fixed_bs, n_worker=0, train=False, shuffle=True)

    for intv in intv_list:
        net = build_net(cfg)
        swap_to_ttn(net, fixed_m)

        a1, a3, ms = run(loader, net, mod, device, intv)
        intv_results[intv] = dict(acc1=a1, acc3=a3, ms=ms)
        print(f"[intv={intv:3d}]  Acc@1 {a1:6.2f}  Acc@3 {a3:6.2f}   {ms:5.2f} ms/img")

        del net
        torch.cuda.empty_cache()

    combined_results["sweep_interval"] = intv_results

    # -------------------------------------------------
    # Sweep 3: m, fixed batch_size=64, interval=10
    # -------------------------------------------------
    m_list = [1.0, 5.0, 10.0, 50.0, 100.0]
    fixed_bs = 2
    fixed_intv = 1
    print(f"\n>>> Sweep m (batch size={fixed_bs}, interval={fixed_intv})")
    m_results = {}
    loader = get_data(cfg.dataset, fixed_bs, n_worker=0, train=False, shuffle=True)

    for m in m_list:
        net = build_net(cfg)
        swap_to_ttn(net, m)

        a1, a3, ms = run(loader, net, mod, device, fixed_intv)
        m_results[m] = dict(acc1=a1, acc3=a3, ms=ms)
        print(f"[m={m:5.1f}]  Acc@1 {a1:6.2f}  Acc@3 {a3:6.2f}   {ms:5.2f} ms/img")

        del net
        torch.cuda.empty_cache()

    combined_results["sweep_m"] = m_results

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir,
                             f"ttn_sweep_{cfg.dataset}_{cfg.channel}.pt")
    torch.save(combined_results, save_path)
    print(f"\n[Done] all sweep results saved to: {save_path}")




# ---------------------------- CLI ---------------------------- #
if __name__ == "__main__":
    import argparse
    P = argparse.ArgumentParser("TT-Norm batch-size / interval sweep")

    # data / model
    P.add_argument("--dataset",          default="CIFAR10")
    P.add_argument("--num_embeddings",   type=int, default=16)
    P.add_argument("--latent_d",         type=int, default=512)
    P.add_argument("--num_latent",       type=int, default=4)
    P.add_argument("--model_dir",        default="JT-JSCC/trained_models")
    P.add_argument("--model_file",       default="best.pt")

    # channel / modulation
    P.add_argument("--mod",              choices=["psk", "qam"], default="psk")
    P.add_argument("--channel",          choices=["awgn", "rayleigh", "rician", "nakagami"],
                                          default="rayleigh")
    P.add_argument("--K",                type=float, default=3.0)
    P.add_argument("--m",                type=float, default=2.0)
    P.add_argument("--snr",              type=float, default=8.0)



    # misc
    P.add_argument("--save_dir",         default="AdapS/eval")
    P.add_argument("--device",           default="cuda:0")
    args = P.parse_args()

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.model_path = os.path.join(args.model_dir, args.model_file)

    main(args)
