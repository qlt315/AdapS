# mascn_gan_eval.py
# Evaluate MASCN-style GAN + JSCC pipeline (no parameter updates):
#   (target image) --G--> (source-style image) --JSCC--> logits -> accuracy
#
# Key points:
# - DataLoader outputs are in [0,1] for CIFAR; we map to [-1,1] before G, then back to [0,1] for JSCC.
# - We ONLY convert target-domain images to source style; source-domain inputs can be bypassed.
# - No training anywhere: model.eval(), G.eval(), torch.no_grad().
# - Optional: --bypass_g 直接跳过 G 做对照； --dump_samples 保存若干可视化样例。

import os, re, torch, torch.nn as nn
from typing import List, Tuple, Dict
from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy
import torchvision.utils as vutils

# ============================ Regex helpers ================================ #
_DATASET_BASE_RE = re.compile(r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=re.IGNORECASE)

def detect_base(dataset_name: str) -> str:
    ds = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(ds)
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).upper()

def infer_num_classes(dataset_name: str) -> int:
    base = detect_base(dataset_name)
    return 100 if base == "CIFAR100" else 10

# ============================= GAN modules ================================= #
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )
    def forward(self, x): return x + self.net(x)

class Generator(nn.Module):
    """
    ResNet encoder-decoder with spatial conditional map.
    Input:  x in [-1,1], c_map one-hot (B,n_domains,H,W)
    Output: y in [-1,1]
    """
    def __init__(self, in_c=3, base=64, n_res=4, n_domains=2):
        super().__init__()
        cin = in_c + n_domains
        self.down = nn.Sequential(
            nn.Conv2d(cin, base, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(base, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base*2, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base*4, affine=True), nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(base*4) for _ in range(n_res)])
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base*2, affine=True), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base*2, base, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(base, in_c, 7, 1, 3),
            nn.Tanh(),
        )
    def forward(self, x, c_map):
        z = torch.cat([x, c_map], dim=1)
        h = self.down(z)
        h = self.body(h)
        y = self.up(h)
        return y

# ================================ Utils ==================================== #
def onehot_map(domain_id: int, B: int, H: int, W: int, n_domains: int, device) -> torch.Tensor:
    t = torch.zeros(B, n_domains, H, W, device=device, dtype=torch.float32)
    t[:, domain_id, :, :] = 1.0
    return t

def build_mod(mod_type: str, n_emb: int, psnr: int, channel: str, ch_args: dict):
    cls = QAM if str(mod_type).lower() == "qam" else PSK
    return cls(n_emb, psnr, channel, ch_args)

def to_m11(x01: torch.Tensor) -> torch.Tensor:   # [0,1] -> [-1,1]
    return x01 * 2.0 - 1.0

def to_01(x11: torch.Tensor) -> torch.Tensor:    # [-1,1] -> [0,1]
    return (x11 + 1.0) * 0.5

def guess_is_source(domain_tags: List[str], ds_name: str, ch_name: str) -> bool:
    """Return True if (ds_name, ch_name) matches the source tag in ckpt, else False."""
    if not domain_tags:  # unknown, assume target
        return False
    src_tag = domain_tags[0]  # trainer约定：第0个是源域
    want = f"SRC:{ds_name}|{ch_name}"
    return (src_tag == want)

def maybe_dump_samples(root_dir, psnr, imgs_in01, imgs_conv01, count=8):
    os.makedirs(root_dir, exist_ok=True)
    grid_in  = vutils.make_grid(imgs_in01[:count],  nrow=4, normalize=False)
    grid_out = vutils.make_grid(imgs_conv01[:count], nrow=4, normalize=False)
    vutils.save_image(grid_in,  os.path.join(root_dir, f"samples_psnr{psnr:02d}_in.png"))
    vutils.save_image(grid_out, os.path.join(root_dir, f"samples_psnr{psnr:02d}_conv.png"))

# ================================ Eval ===================================== #
def eval_one_psnr(val_loader, model, G, domain_tags, SRC_ID, n_domains,
                  assume_loader_range01, convert_this_domain: bool,
                  mod, device, clamp_out=True, dump_dir=None, dump_n=8):
    """
    For each batch:
      If convert_this_domain:
          X[0,1] -> [-1,1] -> G(X, c=SRC) -> [-1,1] -> [0,1] -> JSCC
      else:
          X[0,1] -> (bypass G) -> JSCC
    """
    model.eval(); G.eval()
    acc1 = acc3 = 0.0
    with torch.no_grad():
        first_dumped = False
        for imgs, labs in val_loader:
            imgs = imgs.to(device); labs = labs.to(device)
            if convert_this_domain:
                B, _, H, W = imgs.shape
                c_src = onehot_map(SRC_ID, B, H, W, n_domains, device)
                x_in = to_m11(imgs) if assume_loader_range01 else imgs
                x_conv = G(x_in, c_src)
                x_conv01 = to_01(x_conv) if assume_loader_range01 else x_conv
                if clamp_out: x_conv01 = x_conv01.clamp_(0.0, 1.0)
                logits, _ = model(x_conv01, mod=mod)
                if (dump_dir is not None) and (not first_dumped):
                    maybe_dump_samples(dump_dir, mod.snr if hasattr(mod, 'snr') else -1, imgs, x_conv01, dump_n)
                    first_dumped = True
            else:
                logits, _ = model(imgs, mod=mod)

            a1, a3 = accuracy(logits, labs, (1, 3))
            acc1 += a1.item(); acc3 += a3.item()

    acc1 /= len(val_loader); acc3 /= len(val_loader)
    return acc1, acc3

# ================================= Main ==================================== #
def main(args):
    print(f"\n[MASCN-Eval] Dataset={args.dataset} | Mod={args.mod.upper()} | Channel={args.channel.upper()}\n")

    # -------- 1) Build & load JSCC (frozen) --------
    base = detect_base(args.dataset)
    num_classes = infer_num_classes(args.dataset)
    if base == "CIFAR10":
        Model = JSCC_model.DTJSCC_CIFAR10
    elif base == "CIFAR100":
        Model = JSCC_model.DTJSCC_CIFAR100
    else:
        Model = JSCC_model.DTJSCC_CINIC10

    model = Model(args.in_channels, args.latent_d, num_classes, num_embeddings=args.num_embeddings)
    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_states"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # -------- 2) Load G (frozen) --------
    g_ckpt = torch.load(args.g_ckpt, map_location="cpu")
    domain_tags: List[str] = g_ckpt.get("domain_tags", [])
    if domain_tags:
        print("[G] Domains in ckpt (order):")
        for i, t in enumerate(domain_tags):
            print(f"  {i}: {t}")
    else:
        print("[Warn] 'domain_tags' missing in G checkpoint; assuming n_domains=2, SRC_ID=0.")
    n_domains = len(domain_tags) if domain_tags else 2
    SRC_ID = 0

    G = Generator(in_c=3, base=args.g_ch, n_res=4, n_domains=n_domains)
    G.load_state_dict(g_ckpt["G"] if "G" in g_ckpt else g_ckpt, strict=True)
    G.to(device).eval()

    # -------- 3) Dataloader (your target dataset variant) --------
    val_loader = get_data(args.dataset, args.bs, n_worker=0, train=False)

    # 是否把当前评测数据当作“目标域需要转换”
    convert_this_domain = not args.bypass_g
    # 如果 ckpt 里包含域标签，且当前数据刚好是源域，就强制旁路（更稳）
    if domain_tags and guess_is_source(domain_tags, args.dataset, args.channel):
        print("[Info] Current (dataset,channel) matches SRC in ckpt; bypass G for stability.")
        convert_this_domain = False

    # -------- 4) PSNR sweep --------
    acc1s, acc3s = [], []
    dump_dir = args.dump_dir if args.dump_dir else None

    for psnr in range(0, 26):
        ch_args = {"K": args.K} if args.channel == "rician" else {"m": args.m} if args.channel == "nakagami" else {}
        mod = build_mod(args.mod, args.num_embeddings, psnr, args.channel, ch_args)

        a1, a3 = eval_one_psnr(
            val_loader, model, G, domain_tags, SRC_ID, n_domains,
            assume_loader_range01=True,                     # CIFAR loader -> [0,1]
            convert_this_domain=convert_this_domain,
            mod=mod, device=device, clamp_out=True,
            dump_dir=(os.path.join(args.save_dir, "samples") if dump_dir else None),
            dump_n=args.dump_n
        )
        print(f"[Eval] PSNR={psnr:2d}dB | Acc@1={a1:.4f} | Acc@3={a3:.4f}")
        acc1s.append(a1); acc3s.append(a3)

    # -------- 5) Save results --------
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"mascn-eval-{args.dataset}-{args.channel}.pt")
    torch.save({"acc1s": acc1s, "acc3s": acc3s}, out_path)
    print(f"\n[Done] Saved: {out_path}")

# ================================== CLI ==================================== #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("MASCN Eval: GAN pre-conversion + JSCC (frozen)")
    # Data / model
    p.add_argument("--dataset", default="CIFAR10_noise", help="TARGET dataset variant, e.g., CIFAR10_noise")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--latent_d",   type=int, default=512)
    p.add_argument("--num_embeddings", type=int, default=16)
    # Paths
    p.add_argument("--model_path", default="JT-JSCC/trained_models/cifar10_best.pt",
                   help="Pretrained JSCC checkpoint (dict with 'model_states').")
    p.add_argument("--g_ckpt", default="AdapS/trained_models/CIFAR10_mascn_g_best.pt",
                   help="Trained generator checkpoint from mascn_gan_train.py")
    p.add_argument("--save_dir", default="AdapS/eval")
    # Channel / modulation
    p.add_argument("--mod", choices=["psk","qam"], default="psk")
    p.add_argument("--channel", choices=["awgn","rayleigh","rician","nakagami"], default="awgn")
    p.add_argument("--K", type=float, default=3.0)
    p.add_argument("--m", type=float, default=2.0)
    # GAN config
    p.add_argument("--g_ch", type=int, default=64, help="Generator base channels (must match training).")
    p.add_argument("--bypass_g", action="store_true", help="Bypass GAN for ablation / sanity check.")
    # Visualization
    p.add_argument("--dump_dir", type=str, default="", help="If set, dump a few input/converted samples per PSNR.")
    p.add_argument("--dump_n", type=int, default=8)
    # Eval
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)
