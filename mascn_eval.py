# mascn_eval.py
# Evaluation with MASCN: insert a pre-trained generator BEFORE the JSCC encoder.
# Pipeline: x_tgt --G(·; domain=SRC)--> x_src_like --JSCC--> logits
# We reuse your modulation/channel/eval loop; no BN stat updates are introduced here.

import os, re, torch
import torch.nn as nn
from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy

# ----------------------------- regex helpers ------------------------------ #
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

# ----------------------------- MASCN modules ------------------------------ #
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        return self.relu(x + y)

def onehot_map(domain_id: int, B: int, H: int, W: int, n_domains: int, device):
    t = torch.zeros(B, n_domains, H, W, device=device, dtype=torch.float32)
    t[:, domain_id, :, :] = 1.0
    return t

class Generator(nn.Module):
    """Same as in training."""
    def __init__(self, in_c=3, base=64, n_domains=2):
        super().__init__()
        self.n_domains = n_domains
        cin = in_c + n_domains
        self.enc = nn.Sequential(
            nn.Conv2d(cin, base, 7, 1, 3), nn.ReLU(True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.ReLU(True),
        )
        self.res = nn.Sequential(
            ResBlock(base*4), ResBlock(base*4), ResBlock(base*4), ResBlock(base*4)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base*2, base,   4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base, in_c, 7, 1, 3), nn.Tanh()
        )
    def forward(self, x, c_onehot):
        z = torch.cat([x, c_onehot], dim=1)
        y = self.enc(z)
        y = self.res(y)
        y = self.dec(y)
        return y

# ------------------------------- eval util -------------------------------- #
def eval_one_psnr(val_loader, model, gen, device, psnr, mod, src_domain_id=0):
    model.eval(); gen.eval()
    acc1 = acc3 = 0.0
    with torch.no_grad():
        for imgs, labs in val_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            B, _, H, W = imgs.shape
            c_src = onehot_map(src_domain_id, B, H, W, gen.n_domains, device)
            imgs_conv = gen(imgs, c_src)                  # convert to SRC domain
            # JSCC forward (single decoder)
            logits, _ = model(imgs_conv, mod=mod)
            a1, a3 = accuracy(logits, labs, (1, 3))
            acc1 += a1.item(); acc3 += a3.item()
    acc1 /= len(val_loader); acc3 /= len(val_loader)
    print(f"[Eval] PSNR={psnr:2d}dB | Acc@1={acc1:.4f} | Acc@3={acc3:.4f}")
    return acc1, acc3

# ---------------------------------- main ---------------------------------- #
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = detect_base(args.dataset_tgt)
    num_classes = infer_num_classes(args.dataset_tgt)

    # ---- JSCC model (single decoder; without task_key) ----
    if base == "CIFAR10":
        Model = JSCC_model.DTJSCC_CIFAR10
    elif base == "CIFAR100":
        Model = JSCC_model.DTJSCC_CIFAR100
    else:
        Model = JSCC_model.DTJSCC_CINIC10

    model = Model(args.in_channels, args.latent_d, num_classes,
                  num_embeddings=args.num_embeddings)
    ckpt = torch.load(args.jscc_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_states"])
    model.to(device)

    # ---- load MASCN generator ----
    gen = Generator(in_c=3, base=args.g_ch, n_domains=2).to(device)
    gckpt = torch.load(args.g_ckpt, map_location="cpu")
    gen.load_state_dict(gckpt["G"])
    gen.to(device)
    gen.eval()

    # ---- data (target stream) ----
    val_loader = get_data(args.dataset_tgt, args.bs, n_worker=0, train=False)

    # ---- PSNR sweep ----
    acc1s, acc3s = [], []
    for psnr in range(0, 26):
        ch_args = {"K": args.K} if args.channel == "rician" else \
                  {"m": args.m} if args.channel == "nakagami" else {}
        mod = (QAM if args.mod == "qam" else PSK)(
            args.num_embeddings, psnr, args.channel, ch_args
        )
        a1, a3 = eval_one_psnr(val_loader, model, gen, device, psnr, mod, src_domain_id=0)
        acc1s.append(a1); acc3s.append(a3)

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"mascn-eval-{args.dataset_tgt}-{args.channel}.pt")
    torch.save({"acc1s": acc1s, "acc3s": acc3s}, out_path)
    print(f"\n[Done] Saved: {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("MASCN eval (JSCC + pre-GAN)")
    # Data / domains
    ap.add_argument("--dataset_tgt", default="CIFAR10_noise",
                    help="target stream (test-time) domain; images will be converted to SRC")
    # JSCC
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--latent_d",   type=int, default=512)
    ap.add_argument("--num_embeddings", type=int, default=16)
    ap.add_argument("--jscc_ckpt", default="JSCC/trained_models/CIFAR10-awgn/best.pt")
    # MASCN generator
    ap.add_argument("--g_ckpt", default="MASCN-GAN/trained_models/mascn_g_best.pt")
    ap.add_argument("--g_ch", type=int, default=64)
    # Channel / modulation
    ap.add_argument("--mod", choices=["psk","qam"], default="psk")
    ap.add_argument("--channel", choices=["awgn","rayleigh","rician","nakagami"], default="awgn")
    ap.add_argument("--K", type=float, default=3.0)
    ap.add_argument("--m", type=float, default=2.0)
    # Eval
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--save_dir", default="AdapS/eval")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    main(args)
