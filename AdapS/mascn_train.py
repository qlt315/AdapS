# mascn_gan_train.py
# Train the MASCN-style domain converter (GAN) that runs BEFORE the JSCC encoder.
# Task format: a list of (dataset_variant, channel) pairs.
# - The FIRST task is the SOURCE domain (e.g., (base, "awgn")).
# - Remaining tasks are TARGET domains (e.g., (f"{base}_noise", "awgn")).
# Channel tag is only for bookkeeping; the image GAN itself is channel-agnostic.

import os, re, random, torch, torch.nn as nn, torch.optim as optim
from typing import List, Tuple, Dict
from datasets.dataloader import get_data

# --------------------------- Regex dataset helpers ------------------------- #
_DATASET_BASE_RE = re.compile(r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=re.IGNORECASE)

def detect_base(dataset_name: str) -> str:
    """Return canonical base name in UPPER via regex ('CIFAR10'/'CIFAR100'/'CINIC10')."""
    ds = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(ds)
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).upper()

def swap_base_token(s: str, base_upper: str) -> str:
    """Replace the placeholder token 'CIFAR10' (exact case) inside strings."""
    return s.replace("CIFAR10", base_upper)

# --------------------------- Task list (template) -------------------------- #
# Use CIFAR10 as a placeholder; we rewrite it to CIFAR100/CINIC10 if needed.
# Each entry is (dataset_variant, channel).
# IMPORTANT: The FIRST entry is used as the SOURCE domain.
TASKS_TEMPLATE: List[Tuple[str, str]] = [
    ("CIFAR10",        "awgn"),   # SOURCE (clean base)
    ("CIFAR10_noise",  "awgn"),   # TARGET (noisy variant, same channel)
    # You can add more targets if needed, e.g. ("CIFAR10", "rician")
]

def expand_tasks_for_base(base_upper: str) -> List[Tuple[str, str]]:
    """Swap 'CIFAR10' to the chosen base in TASKS_TEMPLATE."""
    return [(swap_base_token(ds, base_upper), ch) for ds, ch in TASKS_TEMPLATE]

# ------------------------------- Small GAN -------------------------------- #
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
    ResNet encoder-decoder with conditional injection (domain one-hot map).
    Input:  x (B,3,H,W), c_map (B,n_domains,H,W)
    Output: fake_src (B,3,H,W) converted toward the SOURCE domain
    Note: outputs are in [-1, 1] due to final Tanh.
    """
    def __init__(self, in_c=3, base=64, n_res=4, n_domains=2):
        super().__init__()
        self.n_domains = n_domains
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
            nn.Tanh(),  # -> [-1, 1]
        )
    def forward(self, x, c_map):
        z = torch.cat([x, c_map], dim=1)
        h = self.down(z)
        h = self.body(h)
        y = self.up(h)
        return y

class MultiHeadDiscriminator(nn.Module):
    """
    PatchGAN discriminator with:
    - adversarial head: real/fake (hinge loss)
    - domain classification head: predicts one of N domain IDs (SRC + K targets)
    """
    def __init__(self, in_c=3, base=64, n_domains=2):
        super().__init__()
        ch = base
        layers = [nn.Conv2d(in_c, ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(3):
            layers += [
                nn.Conv2d(ch, ch*2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(ch*2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch *= 2
        self.feat = nn.Sequential(*layers)
        self.adv = nn.Conv2d(ch, 1, 1)             # patch logits
        self.cls = nn.Conv2d(ch, n_domains, 1)     # per-patch domain logits

    def forward(self, x):
        h = self.feat(x)
        adv = self.adv(h).mean(dim=(2,3))          # [B]
        dom = self.cls(h).mean(dim=(2,3))          # [B, n_domains]
        return adv, dom

# ---------------------------- Utils & Losses ------------------------------- #
def onehot_map(domain_id: int, B: int, H: int, W: int, n_domains: int, device) -> torch.Tensor:
    """Spatial one-hot conditioning map (B, n_domains, H, W)."""
    t = torch.zeros(B, n_domains, H, W, device=device, dtype=torch.float32)
    t[:, domain_id, :, :] = 1.0
    return t

def hinge_g_loss(fake_logits):
    """Generator hinge loss: encourage D(fake) to be large (real)."""
    return -fake_logits.mean()

# R1 gradient penalty on REAL samples (StyleGAN2)
def r1_penalty(d_out: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
    """R1 regularization: ||∇_x D(x)||^2 on REAL inputs."""
    grads = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grads.view(grads.size(0), -1).pow(2).sum(dim=1).mean()

# Range helpers: dataloader provides tensors in [0,1]; G/D expect [-1,1].
def to_m11(x):  # [0,1] -> [-1,1]
    return x * 2.0 - 1.0

# --------------------------------- Train ---------------------------------- #
def main(args):
    # 1) Build tasks with the requested base dataset
    base_upper = detect_base(args.base_dataset)    # "CIFAR10"/"CIFAR100"/"CINIC10"
    tasks: List[Tuple[str, str]] = expand_tasks_for_base(base_upper)
    assert len(tasks) >= 2, "Need at least 1 source + 1 target task."
    # FIRST task is SOURCE by convention:
    src_dataset, src_channel = tasks[0]
    tgt_tasks = tasks[1:]

    # Domain tags aligned with tasks order, for domain classification head
    domain_tags = [f"SRC:{src_dataset}|{src_channel}"] + [
        f"TGT:{ds}|{ch}" for (ds, ch) in tgt_tasks
    ]
    n_domains = len(domain_tags)
    SRC_ID = 0
    tag2id: Dict[str, int] = {t: i for i, t in enumerate(domain_tags)}

    print("\n[Task list]")
    for i, (ds, ch) in enumerate(tasks):
        tag = domain_tags[i]
        print(f"  - {tag:>30s}  (dataset='{ds}', channel='{ch}')")
    print(f"\n[Config] Base = {base_upper} | #Domains = {n_domains} (1 source + {n_domains-1} targets)\n")

    # 2) Dataloaders for each (dataset, channel) pair (channel unused by GAN)
    loaders: Dict[str, torch.utils.data.DataLoader] = {}
    iters:   Dict[str, any] = {}
    for i, (ds, ch) in enumerate(tasks):
        tag = domain_tags[i]
        ld = get_data(ds, args.bs, n_worker=0, train=True)  # returns images in [0,1]
        loaders[tag] = ld
        iters[tag]   = iter(ld)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 3) Models
    G = Generator(in_c=3, base=args.g_ch, n_res=4, n_domains=n_domains).to(device)
    D = MultiHeadDiscriminator(in_c=3, base=args.d_ch, n_domains=n_domains).to(device)

    # 4) Optimizers (TTUR: slower D, faster G)
    opt_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.5, 0.999))

    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()

    out_dir = os.path.join("AdapS", "trained_models")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{base_upper}_mascn_g_best.pt")

    # 5) Training loop
    global_step = 0
    epochs = args.epochs
    src_tag = domain_tags[0]
    tgt_tags = domain_tags[1:]

    for ep in range(epochs):
        # iterate roughly the length of the source loader
        for _ in range(len(loaders[src_tag])):
            # ----- Source batch -----
            try:
                src_imgs, _ = next(iters[src_tag])
            except StopIteration:
                iters[src_tag] = iter(loaders[src_tag])
                src_imgs, _ = next(iters[src_tag])

            # ----- Random target domain batch -----
            tgt_tag = random.choice(tgt_tags)
            try:
                tgt_imgs, _ = next(iters[tgt_tag])
            except StopIteration:
                iters[tgt_tag] = iter(loaders[tgt_tag])
                tgt_imgs, _ = next(iters[tgt_tag])

            # Map real images from [0,1] -> [-1,1] to match G/D domain.
            src_imgs = to_m11(src_imgs.to(device))
            tgt_imgs = to_m11(tgt_imgs.to(device))

            B, _, H, W = tgt_imgs.shape
            c_src = onehot_map(SRC_ID, B, H, W, n_domains, device)

            # ------------------ 1) Update D (one step) ------------------ #
            D.train(); G.train()
            opt_D.zero_grad(set_to_none=True)

            # Real source: adv real + domain=SRC  (enable grad for R1)
            src_imgs_req = src_imgs.detach().requires_grad_(args.r1_gamma > 0.0)
            real_logit_src, real_dom_src = D(src_imgs_req)
            # hinge real: relu(1 - D(x_real))
            loss_D_real_src = torch.relu(1.0 - real_logit_src).mean()
            loss_D_dom_src  = ce(real_dom_src, torch.full((src_imgs.size(0),), SRC_ID, device=device, dtype=torch.long))

            # Real target: adv real + domain=target_id
            tgt_id = tag2id[tgt_tag]
            real_logit_tgt, real_dom_tgt = D(tgt_imgs.detach())
            loss_D_real_tgt = torch.relu(1.0 - real_logit_tgt).mean()
            loss_D_dom_tgt  = ce(real_dom_tgt, torch.full((tgt_imgs.size(0),), tgt_id, device=device, dtype=torch.long))

            # Fake-as-source (from target): adv fake
            with torch.no_grad():
                fake_src = G(tgt_imgs, c_src)  # in [-1,1]
            fake_logit_src, _ = D(fake_src)
            # hinge fake: relu(1 + D(x_fake))
            loss_D_fake = torch.relu(1.0 + fake_logit_src).mean()

            # Adversarial + domain classification heads
            loss_D_adv = loss_D_real_src + loss_D_real_tgt + loss_D_fake
            loss_D_cls = loss_D_dom_src + loss_D_dom_tgt
            loss_D = loss_D_adv + args.lambda_dom * loss_D_cls

            # R1 on real source only (cheap and effective)
            if args.r1_gamma > 0.0:
                r1 = r1_penalty(real_logit_src, src_imgs_req)
                loss_D = loss_D + 0.5 * args.r1_gamma * r1  # StyleGAN2 convention

            loss_D.backward()
            opt_D.step()

            # ------------------ 2) Update G (g_steps times) -------------- #
            # G tries to fool D and align domain label to SRC; also identity loss on SRC.
            for _ in range(args.g_steps):
                opt_G.zero_grad(set_to_none=True)

                fake_src = G(tgt_imgs, c_src)       # [-1,1]
                fake_logit, fake_dom = D(fake_src)

                # Adversarial: fool D as real
                loss_G_adv = hinge_g_loss(fake_logit)
                # Domain classification: push fake → SRC id
                loss_G_dom = ce(fake_dom, torch.full((B,), SRC_ID, device=device, dtype=torch.long))
                # Identity/content: source should map to itself under SRC condition
                c_id = onehot_map(SRC_ID, src_imgs.size(0), src_imgs.size(2), src_imgs.size(3), n_domains, device)
                id_out = G(src_imgs, c_id)          # both in [-1,1]
                loss_G_id = l1(id_out, src_imgs)

                loss_G = loss_G_adv + args.lambda_dom * loss_G_dom + args.lambda_id * loss_G_id
                loss_G.backward()
                opt_G.step()

            # ------------------ 3) Optional: extra G step if D too strong -- #
            if args.extra_g_when_d_small and loss_D_adv.item() < args.d_small_thr:
                opt_G.zero_grad(set_to_none=True)
                fake_src = G(tgt_imgs, c_src)
                fake_logit, fake_dom = D(fake_src)
                loss_G_adv = hinge_g_loss(fake_logit)
                loss_G_dom = ce(fake_dom, torch.full((B,), SRC_ID, device=device, dtype=torch.long))
                c_id = onehot_map(SRC_ID, src_imgs.size(0), src_imgs.size(2), src_imgs.size(3), n_domains, device)
                id_out = G(src_imgs, c_id)
                loss_G_id = l1(id_out, src_imgs)
                loss_G = loss_G_adv + args.lambda_dom * loss_G_dom + args.lambda_id * loss_G_id
                loss_G.backward()
                opt_G.step()

            global_step += 1
            if global_step % 200 == 0:
                print(f"[E{ep}] step {global_step} | "
                      f"D_adv {loss_D_adv.item():.3f} D_dom(SRC) {loss_D_dom_src.item():.3f} D_dom(TGT) {loss_D_dom_tgt.item():.3f} "
                      f"| G_adv {loss_G_adv.item():.3f} G_dom {loss_G_dom.item():.3f} G_id {loss_G_id.item():.3f}")

        # Save G each epoch
        torch.save({
            "G": G.state_dict(),
            "domain_tags": domain_tags,   # order matters
            "tasks": tasks,               # [(dataset_variant, channel), ...]
        }, ckpt_path)
        print(f"[CKPT] Saved generator to {ckpt_path}")

    print("\n[Done] Training finished.")
    print(f"Generator checkpoint: {ckpt_path}")
    print(f"Domains (order): {domain_tags}")
    print("Use this G in eval: convert incoming (dataset,channel) target images to source style before JSCC.")

# --------------------------------- CLI ------------------------------------ #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("MASCN GAN Trainer (tasks: (dataset_variant, channel))")
    ap.add_argument("--base_dataset", type=str, default="CIFAR10",
                    help="One of CIFAR10/CIFAR100/CINIC10; TASKS_TEMPLATE will be rewritten accordingly.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)

    # ---- separate LR for G/D; make D slower, G faster (TTUR) ----
    ap.add_argument("--lr_G", type=float, default=5e-4)
    ap.add_argument("--lr_D", type=float, default=2e-5)

    # ---- step ratio and stabilization ----
    ap.add_argument("--g_steps", type=int, default=2, help="How many G updates per D update.")
    ap.add_argument("--r1_gamma", type=float, default=5.0, help="R1 penalty weight on real samples (0 to disable).")
    ap.add_argument("--extra_g_when_d_small", action="store_true",
                    help="If set, give G one extra step when D_adv < d_small_thr.")
    ap.add_argument("--d_small_thr", type=float, default=0.1,
                    help="Threshold on D_adv to trigger extra G step.")

    ap.add_argument("--g_ch", type=int, default=64)
    ap.add_argument("--d_ch", type=int, default=64)
    ap.add_argument("--lambda_dom", type=float, default=1.0, help="Weight of domain classification loss.")
    ap.add_argument("--lambda_id",  type=float, default=10.0, help="Weight of identity/content loss on SRC images.")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()
    main(args)
