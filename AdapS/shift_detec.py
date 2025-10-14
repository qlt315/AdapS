import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from scipy.io import savemat
from datasets.dataloader import get_data
from model.DT_JSCC import DTJSCC_CIFAR10
from utils.modulation import QAM
import random


def symmetric_kl_distance(mean1, std1, mean2, std2, eps=1e-6):
    if isinstance(mean1, np.ndarray): mean1 = torch.from_numpy(mean1)
    if isinstance(std1, np.ndarray):  std1  = torch.from_numpy(std1)
    if isinstance(mean2, np.ndarray): mean2 = torch.from_numpy(mean2)
    if isinstance(std2, np.ndarray):  std2  = torch.from_numpy(std2)
    var1 = std1**2 + eps
    var2 = std2**2 + eps
    d1 = (var1 + (mean1 - mean2)**2) / (2.0 * var2) - 0.5
    d2 = (var2 + (mean1 - mean2)**2) / (2.0 * var1) - 0.5
    return torch.mean(d1 + d2).item()


def l2_distance(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    return F.mse_loss(a, b).item()


def wasserstein_gaussian_distance(mean1, std1, mean2, std2, eps=1e-8):
    """
    W2 distance between 1D Gaussians for each channel:
        W2^2 = (mu1 - mu2)^2 + (sigma1 - sigma2)^2
    We return the average W2 over channels.
    """
    if isinstance(mean1, np.ndarray): mean1 = torch.from_numpy(mean1)
    if isinstance(std1,  np.ndarray): std1  = torch.from_numpy(std1)
    if isinstance(mean2, np.ndarray): mean2 = torch.from_numpy(mean2)
    if isinstance(std2,  np.ndarray): std2  = torch.from_numpy(std2)

    # ensure same dtype
    mean1 = mean1.float(); std1 = std1.float()
    mean2 = mean2.float(); std2 = std2.float()

    # small epsilon to avoid degenerate std
    std1 = torch.clamp(std1, min=eps)
    std2 = torch.clamp(std2, min=eps)

    w2_per_ch = torch.sqrt((mean1 - mean2) ** 2 + (std1 - std2) ** 2)  # [C]
    return w2_per_ch.mean().item()


def detect_image_shift(imgs, model, ref_img_stats, config):
    with torch.no_grad():
        # use stem conv BEFORE BN for style-sensitive stats
        stem_feat = model.prep[0](imgs)
        mean = stem_feat.mean(dim=(0, 2, 3)).cpu().numpy()  # [C]
        std  = stem_feat.std(dim=(0, 2, 3)).cpu().numpy()   # [C]

    ref_mean, ref_std = ref_img_stats

    metric = config.get('img_metric', 'kl').lower()
    if metric == 'kl':
        score = symmetric_kl_distance(mean, std, ref_mean, ref_std)
    elif metric == 'l2':
        score = l2_distance(mean, ref_mean) + l2_distance(std, ref_std)
    elif metric == 'wasserstein':
        # W2 on per-channel Gaussian summaries
        score = wasserstein_gaussian_distance(mean, std, ref_mean, ref_std)
    else:
        raise ValueError(f"Unsupported image metric: {metric}")

    return (score > config['thresh_img']), score, (mean, std), stem_feat


def detect_channel_shift(mod, latent_batch, ref_stats, metric='wasserstein'):
    mod.channel_estimates = []
    mod.apply_channel(latent_batch)
    est = torch.cat(mod.channel_estimates, dim=0)  # (B, 2)
    h_mag = torch.norm(est, dim=1).cpu().numpy()
    h_norm = (h_mag - h_mag.mean()) / (h_mag.std() + 1e-6)
    hist, _ = np.histogram(h_norm, bins=50, range=(-5, 5), density=True)
    ref_stats = np.array(ref_stats).flatten()

    if metric == 'wasserstein':
        dist = wasserstein_distance(hist, ref_stats)
    elif metric == 'kl':
        epsilon = 1e-8
        dist = entropy(hist + epsilon, ref_stats + epsilon)
    elif metric == 'l2':
        dist = np.linalg.norm(hist - ref_stats)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return dist


def detect_dual_domain_shift(imgs, model, mod, ref_img_stats, ref_ch_stats, config):
    # image shift
    shift_img, shift_img_score, cur_img_stats, _ = detect_image_shift(imgs, model, ref_img_stats, config)
    # channel shift
    with torch.no_grad():
        z_enc = model.encoder(imgs)                 # (B, 512, H, W)
        z_flat = z_enc.view(z_enc.size(0), -1)      # (B, 512*H*W)
    shift_ch_score = detect_channel_shift(mod, z_flat, ref_ch_stats, config['ch_metric'])
    shift_ch = shift_ch_score > config['thresh_ch']

    print(f"[Scores] image: {shift_img_score:.4f} (th={config['thresh_img']}) | "
          f"channel: {shift_ch_score:.4f} (th={config['thresh_ch']})")

    shift_flag = shift_img or shift_ch
    if shift_flag:
        print("[SHIFT DETECTED]")
    return shift_flag, cur_img_stats, z_flat


def sample_random_batch(loader, batch_size: int):
    """Randomly sample (with replacement) from loader.dataset."""
    ds = loader.dataset
    idxs = np.random.randint(0, len(ds), size=batch_size)
    batch = [ds[i] for i in idxs]
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels0 = batch[0][1]
    if isinstance(labels0, torch.Tensor):
        labels = torch.stack([b[1] for b in batch], dim=0)
    else:
        labels = torch.tensor([b[1] for b in batch])
    return imgs, labels


def run_sim_for_batch_size(model, config, batch_size: int, sim_steps: int):
    """Run a full simulation for a given batch size and return accuracy."""
    source_domain = ('CIFAR10', 'awgn')
    shift_domains = [
        ('CIFAR10_noise', 'awgn'),
        ('CIFAR10_noise', 'rician'),
        ('CIFAR10_noise', 'rayleigh'),

        ('CIFAR10_fog', 'awgn'),
        ('CIFAR10_fog', 'rician'),
        ('CIFAR10_fog', 'rayleigh'),

        ('CIFAR10_bright', 'awgn'),
        ('CIFAR10_bright', 'rician'),
        ('CIFAR10_bright', 'rayleigh'),

        ('CIFAR10_snow', 'awgn'),
        ('CIFAR10_snow', 'rician'),
        ('CIFAR10_snow', 'rayleigh'),

        ('CIFAR10_rain', 'awgn'),
        ('CIFAR10_rain', 'rician'),
        ('CIFAR10_rain', 'rayleigh'),

        ('CIFAR10_blur', 'awgn'),
        ('CIFAR10_blur', 'rician'),
        ('CIFAR10_blur', 'rayleigh'),
    ]

    # build domain sequence
    domain_sequence = []
    while len(domain_sequence) < sim_steps:
        if not domain_sequence:
            domain_sequence.extend([source_domain] * config['batches_per_domain'])
        else:
            domain_sequence.extend([random.choice(shift_domains)] * config['batches_per_domain'])
    domain_sequence = domain_sequence[:sim_steps]

    # make loaders just to access .dataset
    datasets_all = {source_domain[0]} | {ds for ds, _ in shift_domains}
    loaders = {
        ds: get_data(ds, batch_size=1, n_worker=4, train=False)
        for ds in datasets_all
    }

    correct, total = 0, 0
    ref_img_stats, ref_ch_stats = None, None
    device = config['device']

    for step in range(sim_steps):
        dataset_name, channel_type = domain_sequence[step]
        print(f"\n[Batch {step}] bs={batch_size} | Dataset={dataset_name}, Channel={channel_type}")

        imgs, _ = sample_random_batch(loaders[dataset_name], batch_size)
        imgs = imgs.to(device)

        ch_args = {}
        if channel_type == 'rician':
            ch_args['K'] = 3.0
        elif channel_type == 'nakagami':
            ch_args['m'] = 2.0
        mod = QAM(config['num_embeddings'], config['snr'], channel_type=channel_type, channel_args=ch_args)

        if step == 0:
            with torch.no_grad():
                stem0 = model.prep[0](imgs)
                ref_img_stats = (
                    stem0.mean(dim=(0, 2, 3)).cpu().numpy(),
                    stem0.std(dim=(0, 2, 3)).cpu().numpy()
                )
                z_enc0 = model.encoder(imgs)
                z_flat0 = z_enc0.view(z_enc0.size(0), -1)
                mod.channel_estimates = []
                mod.apply_channel(z_flat0)
                est = torch.cat(mod.channel_estimates, dim=0)
                h_mag = torch.norm(est, dim=1).cpu().numpy()
                h_norm = (h_mag - h_mag.mean()) / (h_mag.std() + 1e-6)
                ref_ch_stats, _ = np.histogram(h_norm, bins=50, range=(-5, 5), density=True)
            print("[Batch 0] Set as reference.")
            correct += 1
            total += 1
            continue

        shift_flag, cur_img_stats, z_batch = detect_dual_domain_shift(
            imgs, model, mod, ref_img_stats, ref_ch_stats, config
        )
        predicted_shift = shift_flag
        expected_shift = domain_sequence[step] != domain_sequence[step - 1]
        if predicted_shift == expected_shift:
            correct += 1
        total += 1

        # update refs
        ref_img_stats = cur_img_stats
        with torch.no_grad():
            mod.channel_estimates = []
            mod.apply_channel(z_batch)
            est = torch.cat(mod.channel_estimates, dim=0)
            h_mag = torch.norm(est, dim=1).cpu().numpy()
            h_norm = (h_mag - h_mag.mean()) / (h_mag.std() + 1e-6)
            ref_ch_stats, _ = np.histogram(h_norm, bins=50, range=(-5, 5), density=True)

    acc = correct / total * 100
    print(f"\n[Result] bs={batch_size}: accuracy = {acc:.2f}%  ({correct}/{total})")
    return acc


def main():
    # base config
    config = {
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'img_metric': 'kl',  # kl or wasserstein or l2
        'ch_metric': 'wasserstein',  # kl or wasserstein or l2
        'thresh_img': 0.05,
        'thresh_ch': 0.05,
        'num_embeddings': 16,
        'snr': 8,
        'model_path': 'JSCC/trained_models/CIFAR10-awgn/best.pt',
        'batches_per_domain': 3,
    }

    # batch sizes to scan & total simulated steps per size
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    SIM_STEPS = 5000

    # load model once
    model = DTJSCC_CIFAR10(3, 512, 10, num_embeddings=config['num_embeddings'])
    model.load_state_dict(torch.load(config['model_path'], map_location='cpu')['model_states'])
    model = model.to(config['device']).eval()

    results = {}
    for bs in BATCH_SIZES:
        try:
            acc = run_sim_for_batch_size(model, config, batch_size=bs, sim_steps=SIM_STEPS)
            results[bs] = acc
        except RuntimeError as e:
            print(f"[WARN] bs={bs} failed: {e}")
            results[bs] = np.nan
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n======== Summary (Detection Accuracy) ========")
    for bs in BATCH_SIZES:
        val = results[bs]
        if np.isnan(val):
            print(f"bs={bs:>4}: FAILED")
        else:
            print(f"bs={bs:>4}: {val:6.2f}%")
    print("=============================================")

    # ----- save to .mat -----
    out_path = f"results/detection_diff_bs_results_{config['img_metric']}_{config['ch_metric']}.mat"
    savemat(out_path, {
        'batch_sizes': np.array(BATCH_SIZES, dtype=np.int32),
        'accuracy': np.array([results[bs] for bs in BATCH_SIZES], dtype=np.float32),
        'sim_steps': np.array([SIM_STEPS], dtype=np.int32),
        'thresh_img': np.array([config['thresh_img']], dtype=np.float32),
        'thresh_ch': np.array([config['thresh_ch']], dtype=np.float32),
    })
    print(f"[Saved] MATLAB file written to: {out_path}")


if __name__ == '__main__':
    main()
