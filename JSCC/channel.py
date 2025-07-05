# channel.py
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rayleigh
from scipy.signal import lfilter


def generate_correlated_rayleigh_trace(num_samples, fdT):
    rho = np.clip(np.abs(np.i0(2 * np.pi * fdT) / np.exp(2 * np.pi * fdT)), 0.0, 0.99)
    h_real = lfilter([np.sqrt(1 - rho**2)], [1, -rho], np.random.randn(num_samples))
    h_imag = lfilter([np.sqrt(1 - rho**2)], [1, -rho], np.random.randn(num_samples))
    envelope = np.sqrt(h_real**2 + h_imag**2)
    return envelope


def quantize_trace(envelope, num_states=8):
    dist = rayleigh(scale=1.0)
    eps = 1e-6
    cdf_points = np.linspace(eps, 1 - eps, num_states + 1)
    state_edges = dist.ppf(cdf_points)
    state_vals = 0.5 * (state_edges[:-1] + state_edges[1:])
    state_seq = np.digitize(envelope, state_edges) - 1
    state_seq = np.clip(state_seq, 0, num_states - 1)
    return state_seq, state_edges, state_vals


def estimate_transition_matrix(state_seq, num_states):
    P = np.zeros((num_states, num_states))
    for i in range(len(state_seq) - 1):
        s_curr, s_next = state_seq[i], state_seq[i + 1]
        P[s_curr, s_next] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.where(row_sums > 0, P / row_sums, 1.0 / num_states)
    pi = np.ones(num_states) / num_states
    state_vals = np.arange(num_states)
    return (
        torch.tensor(P, dtype=torch.float32),
        torch.tensor(pi, dtype=torch.float32),
        torch.tensor(state_vals, dtype=torch.float32)
    )


class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=20, rician_k=3.0, nakagami_m=1.0, fading_mode='per_image',
                 num_states=16, fdT=0.1, block_size=1000, flip_ratio=0.2, flip_mode='random', flip_mask=None):
        super(Channel, self).__init__()
        if channel_type not in ['AWGN', 'Rayleigh', 'Rician', 'Nakagami', 'Flip']:
            raise ValueError('Unknown type of channel')
        if fading_mode not in ['per_batch', 'per_image', 'per_channel', 'per_pixel', 'markov']:
            raise ValueError('Invalid fading mode')

        self.channel_type = channel_type
        self.fading_mode = fading_mode
        self.snr = snr
        self.rician_k = rician_k
        self.nakagami_m = nakagami_m
        self.num_states = num_states
        self.fdT = fdT
        self.block_size = block_size
        self.flip_ratio = flip_ratio
        self.flip_mode = flip_mode
        self.flip_mask = flip_mask

        if fading_mode == 'markov':
            self._init_markov_fading()

    def _init_markov_fading(self):
        envelope = generate_correlated_rayleigh_trace(num_samples=100000, fdT=self.fdT)
        state_seq, _, state_vals = quantize_trace(envelope, num_states=self.num_states)
        self.P, self.pi, self.state_vals = estimate_transition_matrix(state_seq, num_states=self.num_states)

    def get_fading_shape(self, shape):
        B, C, H, W = shape
        if self.fading_mode == 'per_batch':
            return (1, 1, 1, 1)
        elif self.fading_mode == 'per_image':
            return (B, 1, 1, 1)
        elif self.fading_mode == 'per_channel':
            return (B, C, 1, 1)
        elif self.fading_mode == 'per_pixel':
            return (B, C, H, W)

    def simulate_markov_trace(self, length):
        fading = []
        state = torch.multinomial(self.pi, 1).item()
        for _ in range(length):
            fading.append(self.state_vals[state].item())
            state = torch.multinomial(self.P[state], 1).item()
        return torch.tensor(fading)

    def forward(self, z_hat):
        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)
        elif z_hat.dim() != 4:
            raise ValueError('Input tensor must be 3D or 4D')

        B, C, H, W = z_hat.shape
        k = z_hat[0].numel()
        sig_pwr = torch.sum(z_hat.pow(2), dim=(1, 2, 3), keepdim=True) / k
        noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2)

        if self.channel_type == 'AWGN':
            return (z_hat + noise).squeeze(0) if z_hat.shape[0] == 1 else z_hat + noise

        fading_shape = self.get_fading_shape(z_hat.shape) if self.fading_mode != 'markov' else None

        if self.channel_type == 'Rayleigh':
            if self.fading_mode == 'markov':
                total_pixels = H * W
                num_blocks = math.ceil(total_pixels / self.block_size)
                z_hat_flat = z_hat.view(B, C, -1)
                out_flat = torch.zeros_like(z_hat_flat)
                for b in range(B):
                    fading_trace = self.simulate_markov_trace(num_blocks)
                    for blk in range(num_blocks):
                        start = blk * self.block_size
                        end = min((blk + 1) * self.block_size, total_pixels)
                        h = fading_trace[blk]
                        out_flat[b, :, start:end] = z_hat_flat[b, :, start:end] * h
                out = out_flat.view(B, C, H, W) + noise
            else:
                h = torch.randn(fading_shape, device=z_hat.device)
                out = z_hat * h + noise

        elif self.channel_type == 'Rician':
            k = self.rician_k
            s = math.sqrt(k / (k + 1))
            sigma = math.sqrt(1 / (2 * (k + 1)))
            los = s
            nlos = torch.randn(fading_shape, device=z_hat.device) * sigma
            h = los + nlos
            out = z_hat * h + noise

        elif self.channel_type == 'Nakagami':
            m = self.nakagami_m
            gamma_shape = m
            gamma_scale = 1.0 / m
            h_mag_sq = torch.distributions.Gamma(gamma_shape, gamma_scale).sample(fading_shape).to(z_hat.device)
            h = torch.sqrt(h_mag_sq)
            out = z_hat * h + noise

        elif self.channel_type == 'Flip':
            out = z_hat.clone()
            total_pixels = C * H * W
            num_flips = int(self.flip_ratio * total_pixels)
            if num_flips == 0:
                return z_hat + noise  # no disturbance

            for b in range(B):
                if self.flip_mode == 'random':
                    idx = torch.randperm(total_pixels, device=z_hat.device)[:num_flips]
                    out[b].view(-1)[idx] *= -1

                elif self.flip_mode == 'center_block':
                    block_area = max(1, num_flips // C)
                    side = max(1, int(math.sqrt(block_area)))
                    h_start = max(0, H // 2 - side // 2)
                    w_start = max(0, W // 2 - side // 2)
                    h_end = min(H, h_start + side)
                    w_end = min(W, w_start + side)
                    out[b, :, h_start:h_end, w_start:w_end] *= -1

                elif self.flip_mode == 'mask':
                    assert self.flip_mask is not None, "flip_mask must be provided"
                    mask = self.flip_mask.to(z_hat.device)
                    assert mask.shape == z_hat.shape or mask.shape == (1, C, H, W)
                    out[b][mask.bool()] *= -1
            out = out + noise
        return out.squeeze(0) if out.shape[0] == 1 else out

    def get_channel(self):
        return self.channel_type, self.snr, self.fading_mode

    def visualize_transition_matrix(self):
        if self.fading_mode != 'markov':
            print("visualize_transition_matrix only applicable in 'markov' mode.")
            return
        sns.heatmap(self.P.numpy(), annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Transition Matrix (fdT={self.fdT})")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.tight_layout()
        plt.show()

    def plot_acf(self, num_blocks=1000, lags=50):
        if self.fading_mode != 'markov':
            print("plot_acf only applicable in 'markov' mode.")
            return
        fading_trace = self.simulate_markov_trace(num_blocks)
        x = fading_trace - fading_trace.mean()
        acf_vals = torch.tensor([torch.dot(x[:-k], x[k:]) / torch.dot(x, x) for k in range(1, lags + 1)])
        plt.figure(figsize=(6, 4))
        plt.stem(range(1, lags + 1), acf_vals.numpy())
        plt.title(f"Autocorrelation of Markov Rayleigh Fading (fdT={self.fdT})")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def verify_limit_distribution(self, max_power=10000, verbose=False):
        if self.fading_mode != 'markov':
            print("verify_limit_distribution only applicable in 'markov' mode.")
            return

        Pn = self.P.clone()
        for _ in range(max_power):
            Pn = torch.matmul(Pn, self.P)

        print("\n--- Limit Distribution Check ---")
        print("First row of P^n:")
        print(Pn[0].round(decimals=4))
        print("Expected stationary distribution (pi):")
        print(self.pi.round(decimals=4))
        full_error_matrix = torch.abs(Pn - self.pi.unsqueeze(0))
        max_error = full_error_matrix.max().item()
        mean_error = full_error_matrix.mean().item()
        print(f"Max absolute error over entire matrix: {max_error:.6f}")
        print(f"Mean absolute error over entire matrix: {mean_error:.6f}")
