import numpy as np
import torch
from scipy.stats import rayleigh
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import seaborn as sns

def generate_correlated_rayleigh_trace(num_samples, fdT, method='jakes'):
    """
    Generate a correlated Rayleigh fading trace using a first-order Gauss-Markov process.

    Args:
        num_samples (int): Length of the fading trace.
        fdT (float): Normalized Doppler frequency.
        method (str): Only 'jakes' (approximation) is currently supported.

    Returns:
        envelope_trace (np.ndarray): Rayleigh fading envelope sequence.
    """
    rho = np.clip(np.abs(np.i0(2 * np.pi * fdT) / np.exp(2 * np.pi * fdT)), 0.0, 0.99)  # approx autocorr
    h_real = lfilter([np.sqrt(1 - rho**2)], [1, -rho], np.random.randn(num_samples))
    h_imag = lfilter([np.sqrt(1 - rho**2)], [1, -rho], np.random.randn(num_samples))
    envelope = np.sqrt(h_real**2 + h_imag**2)
    return envelope

def quantize_rayleigh_trace(envelope, num_states=8, method='equal_cdf'):
    """
    Quantize the Rayleigh fading envelope into discrete states.

    Args:
        envelope (np.ndarray): Rayleigh envelope samples.
        num_states (int): Number of discrete states.
        method (str): 'equal_cdf' or 'equal_width'.

    Returns:
        state_seq (np.ndarray): Sequence of state indices.
        state_edges (np.ndarray): Bin edges.
        state_vals (np.ndarray): Bin centers.
    """
    dist = rayleigh(scale=1.0)
    if method == 'equal_cdf':
        eps = 1e-6
        cdf_points = np.linspace(eps, 1 - eps, num_states + 1)
        state_edges = dist.ppf(cdf_points)
    elif method == 'equal_width':
        x_max = 4.0
        state_edges = np.linspace(0.0, x_max, num_states + 1)
    else:
        raise ValueError("Unknown method")

    state_vals = 0.5 * (state_edges[:-1] + state_edges[1:])
    state_seq = np.digitize(envelope, state_edges) - 1
    state_seq = np.clip(state_seq, 0, num_states - 1)
    return state_seq, state_edges, state_vals

def estimate_transition_matrix(state_seq, num_states):
    """
    Estimate the transition matrix from the discrete state sequence.

    Args:
        state_seq (np.ndarray): Sequence of state indices.
        num_states (int): Number of discrete states.

    Returns:
        P (torch.Tensor): [num_states x num_states] transition matrix.
        pi (torch.Tensor): [num_states] empirical steady-state distribution.
    """
    P = np.zeros((num_states, num_states))
    for i in range(len(state_seq) - 1):
        s_curr, s_next = state_seq[i], state_seq[i + 1]
        P[s_curr, s_next] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.where(row_sums > 0, P / row_sums, 1.0 / num_states)
    pi = np.bincount(state_seq, minlength=num_states)
    pi = pi / pi.sum()
    return torch.tensor(P, dtype=torch.float32), torch.tensor(pi, dtype=torch.float32)

def plot_transition_matrix(P, title="Transition Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(P.numpy(), annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.tight_layout()
    plt.show()

def plot_autocorrelation(state_seq, max_lag=50):
    x = torch.tensor(state_seq, dtype=torch.float32)
    x = x - x.mean()
    acf_vals = [torch.dot(x[:-k], x[k:]) / torch.dot(x, x) for k in range(1, max_lag + 1)]
    plt.figure(figsize=(6, 4))
    plt.stem(range(1, max_lag + 1), acf_vals, use_line_collection=True)
    plt.title("State Sequence Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def verify_steady_state(P, pi_expected):
    Pn = P.clone()
    for _ in range(1000):
        Pn = torch.matmul(Pn, P)
    steady_row = Pn[0]
    diff = torch.abs(Pn - pi_expected.unsqueeze(0))
    print("\n--- Steady-State Verification ---")
    print("First row of P^n:")
    print(steady_row.round(decimals=4))
    print("Expected steady-state distribution (pi):")
    print(pi_expected.round(decimals=4))
    print("Max row-wise absolute error:", diff.max().item())

# Example usage
if __name__ == '__main__':
    trace = generate_correlated_rayleigh_trace(num_samples=100000, fdT=0.05)
    states, edges, centers = quantize_rayleigh_trace(trace, num_states=8, method='equal_cdf')
    P, pi = estimate_transition_matrix(states, num_states=8)
    verify_steady_state(P, pi)
    print("Estimated Transition Matrix (P):")
    print(P)
    print("\nEmpirical Stationary Distribution (pi):")
    print(pi)
    print("\nState Centers:")
    print(centers)

    plot_transition_matrix(P, title="Estimated Transition Matrix")
    plot_autocorrelation(states)

