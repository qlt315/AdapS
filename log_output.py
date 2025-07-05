import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from scipy.io import savemat

def extract_and_save_psnr(log_list, output_filename):
    all_psnr = []
    log_names = []
    max_len = 0

    for item in log_list:
        name = item['name']
        base_path = item['base_path']
        subdir = item['subdir']

        full_path = os.path.join(base_path, subdir)
        ea = event_accumulator.EventAccumulator(full_path, size_guidance={event_accumulator.SCALARS: 0})
        try:
            ea.Reload()
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
            continue

        if "psnr" not in ea.Tags().get("scalars", []):
            print(f"Tag 'psnr' not found in {full_path}, skipping.")
            continue

        psnr_events = ea.Scalars("psnr")
        values = [e.value for e in psnr_events]
        all_psnr.append(values)
        log_names.append(name)
        max_len = max(max_len, len(values))

    if not all_psnr:
        print(f"No valid PSNR data found for {output_filename}")
        return

    psnr_matrix = np.full((len(all_psnr), max_len), np.nan)
    for i, vals in enumerate(all_psnr):
        psnr_matrix[i, :len(vals)] = vals

    # === New: use output_filename to decide variable names ===
    if "1_a" in output_filename:
        var_psnr = "psnr_matrix_1a"
        var_names = "log_names_1a"
    elif "1_b" in output_filename:
        var_psnr = "psnr_matrix_1b"
        var_names = "log_names_1b"
    else:
        var_psnr = "psnr_matrix"
        var_names = "log_names"

    savemat(output_filename, {
        var_psnr: psnr_matrix,
        var_names: np.array(log_names, dtype=object).reshape(1, -1)
    })
    print(f"Saved: {output_filename} | Shape: {psnr_matrix.shape} | Variables: {var_psnr}, {var_names}")


# -------------------------------
# Fig.1(a) PSNR performance of JSCC with diff channels under diff SNRs
# -------------------------------

log_list_1 = [
    {
        "name": "AWGN",
        "base_path": "JSCC/out/eval",
        "subdir": "CIFAR10_8_7.0_0.17_AWGN_AWGN_23h14m26s_on_Jun_19_2025"
    },
    {
        "name": "Rayleigh",
        "base_path": "JSCC/out/eval",
        "subdir": "CIFAR10_8_7.0_0.17_AWGN_Rayleigh_23h14m26s_on_Jun_19_2025"
    },
    {
        "name": "Rayleigh_w_noise",
        "base_path": "JSCC/out/eval",
        "subdir": "CIFAR10_8_7.0_0.17_AWGN_Rayleigh_w_noise_23h14m26s_on_Jun_19_2025"
    },
    {
        "name": "Rician",
        "base_path": "JSCC/out/eval",
        "subdir": "CIFAR10_8_7.0_0.17_AWGN_Rician_23h14m26s_on_Jun_19_2025"
    },
    {
        "name": "Rician_w_noise",
        "base_path": "JSCC/out/eval",
        "subdir": "CIFAR10_8_7.0_0.17_AWGN_Rician_w_noise_23h14m26s_on_Jun_19_2025"
    }
]

# -------------------------------
# Fig.1(b) PSNR performance of diff JSCCs under diff SNRs (Rayleigh)
# -------------------------------

log_list_2 = [
    {
        "name": "JSCC",
        "base_path": "JSCC/out/eval",
        "subdir": "CIFAR10_8_7.0_0.17_AWGN_Rayleigh_23h14m26s_on_Jun_19_2025"
    },
    {
        "name": "JT-JSCC",
        "base_path": "JT-JSCC/out/eval/",
        "subdir": "checkpoint_best/cifar10_Rayleigh"
    },
    {
        "name": "MD-JSCC",
        "base_path": "MD-JSCC/out/eval/",
        "subdir": "checkpoint_best/cifar10_Rayleigh"
    },
    {
        "name": "RT-JSCC",
        "base_path": "RT-JSCC/out/eval/",
        "subdir": "CIFAR10_8_7.0_0.17_AWGN_RETRAIN_w_Rayleigh"
    }
]

# -------------------------------
# Run both batches
# -------------------------------
extract_and_save_psnr(log_list_1, "fig_1_a.mat")
extract_and_save_psnr(log_list_2, "fig_1_b.mat")
