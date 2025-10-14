import os
import re
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
from scipy.io import savemat

# ======================= choose dataset here (no CLI) =======================
# Options (case-insensitive, suffixes OK): CIFAR10, CIFAR100, CINIC10
TARGET_DATASET = "CIFAR100"

# ======================= dataset helpers ===================================
_DATASET_BASE_RE = re.compile(r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=re.IGNORECASE)

def detect_base(dataset_name: str) -> str:
    """Normalize input dataset name to 'CIFAR10' / 'CIFAR100' / 'CINIC10'."""
    ds = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(ds)
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).upper()

# Compute DS BEFORE defining the templates so f-strings interpolate correctly
DS = detect_base(TARGET_DATASET)   # "CIFAR10" / "CIFAR100" / "CINIC10"

# ======================= I/O helpers =======================================
def load_acc1s(pt_path: str):
    """Load 'acc1s' list from a .pt file; warn and return None if missing."""
    if not os.path.exists(pt_path):
        print(f"[!] File not found: {pt_path}")
        return None
    try:
        data = torch.load(pt_path, map_location="cpu")
        acc1s = data.get("acc1s")
        if acc1s is None:
            print(f"[!] No 'acc1s' key in {pt_path}")
        return acc1s
    except Exception as exc:  # keep behavior consistent with original
        print(f"[!] Failed to load {pt_path}: {exc}")
        return None

def build_matrix(order_list: List[Tuple[str, str]],
                 loaded: Dict[Tuple[str, str], List[float]]):
    """Create a (rows x time) matrix, padding with NaNs for missing entries."""
    acc_rows, missing = [], []
    for tup in order_list:
        acc1s = loaded.get(tup)
        if acc1s is None:
            missing.append(tup[1])        # keep file name for warning
            acc_rows.append([np.nan])     # placeholder row
        else:
            acc_rows.append(np.asarray(acc1s))
    if missing:
        print(f"[!] Warning: missing acc1s for {missing}")
    max_len = max(len(row) for row in acc_rows)
    acc_matrix = np.vstack([
        np.pad(row, (0, max_len - len(row)), constant_values=np.nan)
        for row in acc_rows
    ])
    return acc_matrix

def convert_full_sweep_to_mat(pt_path: str, out_dir="results_mat"):
    """
    Convert a full sweep-style .pt file (no content rewrite) to .mat.
    If file is missing, caller should have checked already.
    """
    os.makedirs(out_dir, exist_ok=True)
    data = torch.load(pt_path, map_location="cpu")
    out_dict = {}

    if isinstance(data, dict):
        for sweep_type, sweep_data in data.items():
            if not isinstance(sweep_data, dict):
                continue
            keys, acc1s, acc3s, times = [], [], [], []
            for k, v in sweep_data.items():
                keys.append(str(k))
                acc1s.append(v.get('acc1', np.nan) if isinstance(v, dict) else np.nan)
                acc3s.append(v.get('acc3', np.nan) if isinstance(v, dict) else np.nan)
                times.append(v.get('ms',   np.nan) if isinstance(v, dict) else np.nan)
            out_dict[f"{sweep_type}_keys"] = np.array(keys, dtype=object)
            out_dict[f"{sweep_type}_acc1"] = np.array(acc1s)
            out_dict[f"{sweep_type}_acc3"] = np.array(acc3s)
            out_dict[f"{sweep_type}_ms"]   = np.array(times)

    mat_name = os.path.basename(pt_path).replace(".pt", ".mat")
    out_path = os.path.join(out_dir, mat_name)
    savemat(out_path, out_dict)
    print(f"[MAT] Saved full sweep to: {out_path}")

# ======================= templates (f-strings with DS) ======================
FILE_LIST_TEMPLATE = [
    # ---------- JSCC ----------
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}-awgn.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}-rician.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}-rayleigh.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_noise-awgn.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_noise-rician.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_noise-rayleigh.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_rain-awgn.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_rain-rician.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_rain-rayleigh.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_snow-awgn.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_snow-rician.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_snow-rayleigh.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_fog-awgn.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_fog-rician.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_fog-rayleigh.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_bright-awgn.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_bright-rician.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_bright-rayleigh.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_blur-awgn.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_blur-rician.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_blur-rayleigh.pt"),

    # ---------- RT-JSCC ----------
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}-rician-eval-{DS}-rician.pt"),
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}-rayleigh-eval-{DS}-rayleigh.pt"),
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}_noise-awgn-eval-{DS}_noise-awgn.pt"),
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}_noise-rician-eval-{DS}_noise-rician.pt"),
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}_noise-rayleigh-eval-{DS}_noise-rayleigh.pt"),

    ("RT-JSCC/eval", f"{DS}-rician-rt-{DS}-rayleigh-eval-{DS}-rayleigh.pt"),
    ("RT-JSCC/eval", f"{DS}-rayleigh-rt-{DS}_noise-awgn-eval-{DS}_noise-awgn.pt"),
    ("RT-JSCC/eval", f"{DS}_noise-awgn-rt-{DS}_noise-rician-eval-{DS}_noise-rician.pt"),
    ("RT-JSCC/eval", f"{DS}_noise-rician-rt-{DS}_noise-rayleigh-eval-{DS}_noise-rayleigh.pt"),

    # ---------- Meta-JSCC ----------
    ("Meta-JSCC/eval", f"meta-eval-{DS}-awgn.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}-rician.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}-rayleigh.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}_noise-awgn.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}_noise-rician.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}_noise-rayleigh.pt"),

    # ---------- JT-JSCC ----------
    ("JT-JSCC/eval", f"jt-eval-{DS}-awgn.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}-rician.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}-rayleigh.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}_noise-awgn.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}_noise-rician.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}_noise-rayleigh.pt"),

    # ---------- MD-JSCC ----------
    ("MD-JSCC/eval", f"md-eval-{DS}-awgn.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}-rician.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}-rayleigh.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}_noise-awgn.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}_noise-rician.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}_noise-rayleigh.pt"),

    # ---------- TTN-JSCC ----------
    ("AdapS/eval", f"ttn-eval-{DS}-awgn.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}-rician.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}-rayleigh.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_noise-awgn.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_noise-rician.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_noise-rayleigh.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_rain-awgn.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_rain-rician.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_rain-rayleigh.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_snow-awgn.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_snow-rician.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_snow-rayleigh.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_fog-awgn.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_fog-rician.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_fog-rayleigh.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_bright-awgn.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_bright-rician.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_bright-rayleigh.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_blur-awgn.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_blur-rician.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_blur-rayleigh.pt"),

    # ---------- TENT-JSCC ----------
    ("AdapS/eval", f"tent-eval-{DS}-awgn.pt"),
    ("AdapS/eval", f"tent-eval-{DS}-rician.pt"),
    ("AdapS/eval", f"tent-eval-{DS}-rayleigh.pt"),
    ("AdapS/eval", f"tent-eval-{DS}_noise-awgn.pt"),
    ("AdapS/eval", f"tent-eval-{DS}_noise-rician.pt"),
    ("AdapS/eval", f"tent-eval-{DS}_noise-rayleigh.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_rain-awgn.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_rain-rician.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_rain-rayleigh.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_snow-awgn.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_snow-rician.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_snow-rayleigh.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_fog-awgn.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_fog-rician.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_fog-rayleigh.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_bright-awgn.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_bright-rician.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_bright-rayleigh.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_blur-awgn.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_blur-rician.pt"),
    # ("AdapS/eval", f"ttn-eval-{DS}_blur-rayleigh.pt"),
]

FIG1A_TEMPLATE = [
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}-awgn.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}-rician.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}-rayleigh.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_blur-awgn.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_fog-rician.pt"),
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_noise-rayleigh.pt"),
]
LOG_NAME_FIG1A = ["AWGN", "Rician", "Rayleigh", "AWGN w/ B",  "Rayleigh w/ F",  "Rician w/ N"]

FIG1B_TEMPLATE = [
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_noise-awgn.pt"),
    ("RT-JSCC/eval", f"{DS}-rayleigh-rt-{DS}_noise-awgn-eval-{DS}_noise-awgn.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}_noise-awgn.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}_noise-awgn.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}_noise-awgn.pt"),
]
LOG_NAME_FIG1B = ["JSCC", "RT-JSCC", "Meta-JSCC", "JT-JSCC", "MD-JSCC"]

FIG2A_TEMPLATE = [
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}-rician.pt"),
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}-rician-eval-{DS}-rician.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}-rician.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}-rician.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}-rician.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}-rician.pt"),
]
LOG_NAME_FIG2A = ["JSCC", "RT-JSCC", "Meta-JSCC", "JT-JSCC", "MD-JSCC", "AdapS"]

FIG2B_TEMPLATE = [
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}-rayleigh.pt"),
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}-rayleigh-eval-{DS}-rayleigh.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}-rayleigh.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}-rayleigh.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}-rayleigh.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}-rayleigh.pt"),
]
LOG_NAME_FIG2B = ["JSCC", "RT-JSCC", "Meta-JSCC", "JT-JSCC", "MD-JSCC", "AdapS"]

FIG2C_TEMPLATE = [
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_noise-awgn.pt"),
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}_noise-awgn-eval-{DS}_noise-awgn.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}_noise-awgn.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}_noise-awgn.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}_noise-awgn.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_noise-awgn.pt"),
]
LOG_NAME_FIG2C = ["JSCC", "RT-JSCC", "Meta-JSCC", "JT-JSCC", "MD-JSCC", "AdapS"]

FIG2D_TEMPLATE = [
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_noise-rician.pt"),
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}_noise-rician-eval-{DS}_noise-rician.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}_noise-rician.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}_noise-rician.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}_noise-rician.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_noise-rician.pt"),
]
LOG_NAME_FIG2D = ["JSCC", "RT-JSCC", "Meta-JSCC", "JT-JSCC", "MD-JSCC", "AdapS"]

FIG2E_TEMPLATE = [
    ("JSCC/eval", f"{DS}-awgn-eval-{DS}_noise-rayleigh.pt"),
    ("RT-JSCC/eval", f"{DS}-awgn-rt-{DS}_noise-rayleigh-eval-{DS}_noise-rayleigh.pt"),
    ("Meta-JSCC/eval", f"meta-eval-{DS}_noise-rayleigh.pt"),
    ("JT-JSCC/eval", f"jt-eval-{DS}_noise-rayleigh.pt"),
    ("MD-JSCC/eval", f"md-eval-{DS}_noise-rayleigh.pt"),
    ("AdapS/eval", f"ttn-eval-{DS}_noise-rayleigh.pt"),
]
LOG_NAME_FIG2E = ["JSCC", "RT-JSCC", "Meta-JSCC", "JT-JSCC", "MD-JSCC", "AdapS"]

FIG4_TEMPLATE = [
    f"AdapS/eval/ttn_sweep_{DS}_rician.pt",
    f"AdapS/eval/ttn_sweep_{DS}_rayleigh.pt",
    f"AdapS/eval/ttn_sweep_{DS}_noise_awgn.pt",
    f"AdapS/eval/ttn_sweep_{DS}_noise_rician.pt",
    f"AdapS/eval/ttn_sweep_{DS}_noise_rayleigh.pt",
]

# ======================= main (no CLI) ======================================
def main():
    ds_tag = DS.lower()

    # Union of needed items for matrices
    set_needed = (
        set(FIG1A_TEMPLATE)
        | set(FIG1B_TEMPLATE)
        | set(FIG2A_TEMPLATE)
        | set(FIG2B_TEMPLATE)
        | set(FIG2C_TEMPLATE)
        | set(FIG2D_TEMPLATE)
        | set(FIG2E_TEMPLATE)
    )

    # 1) load & print (NO fallback; missing is just reported)
    loaded_map: Dict[Tuple[str, str], List[float]] = {}
    current_alg = None

    for folder, fname in FILE_LIST_TEMPLATE:
        alg = folder.split("/")[0]
        if alg != current_alg:
            if current_alg is not None:
                print("\n" + "-" * 80)
            print(f"\n### {alg} ###")
            current_alg = alg

        pt_path = os.path.join(folder, fname)
        acc1s = load_acc1s(pt_path)

        if acc1s is None:
            print(f"[Skipped] {pt_path}")
            continue

        acc_str = " ".join(f"{a:.4f}" for a in acc1s)
        print(f"[File] {pt_path}\n       Acc@1s: {acc_str}")

        if (folder, fname) in set_needed:
            loaded_map[(folder, fname)] = acc1s

    # 2) save .mat with dataset tag in filename (e.g., fig2c_cifar100.mat)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    def with_tag(name: str) -> str:
        stem, ext = os.path.splitext(name)
        return f"{stem}_{ds_tag}{ext or '.mat'}"

    # fig1a
    acc_matrix_1a = build_matrix(FIG1A_TEMPLATE, loaded_map)
    savemat(results_dir / with_tag("fig1a.mat"),
            {"acc_matrix_1a": acc_matrix_1a,
             "log_names": np.array(LOG_NAME_FIG1A, dtype=object)})
    print(f"\nSaved  {results_dir / with_tag('fig1a.mat')}")

    # fig1b
    acc_matrix_1b = build_matrix(FIG1B_TEMPLATE, loaded_map)
    savemat(results_dir / with_tag("fig1b.mat"),
            {"acc_matrix_1b": acc_matrix_1b,
             "log_names": np.array(LOG_NAME_FIG1B, dtype=object)})
    print(f"Saved  {results_dir / with_tag('fig1b.mat')}")

    # fig2a–2e
    for lst, names, tag in [
        (FIG2A_TEMPLATE, LOG_NAME_FIG2A, "fig2a"),
        (FIG2B_TEMPLATE, LOG_NAME_FIG2B, "fig2b"),
        (FIG2C_TEMPLATE, LOG_NAME_FIG2C, "fig2c"),
        (FIG2D_TEMPLATE, LOG_NAME_FIG2D, "fig2d"),
        (FIG2E_TEMPLATE, LOG_NAME_FIG2E, "fig2e"),
    ]:
        acc_mat = build_matrix(lst, loaded_map)
        savemat(results_dir / with_tag(f"{tag}.mat"),
                {"acc_matrix": acc_mat,
                 "log_names":  np.array(names, dtype=object)})
        print(f"Saved  {results_dir / with_tag(f'{tag}.mat')}")

    # 3) full-sweep conversions (NO fallback; missing reported)
    for pt_path in FIG4_TEMPLATE:
        if not os.path.exists(pt_path):
            print(f"[!] Skipped missing file: {pt_path}")
            continue
        try:
            convert_full_sweep_to_mat(pt_path, "results_mat")
        except Exception as e:
            print(f"[!] Failed to convert {pt_path}: {e}")

if __name__ == "__main__":
    main()
