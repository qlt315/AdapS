import torch
import torch.nn as nn
import os
import yaml
import argparse
import higher  # Library for meta-learning
from model import DeepJSCC
from channel import Channel  # <-- Make sure this import is present
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils import get_psnr, image_normalization
from tensorboardX import SummaryWriter
from copy import deepcopy

# This script is designed to evaluate a trained MAML model.
# The evaluation protocol is:
# 1. For each target SNR (e.g., 0, 10, 20 dB):
# 2. Take the base meta-model.
# 3. Fine-tune (adapt) it for a few steps on data with that specific SNR.
# 4. Evaluate the performance of the newly adapted model on the same SNR.

# -----------------------------------------------------------------------
# CORRECTED Inner Adaptation Function
# -----------------------------------------------------------------------
def meta_adapt(model, loader, device, snr, inner_steps=5, lr=1e-4):
    """
    Adapts the meta-model to a specific task (SNR) for a few steps.

    Args:
        model (nn.Module): The base meta-model to be adapted.
        loader (DataLoader): DataLoader for the adaptation data.
        device (torch.device): The device to run on (e.g., 'cuda').
        snr (int): The target SNR for this adaptation task.
        inner_steps (int): The number of adaptation gradient steps.
        lr (float): The learning rate for the inner loop adaptation.

    Returns:
        higher.MonkeyPatched: The adapted model.
    """
    # If inner_steps is 0, no adaptation is performed.
    # This corresponds to a "zero-shot" evaluation of the meta-model.
    if inner_steps == 0:
        print("[Inner Adapt] Skipping adaptation (inner_steps=0).")
        return model

    model.train()
    criterion = nn.MSELoss()
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create a specific channel object for the adaptation task.
    adapt_channel = Channel('AWGN', snr=snr).to(device)

    # Use `higher` to make the adaptation process differentiable if needed,
    # or just to manage temporary model states. `copy_initial_weights=False`
    # means it will modify the model's weights in place for this context.
    with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
        for step in range(inner_steps):
            # Typically, adaptation uses just one or a few batches of data.
            images, _ = next(iter(loader))
            images = images.to(device)

            # --- CRITICAL FIX ---
            # Call the functional model's forward pass, explicitly providing the channel object.
            outputs = fmodel(images, channel=adapt_channel)

            outputs = image_normalization('denormalization')(outputs)
            images = image_normalization('denormalization')(images)
            loss = criterion(outputs, images)
            diffopt.step(loss)  # Apply gradient step to the adapted model

            print(f"[Inner Adapt] Step {step + 1}/{inner_steps} | Loss: {loss.item():.4f} | using snr={snr}")

    # Return the adapted model for evaluation.
    return fmodel


# -----------------------------------------------------------------------
# CORRECTED Evaluation Function
# -----------------------------------------------------------------------
# def evaluate_epoch(model, param, data_loader, snr):
#     """
#     Evaluates a given model over a dataset for a specific SNR.
#
#     Args:
#         model (nn.Module): The model to evaluate (can be the base meta-model or an adapted one).
#         param (dict): Dictionary of parameters, including the device.
#         data_loader (DataLoader): DataLoader for the evaluation data.
#         snr (int): The SNR at which to evaluate the model.
#
#     Returns:
#         float: The average loss (MSE) over the dataset.
#     """
#     model.eval()
#     criterion = nn.MSELoss()
#     epoch_loss = 0.0
#     device = param['device']
#
#     # Create a specific channel object for this evaluation.
#     eval_channel = Channel('AWGN', snr=snr).to(device)
#
#     with torch.no_grad():
#         for iter_idx, (images, _) in enumerate(data_loader):
#             images = images.to(device)
#
#             # --- CRITICAL FIX ---
#             # Call the model's forward pass, explicitly providing the channel object.
#             outputs = model(images, channel=eval_channel)
#
#             outputs = image_normalization('denormalization')(outputs)
#             images = image_normalization('denormalization')(images)
#             loss = criterion(outputs, images)
#             epoch_loss += loss.item()
#
#     return epoch_loss / (iter_idx + 1)

# ---------- CORRECTED Evaluation ----------
def evaluate_epoch(model, param, data_loader, snr):
    model.eval()
    criterion = nn.MSELoss()
    epoch_loss = 0.0
    device = param['device']
    eval_channel = Channel('AWGN', snr=snr).to(device)

    with torch.no_grad():
        # 我们只检查第一个batch就足够了
        images, _ = next(iter(data_loader))
        images = images.to(device)

        # 1. 得到解码器的输入 (带噪的z)
        z_hat = model.encoder(images)
        decoder_input = eval_channel(z_hat)

        # ======================= START: 添加最终调试代码 =======================
        print(f"\n--- Decoder Check for SNR={snr} ---")
        print(f"Std Dev of Decoder INPUT: {decoder_input.std().item():.6f}")
        # ======================= END: 添加最终调试代码 =======================

        # 2. 得到解码器的输出 (重建图像)
        outputs = model.decoder(decoder_input)

        # ======================= START: 添加最终调试代码 =======================
        print(f"Std Dev of Decoder OUTPUT: {outputs.std().item():.6f}")
        print(f"------------------------------------")
        # ======================= END: 添加最终调试代码 =======================

        outputs = image_normalization('denormalization')(outputs)
        images = image_normalization('denormalization')(images)
        loss = criterion(outputs, images)
        epoch_loss = loss.item()

    # 为了快速调试，我们只评估一个batch
    return epoch_loss


# -----------------------------------------------------------------------
# Main Orchestration Loop
# -----------------------------------------------------------------------
def eval_snr_meta(model, test_loader, writer, param, args):
    snr_list = range(0, 26, 10)
    for snr in snr_list:
        print(f"\n==> Adapting and Evaluating for SNR {snr} dB")
        total_loss = 0.0
        for i in range(args.times):
            # --- CRITICAL FIX ---
            # Create a deepcopy of the original meta-model for each run.
            # This ensures each adaptation starts from the same clean state.
            model_clone = deepcopy(model)

            # Pass the CLONE to the adaptation function, not the original model.
            adapted_model = meta_adapt(model_clone, test_loader,
                                       device=param['device'],
                                       snr=snr,
                                       inner_steps=args.inner_steps,
                                       lr=args.inner_lr)

            val_loss = evaluate_epoch(adapted_model, param, test_loader, snr)
            total_loss += val_loss

        avg_loss = total_loss / args.times
        psnr = get_psnr(image=None, gt=None, mse=avg_loss)
        writer.add_scalar('psnr_meta_evaluation', psnr, snr)
        print(f"[Result for SNR {snr} dB] Avg Loss: {avg_loss:.4f} => PSNR: {psnr:.2f} dB")


# -----------------------------------------------------------------------
# Boilerplate: Data Loading and Configuration
# -----------------------------------------------------------------------
def build_eval_loader(params, args):
    transform = transforms.Compose([transforms.ToTensor()])
    if args.eval_dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {args.eval_dataset} not implemented.")

    return DataLoader(test_dataset, shuffle=True,
                      batch_size=params['batch_size'], num_workers=params['num_workers'])


def process_config(config_path, args):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
        params = config['params']
        c = config['inner_channel']

    # Set default values if not present in config
    params.setdefault('parallel', False)
    params.setdefault('device', args.device)

    test_loader = build_eval_loader(params, args)
    writer = SummaryWriter(os.path.join(args.output_dir, 'eval_meta', args.model_dir))

    model = DeepJSCC(c=c)
    model = model.to(params['device'])

    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', args.model_dir + '.pkl')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=params['device']))

    # Start the evaluation process
    eval_snr_meta(model, test_loader, writer, params, args)
    writer.close()


# -----------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluation script for MAML-trained models.")
    parser.add_argument('--model_dir', type=str, default='MAML_cifar10_3_0.17_32_8',
                        help='Name of the model directory (without .pkl)')
    parser.add_argument('--eval_dataset', type=str, default='cifar10', help='Dataset to use for evaluation.')
    parser.add_argument('--output_dir', type=str, default='Meta-JSCC/out', help='Base output directory.')
    parser.add_argument('--times', type=int, default=1,
                        help='Number of times to repeat the adapt/eval process for averaging.')
    parser.add_argument('--inner_steps', type=int, default=1,
                        help='Number of adaptation steps. Set to 0 for zero-shot evaluation.')
    parser.add_argument('--inner_lr', type=float, default=1e-4, help='Learning rate for the inner adaptation loop.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (e.g., "cuda" or "cpu").')
    args = parser.parse_args()

    # Find the corresponding config file
    config_path = os.path.join(args.output_dir, 'configs', args.model_dir + '.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    process_config(config_path, args)


if __name__ == '__main__':
    main()