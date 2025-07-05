import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import DeepJSCC_MultiDecoder, ratio2filtersize, _Encoder, _Decoder
from channel import Channel
from utils import image_normalization
from dataset import Vanilla, Cifar10Modified
import yaml
from tensorboardX import SummaryWriter
import argparse  # Kept to create a simple namespace object
import numpy as np
from fractions import Fraction
from tqdm import tqdm


def eval_single_task(model, test_loader, writer, param, args):
    """
    Evaluates a single, specific task-decoder across a range of SNRs,
    with fallback to alternative decoders in priority order:
    (1) noise variant > (2) Rayleigh > (3) AWGN
    """
    task_name_to_eval = f"{args.eval_dataset}_{args.eval_channel}"
    available_decoders = list(model.decoders.keys())

    if task_name_to_eval not in model.decoders:
        print(f"[Warning] Task '{task_name_to_eval}' not found in available decoders.")
        print(f"Available decoders: {available_decoders}")

        fallback = None

        # Try any decoder containing 'noise' if dataset includes 'noise'
        if 'noise' in args.eval_dataset:
            candidates = [t for t in available_decoders if 'noise' in t]
            if candidates:
                fallback = candidates[0]
                print(f"Fallback: Using '{fallback}' because it includes 'noise'.")

        # Try same dataset with Rayleigh channel
        if fallback is None:
            rayleigh_name = f"{args.eval_dataset}_Rayleigh"
            if rayleigh_name in available_decoders:
                fallback = rayleigh_name
                print(f"Fallback: Using '{fallback}' because Rayleigh decoder found.")

        # Try same dataset with AWGN channel
        if fallback is None:
            awgn_name = f"{args.eval_dataset}_AWGN"
            if awgn_name in available_decoders:
                fallback = awgn_name
                print(f"Fallback: Using '{fallback}' because AWGN decoder found.")

        if fallback is None:
            raise ValueError(
                f"[Error] Could not find any fallback decoder for '{task_name_to_eval}'.\n"
                f"Available decoders are: {available_decoders}"
            )

        task_name_to_eval = fallback

    print(f"--- Evaluating Task Decoder: {task_name_to_eval} ---")

    # === The rest remains unchanged ===
    snr_list = [7]
    criterion = nn.MSELoss()

    for snr in tqdm(snr_list, desc=f"SNR Sweep for {task_name_to_eval}"):
        channel = Channel(channel_type=args.eval_channel, snr=snr).to(param['device'])
        total_mse = 0

        for i in range(args.times):
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(param['device'])
                    outputs = model(images, task_name=task_name_to_eval, channel=channel)

                    outputs = image_normalization('denormalization')(outputs)
                    images = image_normalization('denormalization')(images)

                    total_mse += criterion(outputs, images).item()

        avg_mse = total_mse / (len(test_loader) * args.times)
        psnr = 10 * np.log10(255 ** 2 / avg_mse) if avg_mse > 0 else float('inf')

        print(f"  => SNR: {snr:2d} dB | Average PSNR: {psnr:.2f} dB")
        writer.add_scalar('psnr', psnr, snr)


def build_eval_loader(params, args):
    """
    Builds the DataLoader for the specified evaluation dataset.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    eval_dataset_name = args.eval_dataset

    if eval_dataset_name == 'cifar10':
        test_dataset = datasets.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
    elif eval_dataset_name.startswith('cifar10_'):
        test_dataset = Cifar10Modified(eval_dataset_name, transform=transform, train=False)
    else:
        raise NotImplementedError(f"Dataset '{eval_dataset_name}' not recognized.")

    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=params['batch_size'], num_workers=params.get('num_workers', 0))
    return test_loader


def process_config(config_path, args):
    """
    Loads config, model, and runs the evaluation.
    """
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)

    # --- Model Initialization ---
    dummy_image = torch.randn(1, 3, 32, 32)
    params['fixed_ratio'] = float(Fraction(params.get('fixed_ratio', '1/12')))
    c = ratio2filtersize(dummy_image, params['fixed_ratio'])

    tasks_the_model_was_trained_on = params['training_tasks']
    model = DeepJSCC_MultiDecoder(c=c, tasks=tasks_the_model_was_trained_on)

    device = torch.device(params.get('device', 'cuda:0'))
    model = model.to(device)
    if params.get('parallel', False):
        model = nn.DataParallel(model)

    # --- Load Model Weights ---
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {args.model_path}")

    print(f"Loading model checkpoint from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- Setup and Run Evaluation ---
    test_loader = build_eval_loader(params, args)

    model_name_for_logs = os.path.basename(args.model_path).replace('.pth', '').replace('.pkl', '')
    # Create a unique evaluation log folder
    eval_log_name = f"{model_name_for_logs}/{args.eval_dataset}_{args.eval_channel}"
    writer = SummaryWriter(os.path.join(params['out_dir'], 'eval', eval_log_name))

    eval_single_task(model, test_loader, writer, params, args)
    writer.close()
    print(f"Evaluation complete. Results saved to TensorBoard logs.")


def main():
    # Path to the configuration file used during training.
    config_path = 'MD-JSCC/md_config.yaml'

    # Path to the trained multi-decoder model checkpoint (.pth or .pkl).
    #    You must paste the full, correct path to your saved model here.
    model_path = 'MD-JSCC/out/checkpoints/MD-JSCC_20250702_200336/checkpoint_best.pth'  # <-- IMPORTANT: REPLACE WITH YOUR ACTUAL PATH

    # Specify which task-decoder you want to evaluate.
    #    This must match one of the tasks defined in your training_tasks list.
    evaluation_dataset = 'cifar10_noise'  # e.g., 'cifar10', 'cifar10_blur'
    evaluation_channel = 'Rician'  # e.g., 'AWGN', 'Rayleigh', 'Rician'

    # Number of repetitions for averaging results to get a stable value.
    evaluation_times = 10

    # =================================================================== #

    print("--- Running Evaluation with Hardcoded Settings ---")
    print(f"Model Path:      {model_path}")
    print(f"Config Path:     {config_path}")
    print(f"Evaluating on:   '{evaluation_dataset}' dataset with '{evaluation_channel}' channel")
    print("-------------------------------------------------")

    # Create a simple "args" object to pass settings to the functions
    args = argparse.Namespace(
        model_path=model_path,
        config=config_path,
        eval_dataset=evaluation_dataset,
        eval_channel=evaluation_channel,
        times=evaluation_times
    )

    process_config(args.config, args)


if __name__ == '__main__':
    main()