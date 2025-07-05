# jt_eval_hardcoded.py

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import DeepJSCC, ratio2filtersize  # Make sure these are importable
from channel import Channel
from utils import get_psnr
from dataset import Vanilla, Cifar10Modified
import yaml
from tensorboardX import SummaryWriter
import argparse  # Keep for creating a simple args object
import numpy as np
from fractions import Fraction


# The core functions (eval_on_conditions, build_eval_loader, process_config)
# remain unchanged as they are robust.

def eval_on_conditions(model, test_loader, writer, param, args):
    """
    Evaluates the model across a list of specified SNRs and a channel type.
    """
    # snr_list = range(0, 26, 1)  # Test on SNRs: 0, 5, 10, 15, 20, 25 dB
    snr_list = [7]
    criterion = nn.MSELoss()

    print(f"--- Starting Evaluation on Channel: {args.eval_channel} ---")

    for snr in snr_list:
        channel = Channel(channel_type=args.eval_channel, snr=snr).to(param['device'])
        total_mse = 0

        for i in range(args.times):
            run_mse = 0
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(param['device'])

                    z = model.encoder(images) if not param.get('parallel', False) else model.module.encoder(images)
                    z_noisy = channel(z)
                    outputs = model.decoder(z_noisy) if not param.get('parallel', False) else model.module.decoder(
                        z_noisy)

                    run_mse += criterion(outputs, images).item()

            total_mse += (run_mse / len(test_loader))

        avg_mse = total_mse / args.times
        psnr = 10 * np.log10(1 / avg_mse) if avg_mse > 0 else float('inf')

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
    elif eval_dataset_name == 'imagenet':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
        test_dataset = Vanilla(root='dataset/ImageNet/val', transform=transform)
    else:
        test_dataset = Cifar10Modified(eval_dataset_name, transform=transform, train=False)

    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=params['batch_size'], num_workers=params.get('num_workers', 0))
    return test_loader


def process_config(config_path, args):
    """
    Loads config, model, and runs the evaluation.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create the params dictionary from the loaded config
    params = config

    # Calculate channel size 'c' using the ratio from the config
    # We create a dummy tensor for this calculation
    dummy_image = torch.randn(1, 3, 32, 32)
    params['fixed_ratio'] = float(Fraction(params.get('fixed_ratio', '1/12')))
    c = ratio2filtersize(dummy_image, params['fixed_ratio'])

    test_loader = build_eval_loader(params, args)

    model_name = os.path.basename(args.model_path).replace('.pth', '').replace('.pkl', '')
    writer = SummaryWriter(
        os.path.join(params['out_dir'], 'eval', model_name, f"{args.eval_dataset}_{args.eval_channel}"))

    model = DeepJSCC(c=c)
    model = model.to(params['device'])
    if params.get('parallel', False):
        model = nn.DataParallel(model)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {args.model_path}")

    print(f"Loading model checkpoint from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(params['device'])))
    model.eval()

    eval_on_conditions(model, test_loader, writer, params, args)
    writer.close()
    print(f"Evaluation complete. Results saved to TensorBoard logs.")


def main():

    # 1. Path to the configuration file used during training.
    #    This is needed to load model parameters like ratio, batch size, etc.
    config_path = 'JT-JSCC/jt_config.yaml'

    # 2. Path to the trained model checkpoint file (.pth or .pkl).
    #    You must paste the full, correct path to your saved model here.
    model_path = 'JT-JSCC/out/checkpoints/JT-JSCC_20250702_193032/checkpoint_best.pth'  # <-- IMPORTANT: REPLACE WITH YOUR ACTUAL PATH

    # 3. Evaluation-specific settings.
    #    Choose which dataset and channel you want to test the model on.
    evaluation_dataset = 'cifar10_noise'  # Options: 'cifar10', 'cifar10_blur', etc.
    evaluation_channel = 'Rician'  # Options: 'AWGN', 'Rayleigh', 'Rician'
    evaluation_times = 10  # Number of times to average results for stability.

    # =================================================================== #

    print("--- Running Evaluation with Hardcoded Settings ---")
    print(f"Model Path:      {model_path}")
    print(f"Config Path:     {config_path}")
    print(f"Evaluating on:   {evaluation_dataset} with {evaluation_channel} channel")
    print("-------------------------------------------------")

    # Create a simple "args" object to pass settings to the functions,
    # avoiding the need to rewrite them.
    args = argparse.Namespace(
        model_path=model_path,
        config=config_path,
        eval_dataset=evaluation_dataset,
        eval_channel=evaluation_channel,
        times=evaluation_times
    )

    # Check if files exist before running
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Start the evaluation process
    process_config(args.config, args)


if __name__ == '__main__':
    main()