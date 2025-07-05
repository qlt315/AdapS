import torch
from utils import get_psnr
import os
from model import DeepJSCC
from train import evaluate_epoch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from dataset import Vanilla, Cifar10Modified
import yaml
from tensorboardX import SummaryWriter
import glob
import argparse
from utils import image_normalization
import numpy as np
import torch.nn as nn
from tqdm import tqdm

def eval_snr(model, test_loader, writer, param, args):
    """
    Evaluates the single-decoder JSCC model on a range of SNRs.
    """
    snr_list = range(0, 26, 1)  # Or use step=5 for faster eval
    # snr_list = [7]
    criterion = nn.MSELoss()

    print(f"--- Evaluating JSCC Model ---")

    for snr in tqdm(snr_list, desc=f"SNR Sweep"):
        # Dynamically change channel per SNR
        model.change_channel(args.eval_channel, snr)

        total_mse = 0

        for _ in range(args.times):
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(param['device'])

                    outputs = model(images)

                    # Optional: Denormalize if your model works in [0,1] → [0,255]
                    outputs = image_normalization('denormalization')(outputs)
                    images = image_normalization('denormalization')(images)

                    total_mse += criterion(outputs, images).item()

        avg_mse = total_mse / (len(test_loader) * args.times)

        psnr = 10 * np.log10(255 ** 2 / avg_mse) if avg_mse > 0 else float('inf')

        # Log and print
        writer.add_scalar('psnr', psnr, snr)
        print(f"[Eval] SNR: {snr} dB | PSNR: {psnr:.2f} dB")


def build_eval_loader(params, args):
    transform = transforms.Compose([transforms.ToTensor()])
    eval_dataset_name = args.eval_dataset

    if eval_dataset_name == 'cifar10':
        test_dataset = datasets.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
    elif eval_dataset_name == 'imagenet':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
        test_dataset = Vanilla(root='dataset/ImageNet/val', transform=transform)
    else:
        # All custom CIFAR-10 variants go here
        test_dataset = Cifar10Modified(eval_dataset_name, transform=transform, train=False)

    test_loader = DataLoader(test_dataset, shuffle=True,
                             batch_size=params['batch_size'], num_workers=params['num_workers'])
    return test_loader


def process_config(config_path, args):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
        params = config['params']
        c = config['inner_channel']

    test_loader = build_eval_loader(params, args)
    model_name = args.model_dir
    writer = SummaryWriter(os.path.join(args.output_dir, 'eval', model_name))

    model = DeepJSCC(c=c)
    model = model.to(params['device'])


    pkl_list = sorted(glob.glob(os.path.join(os.path.join(args.output_dir, 'checkpoints', model_name), '*.pkl')))
    if not pkl_list:
        raise FileNotFoundError(f"No checkpoint found in {os.path.join(args.output_dir, 'checkpoints', model_name)}")
    model.load_state_dict(torch.load(pkl_list[-1]))

    eval_snr(model, test_loader, writer, params, args)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default='CIFAR10_8_7.0_0.17_AWGN_23h14m26s_on_Jun_19_2025',
                        help='Directory containing checkpoint and config (e.g., JSCC/out/checkpoints/model_xxx)')
    parser.add_argument('--eval_dataset', type=str, default='cifar10',
                        help='Dataset name (e.g., cifar10, imagenet, cifar10_blur, etc.)')
    parser.add_argument('--eval_channel', type=str, default='Rician',
                        help='Channel type (e.g., Flip, Rayleigh)')
    parser.add_argument('--output_dir', type=str, default='JSCC/out',
                        help='Output directory containing configs and checkpoints')
    parser.add_argument('--times', type=int, default=10,
                        help='Number of evaluation repetitions')
    args = parser.parse_args()

    # Automatically find matching config file based on model_dir name
    model_name = os.path.basename(args.model_dir)
    config_path = os.path.join(args.output_dir, 'configs', model_name + '.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Processing config: {config_path}")
    process_config(config_path, args)


if __name__ == '__main__':
    main()
