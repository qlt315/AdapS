# -*- coding: utf-8 -*-


import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import DeepJSCC, ratio2filtersize
from channel import Channel
from torch.nn.parallel import DataParallel
from utils import image_normalization, set_seed, get_psnr
from fractions import Fraction
from dataset import Vanilla, Cifar10Modified
import numpy as np
import time
from tensorboardX import SummaryWriter
import glob
import yaml
import random


# The functions train_epoch and evaluate_epoch remain unchanged.
# ... (insert the full train_epoch and evaluate_epoch functions here) ...

def train_epoch(model, optimizer, param, data_loader_dict, tasks, snr):
    model.train()
    epoch_loss = 0
    main_loader = data_loader_dict[param['base_dataset']]['train']
    other_loader_iters = {name: iter(loader['train']) for name, loader in data_loader_dict.items() if
                          name != param['base_dataset']}
    for images, _ in tqdm(main_loader, desc=f"Training Epoch"):
        task = random.choice(tasks)
        dataset_name, channel_type = task['dataset'], task['channel']
        if dataset_name != param['base_dataset']:
            try:
                images, _ = next(other_loader_iters[dataset_name])
            except StopIteration:
                other_loader_iters[dataset_name] = iter(data_loader_dict[dataset_name]['train'])
                images, _ = next(other_loader_iters[dataset_name])
        images = images.cuda() if param.get('parallel', False) else images.to(param['device'])
        channel = Channel(channel_type=channel_type, snr=snr).to(param['device'])
        optimizer.zero_grad()
        z = model.encoder(images) if not param.get('parallel', False) else model.module.encoder(images)
        z_noisy = channel(z)
        outputs = model.decoder(z_noisy) if not param.get('parallel', False) else model.module.decoder(z_noisy)
        loss = model.loss(images, outputs) if not param.get('parallel', False) else model.module.loss(images, outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(main_loader)


def evaluate_epoch(model, param, data_loader_dict, tasks, snr):
    model.eval()
    criterion = nn.MSELoss()
    task_results = {}
    with torch.no_grad():
        for task in tqdm(tasks, desc="Evaluating Tasks"):
            dataset_name, channel_type = task['dataset'], task['channel']
            test_loader = data_loader_dict[dataset_name]['test']
            channel = Channel(channel_type=channel_type, snr=snr).to(param['device'])
            total_mse = 0
            for images, _ in test_loader:
                images = images.cuda() if param.get('parallel', False) else images.to(param['device'])
                z = model.encoder(images) if not param.get('parallel', False) else model.module.encoder(images)
                z_noisy = channel(z)
                outputs = model.decoder(z_noisy) if not param.get('parallel', False) else model.module.decoder(z_noisy)
                mse = criterion(outputs, images)
                total_mse += mse.item()
            avg_mse = total_mse / len(test_loader)
            psnr = 10 * torch.log10(torch.tensor(1.0 / avg_mse)) if avg_mse > 0 else float('inf')
            task_results[f"{dataset_name}/{channel_type}"] = {"psnr": psnr.item(), "loss": avg_mse}
    for task_name, results in task_results.items():
        print(f"  - Val PSNR on {task_name}: {results['psnr']:.2f} dB")
    avg_loss_for_scheduler = np.mean([res['loss'] for res in task_results.values()])
    avg_psnr = np.mean([res['psnr'] for res in task_results.values()])
    return avg_loss_for_scheduler, avg_psnr


def train_pipeline(params, tasks):
    """
    The core training pipeline, fully configured by the params dictionary and task list.
    """
    # --- Data Loading ---
    dataloaders = {}
    transform = transforms.Compose([transforms.ToTensor()])
    unique_datasets = sorted(list(set(task['dataset'] for task in tasks)))
    for d_name in unique_datasets:
        print(f"Loading dataset: {d_name}...")
        if d_name == 'cifar10':
            train_dset = datasets.CIFAR10(root='dataset/', train=True, download=True, transform=transform)
            test_dset = datasets.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
        elif d_name.startswith('cifar10_'):
            train_dset = Cifar10Modified(d_name, transform=transform, train=True)
            test_dset = Cifar10Modified(d_name, transform=transform, train=False)
        else:
            raise NotImplementedError(f"Dataset loading for {d_name} not implemented.")
        dataloaders[d_name] = {
            'train': DataLoader(train_dset, batch_size=params['batch_size'], shuffle=True,
                                num_workers=params['num_workers']),
            'test': DataLoader(test_dset, batch_size=params['batch_size'], shuffle=False,
                               num_workers=params['num_workers'])
        }

    # --- Model Creation ---
    image_first, _ = dataloaders[params['base_dataset']]['train'].dataset[0]
    c = ratio2filtersize(image_first, params['fixed_ratio'])
    model = DeepJSCC(c=c)
    print(
        f"Training a robust model. Inner Channel: {c}, Ratio: {params['fixed_ratio']:.2f}, Fixed SNR: {params['fixed_snr']}dB")

    # --- Experiment Directories and Tensorboard Writer ---
    out_dir = params['out_dir']
    phaser = 'JT-JSCC_' + time.strftime('%Y%m%d_%H%M%S')

    # ### MODIFICATION: Define checkpoint directory ###
    log_dir = os.path.join(out_dir, 'logs', phaser)
    ckpt_dir = os.path.join(out_dir, 'checkpoints', phaser)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # --- Model and Optimizer Initialization ---
    device = torch.device(params['device'])
    model = model.to(device)
    if params.get('parallel', False): model = DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params.get('weight_decay', 0))
    scheduler = None
    if params.get('if_scheduler', False) and params.get('reduce_on_plateau', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=params['lr_reduce_factor'],
                                                         patience=params['lr_schedule_patience'])

    # --- Main Training Loop ---
    latest_val_psnr = 0.0
    best_val_psnr = 0.0  # Keep track of the best performance

    for epoch in range(params['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{params['epochs']} ---")
        epoch_train_loss = train_epoch(model, optimizer, params, dataloaders, tasks, params['fixed_snr'])
        writer.add_scalar('train_loss_avg', epoch_train_loss, epoch + 1)

        eval_freq = params.get('eval_frequency', 1)

        # ### MODIFICATION: Re-enabled periodic evaluation ###
        if (epoch + 1) % eval_freq == 0 or (epoch + 1) == params['epochs']:
            print(f"--- Running evaluation for Epoch {epoch + 1} ---")
            epoch_val_loss, epoch_val_psnr = evaluate_epoch(model, params, dataloaders, tasks, params['fixed_snr'])
            latest_val_psnr = epoch_val_psnr

            writer.add_scalar('val_psnr_avg', epoch_val_psnr, epoch + 1)
            if scheduler is not None:
                scheduler.step(epoch_val_loss)

            # ### MODIFICATION: Save the model periodically ###
            # Option 1: Save the latest checkpoint after each evaluation.
            latest_ckpt_path = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
            torch.save(model.state_dict(), latest_ckpt_path)
            print(f"Saved latest checkpoint to: {latest_ckpt_path}")

            # Option 2: Save the checkpoint only if it's the best one so far.
            if latest_val_psnr > best_val_psnr:
                best_val_psnr = latest_val_psnr
                best_ckpt_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"New best performance! Saved best checkpoint to: {best_ckpt_path}")

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, epoch + 1)
        print(
            f"Epoch {epoch + 1} finished. Train Loss: {epoch_train_loss:.4f}, Last Val PSNR: {latest_val_psnr:.2f}, LR: {current_lr:.6f}")

    print("Training complete.")


### --- Main Entry Point --- ###
def main():
    config_path = 'JT-JSCC/jt_config.yaml'  # Hardcoded path

    print(f"Starting Multi-Task Joint Training from hardcoded config file: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please create it.")

    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)

    params['fixed_ratio'] = float(Fraction(params.get('fixed_ratio', '1/12')))

    try:
        tasks_to_train = params['training_tasks']
    except KeyError:
        raise ValueError("The 'training_tasks' list is not defined in your config.yaml file.")

    print("\n--- Training Configuration Loaded ---")
    for key, val in params.items():
        if key != 'training_tasks':
            print(f"{key:<25}: {val}")
    print(f"{'training_tasks':<25}:")
    for task in tasks_to_train:
        print(f"  - {task}")
    print("-------------------------------------\n")

    set_seed(params.get('seed', 42))

    train_pipeline(params, tasks_to_train)


if __name__ == '__main__':
    main()