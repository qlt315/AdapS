# md_train_final.py
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import DeepJSCC_MultiDecoder, ratio2filtersize, _Encoder, _Decoder
from channel import Channel
from utils import image_normalization, set_seed, get_psnr
from fractions import Fraction
from dataset import Cifar10Modified, Vanilla
import numpy as np
import time
from tensorboardX import SummaryWriter
import yaml
import random
from torch.nn.parallel import DataParallel

def train_epoch(model, optimizer, param, data_loader_dict, tasks, snr):
    model.train()
    epoch_loss = 0
    main_loader = data_loader_dict[param['base_dataset']]['train']
    other_loader_iters = {name: iter(loader['train']) for name, loader in data_loader_dict.items() if
                          name != param['base_dataset']}

    for images, _ in tqdm(main_loader, desc=f"Training Epoch"):
        task = random.choice(tasks)
        dataset_name, channel_type = task['dataset'], task['channel']
        task_name = f"{dataset_name}_{channel_type}"

        if dataset_name != param['base_dataset']:
            try:
                images, _ = next(other_loader_iters[dataset_name])
            except StopIteration:
                other_loader_iters[dataset_name] = iter(data_loader_dict[dataset_name]['train'])
                images, _ = next(other_loader_iters[dataset_name])

        images = images.to(param['device'])
        channel = Channel(channel_type=channel_type, snr=snr).to(param['device'])

        optimizer.zero_grad()
        outputs = model(images, task_name=task_name, channel=channel)
        outputs = image_normalization('denormalization')(outputs)
        images = image_normalization('denormalization')(images)
        loss = model.loss(images, outputs)
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
            task_name = f"{dataset_name}_{channel_type}"
            test_loader = data_loader_dict[dataset_name]['test']
            channel = Channel(channel_type=channel_type, snr=snr).to(param['device'])

            total_mse = 0
            for images, _ in test_loader:
                images = images.to(param['device'])
                outputs = model(images, task_name=task_name, channel=channel)

                ### --- THIS IS THE CORRECTED PART --- ###
                # Denormalize both tensors to [0, 255] range before calculating MSE.
                # This ensures consistency with your training loss calculation.
                outputs = image_normalization('denormalization')(outputs)
                images = image_normalization('denormalization')(images)

                mse = criterion(outputs, images)
                total_mse += mse.item()

            avg_mse = total_mse / len(test_loader)

            # Use the standard PSNR formula for 8-bit images (MAX_I = 255).
            if avg_mse > 0:
                psnr = 10 * np.log10(255 ** 2 / avg_mse)
            else:
                psnr = float('inf')

            task_results[task_name] = {"psnr": psnr, "loss": avg_mse}

    for name, results in task_results.items():
        print(f"  - Val PSNR on {name}: {results['psnr']:.2f} dB")

    avg_loss_for_scheduler = np.mean([res['loss'] for res in task_results.values()])
    avg_psnr = np.mean([res['psnr'] for res in task_results.values()])
    return avg_loss_for_scheduler, avg_psnr


def train_pipeline(params, tasks):
    """
    The core training pipeline, now with model saving logic included.
    """
    # --- Data Loading (No changes needed here) ---
    dataloaders = {}
    transform = transforms.Compose([transforms.ToTensor()])
    unique_datasets = sorted(list(set(task['dataset'] for task in tasks)))
    batch_size = params.get('batch_size', 32)
    num_workers = params.get('num_workers', 0)
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
            'train': DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test': DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }

    # --- Model Creation (No changes needed here) ---
    image_first, _ = dataloaders[params['base_dataset']]['train'].dataset[0]
    c = ratio2filtersize(image_first, params['fixed_ratio'])
    model = DeepJSCC_MultiDecoder(c=c, tasks=tasks).to(params['device'])
    print(
        f"Training a Multi-Decoder model. Base Channel Dim: {c}, Ratio: {params['fixed_ratio']:.2f}, Fixed SNR: {params['fixed_snr']}dB")

    # ### MODIFICATION 1: Create checkpoint directory ###
    # --- Experiment Directories and Tensorboard Writer ---
    out_dir = params['out_dir']
    phaser = 'MD-JSCC_' + time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(out_dir, 'logs', phaser)
    ckpt_dir = os.path.join(out_dir, 'checkpoints', phaser)  # Directory for model checkpoints
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)  # Create the checkpoint directory

    writer = SummaryWriter(log_dir=log_dir)

    # --- Model and Optimizer Initialization (No changes needed here) ---
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
    best_val_psnr = 0.0  # Variable to keep track of the best performance

    for epoch in range(params['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{params['epochs']} ---")
        epoch_train_loss = train_epoch(model, optimizer, params, dataloaders, tasks, params['fixed_snr'])
        writer.add_scalar('train_loss_avg', epoch_train_loss, epoch + 1)

        eval_freq = params.get('eval_frequency', 1)
        if (epoch + 1) % eval_freq == 0 or (epoch + 1) == params['epochs']:
            print(f"--- Running evaluation for Epoch {epoch + 1} ---")
            epoch_val_loss, epoch_val_psnr = evaluate_epoch(model, params, dataloaders, tasks, params['fixed_snr'])
            latest_val_psnr = epoch_val_psnr
            writer.add_scalar('val_psnr_avg', epoch_val_psnr, epoch + 1)
            if scheduler is not None:
                scheduler.step(epoch_val_loss)

            # ### MODIFICATION 2: Add model saving logic ###
            # -----------------------------------------------------------
            # Always save the latest model checkpoint after an evaluation.
            latest_ckpt_path = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
            torch.save(model.state_dict(), latest_ckpt_path)
            print(f"Saved latest checkpoint to: {latest_ckpt_path}")

            # Save another copy only if it's the best performing model so far.
            if latest_val_psnr > best_val_psnr:
                best_val_psnr = latest_val_psnr
                best_ckpt_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"New best performance! Saved best checkpoint to: {best_ckpt_path}")
            # -----------------------------------------------------------

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, epoch + 1)
        print(
            f"Epoch {epoch + 1} finished. Train Loss: {epoch_train_loss:.4f}, Last Val PSNR: {latest_val_psnr:.2f}, LR: {current_lr:.6f}")

    print("Training complete.")


def main():
    # Set the hardcoded path to your config file
    config_path = 'MD-JSCC/md_config.yaml'

    print(f" Starting Multi-Task Joint Training from hardcoded config file: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please create it.")

    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)

    # Ensure required keys exist with default values if necessary
    params['fixed_ratio'] = float(Fraction(params.get('fixed_ratio', '1/12')))

    try:
        tasks_to_train = params['training_tasks']
    except KeyError:
        raise ValueError("The 'training_tasks' list must be defined in your config.yaml file.")

    # Call the main training pipeline
    train_pipeline(params, tasks_to_train)


if __name__ == '__main__':
    main()