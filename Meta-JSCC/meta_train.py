import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import DeepJSCC, ratio2filtersize
from utils import image_normalization, get_psnr
from dataset import Cifar10Modified
import higher
import os
import time
import numpy as np
from channel import Channel
from copy import deepcopy
import yaml

# -----------------------------------------------------------------------
# Recommended Hyperparameters to Start
# -----------------------------------------------------------------------
# Meta-Learning Hyperparameters
META_EPOCHS = 100  # Train for longer to see a clear trend
INNER_STEPS = 5  # Number of inner loop adaptation steps (now means 5 batches total)
META_BATCH_SIZE = 4  # Use all tasks for a stable meta-gradient
INNER_LR = 1e-7  # Inner loop learning rate for Adam
META_LR = 1e-5  # A stable starting point for the meta-learning rate

# General Training Hyperparameters
BATCH_SIZE = 32
RATIO = 1 / 6


# -----------------------------------------------------------------------
# CORRECTED MAML Training Function
# -----------------------------------------------------------------------
def maml_train(args):
    """
    Main function for MAML training with a corrected "few-shot" inner loop.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.CIFAR10(root='dataset/', train=True, download=True, transform=transform)
    image_first = full_dataset[0][0]
    c = ratio2filtersize(image_first, RATIO)

    meta_model = DeepJSCC(c=c).to(device)
    # Use Adam for the meta-optimizer as it's generally robust.
    meta_optimizer = optim.SGD(meta_model.parameters(), lr=META_LR)
    criterion = nn.MSELoss()

    t0 = time.time()
    per_epoch_time = []

    # --- Start of the Meta-Training Outer Loop ---
    for meta_epoch in range(META_EPOCHS):
        # We deepcopy the model at the start of the epoch for a clean reference
        # This is not strictly necessary but can help if you add more complex logic later
        meta_model_epoch_start = deepcopy(meta_model)

        sampled_tasks = [TASKS[i] for i in torch.randperm(len(TASKS))[:META_BATCH_SIZE]]
        meta_losses = []

        print(f"[Meta Epoch {meta_epoch + 1}/{META_EPOCHS}]")

        # --- Loop over the sampled tasks ---
        for task_id, task in enumerate(sampled_tasks):
            dataset_name = task['dataset']
            channel_type = task['channel']
            snr = task['snr']

            print(f"  Task {task_id + 1}: Dataset={dataset_name}, Channel={channel_type}, SNR={snr}")

            # === Dataset Loading for the current task ===
            # (Your data loading code here remains unchanged)
            # ...
            train_len = int(0.5 * len(full_dataset))
            val_len = len(full_dataset) - train_len
            train_set, val_set = random_split(full_dataset, [train_len, val_len])
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
            # ...

            task_channel = Channel(channel_type, snr=snr).to(device)

            # Create a temporary task model for the inner loop.
            # We use the clean model from the start of the epoch.
            task_model = deepcopy(meta_model_epoch_start)
            inner_optimizer = optim.Adam(task_model.parameters(), lr=INNER_LR)

            # Use FOMAML (track_higher_grads=False) for stability.
            with higher.innerloop_ctx(task_model, inner_optimizer, track_higher_grads=False) as (fmodel, diffopt):

                # --- CRITICAL FIX: Correct "Few-Shot" Inner Loop ---
                train_loader_iter = iter(train_loader)

                # The inner loop now runs for exactly INNER_STEPS gradient updates.
                for step in range(INNER_STEPS):
                    try:
                        # Get ONE batch of data for this ONE adaptation step.
                        images, _ = next(train_loader_iter)
                    except StopIteration:
                        # Reset the iterator if the dataloader is exhausted
                        train_loader_iter = iter(train_loader)
                        images, _ = next(train_loader_iter)

                    images = images.to(device)

                    # Perform one adaptation step
                    outputs = fmodel(images, channel=task_channel)
                    outputs = image_normalization('denormalization')(outputs)
                    images = image_normalization('denormalization')(images)
                    loss = criterion(outputs, images)
                    diffopt.step(loss)

                    print(f"    Inner Step {step + 1}/{INNER_STEPS}, Loss: {loss.item():.4f}")

                # --- Validation ---
                fmodel.eval()
                val_images, _ = next(iter(val_loader))
                val_images = val_images.to(device)

                val_outputs = fmodel(val_images, channel=task_channel)
                val_outputs = image_normalization('denormalization')(val_outputs)
                val_images = image_normalization('denormalization')(val_images)

                val_loss = criterion(val_outputs, val_images)
                val_psnr = get_psnr(image=None, gt=None, mse=val_loss.item())

                print(f"    Validation Loss: {val_loss.item():.4f}")
                print(f"    Validation PSNR: {val_psnr:.2f} dB")
                meta_losses.append(val_loss)

        # --- Meta-Update ---
        meta_loss = torch.stack(meta_losses).mean()

        # We perform the backward pass on the original meta_model
        # The gradient calculation will be based on the difference between
        # the adapted models and the original meta_model.
        meta_optimizer.zero_grad()
        meta_loss.backward()

        # Optional: Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=1.0)

        meta_optimizer.step()

        per_epoch_time.append(time.time() - t0)
        print(f"  Meta Loss: {meta_loss.item():.4f}\n")

    # --- End of Training ---
    # (Your end-of-training code for saving models etc. remains unchanged)
    # ...


if __name__ == "__main__":
    import argparse

    # Creating a simple args object for demonstration
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--saved', type=str, default='Meta-JSCC/out/checkpoints')
    args = parser.parse_args()

    # Define TASKS here for the script to be self-contained
    TASKS = [
        {'dataset': 'cifar10', 'channel': 'AWGN', 'snr': 0},
        {'dataset': 'cifar10', 'channel': 'AWGN', 'snr': 5},
        {'dataset': 'cifar10', 'channel': 'AWGN', 'snr': 10},
        {'dataset': 'cifar10', 'channel': 'AWGN', 'snr': 15},
    ]

    maml_train(args)