# retrain.py
import os
import torch
import time
from fractions import Fraction
from model import DeepJSCC, ratio2filtersize
from utils import set_seed, image_normalization, view_model_param
from dataset import Cifar10Modified, Vanilla
from torch.nn.parallel import DataParallel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import glob
import yaml
from collections import OrderedDict

def train_epoch(model, optimizer, param, data_loader):
    model.train()
    epoch_loss = 0
    for iter, (images, _) in enumerate(data_loader):
        images = images.cuda() if param['parallel'] and torch.cuda.device_count() > 1 else images.to(param['device'])
        optimizer.zero_grad()
        outputs = model.forward(images)
        outputs = image_normalization('denormalization')(outputs)
        images = image_normalization('denormalization')(images)
        loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(images, outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    return epoch_loss / (iter + 1), optimizer

def evaluate_epoch(model, param, data_loader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for iter, (images, _) in enumerate(data_loader):
            images = images.cuda() if param['parallel'] and torch.cuda.device_count() > 1 else images.to(param['device'])
            outputs = model.forward(images)
            outputs = image_normalization('denormalization')(outputs)
            images = image_normalization('denormalization')(images)
            loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(images, outputs)
            epoch_loss += loss.detach().item()
    return epoch_loss / (iter + 1)

def retrain():
    # === Set parameters manually ===
    resume_path = 'checkpoints/CIFAR10_8_7.0_0.17_Rayleigh_w_noise_RETRAIN_w_Rician_20h52m50s_on_Jul_01_2025/epoch_499.pkl'
    dataset_name = 'cifar10'
    snr = 7.0
    ratio = float(Fraction('1/6'))
    device = 'cuda:0'

    params = {
        'dataset': dataset_name,
        'out_dir': 'RT-JSCC/out',
        'device': device,
        'parallel': False,
        'snr': snr,
        'ratio': ratio,
        'channel': 'AWGN',
        'disable_tqdm': False,
        'resume': resume_path,
        'seed': 42,
        'batch_size': 64,
        'num_workers': 4,
        'epochs': 500,
        'init_lr': 1e-3,
        'weight_decay': 5e-4,
        'if_scheduler': True,
        'step_size': 640,
        'gamma': 0.1,
        'ReduceLROnPlateau': False,
        'lr_reduce_factor': 0.5,
        'lr_schedule_patience': 15,
        'max_time': 12,
        'min_lr': 1e-5,
    }

    set_seed(params['seed'])

    # === Load data ===
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='dataset/', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

    image_first = train_dataset[0][0]
    c = ratio2filtersize(image_first, params['ratio'])

    print("[Retrain] Dataset: {}, SNR: {}, Channel: {}, Ratio: {:.2f}, C: {}".format(params['dataset'], params['snr'], params['channel'], params['ratio'], c))

    # === Build model ===
    model = DeepJSCC(c=c, channel_type=params['channel'], snr=params['snr'])
    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')

    if 'resume' in params and os.path.exists(params['resume']):
        print(f"Loading pretrained weights from {params['resume']}")
        state_dict = torch.load(params['resume'], map_location=device)
        if any(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
            state_dict = new_state_dict
        model.load_state_dict(state_dict)

    if params['parallel'] and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()
    else:
        model = model.to(device)

    # === Setup optimizer ===
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma']) if params['if_scheduler'] else None

    # === Setup logging ===
    timestamp = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    phase = f"{dataset_name.upper()}_{c}_{params['snr']}_{params['ratio']:.2f}_{params['channel']}_RETRAIN_{timestamp}"
    root_log_dir = os.path.join(params['out_dir'], 'logs', phase)
    root_ckpt_dir = os.path.join(params['out_dir'], 'checkpoints', phase)
    root_config_dir = os.path.join(params['out_dir'], 'configs', phase)
    os.makedirs(root_ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=root_log_dir)

    t0 = time.time()
    per_epoch_time = []

    try:
        for epoch in tqdm(range(params['epochs']), disable=params['disable_tqdm']):
            start = time.time()
            train_loss, optimizer = train_epoch(model, optimizer, params, train_loader)
            val_loss = evaluate_epoch(model, params, test_loader)

            writer.add_scalar('train/_loss', train_loss, epoch)
            writer.add_scalar('val/_loss', val_loss, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            per_epoch_time.append(time.time() - start)
            print(f"[Epoch {epoch}] Train: {train_loss:.4f}, Val: {val_loss:.4f}, Time: {per_epoch_time[-1]:.2f}s")

            # Save checkpoint every 10 epochs
            if epoch % 100 == 0 or epoch == params['epochs'] - 1:
                ckpt_path = os.path.join(root_ckpt_dir, f'epoch_{epoch}.pkl')
                torch.save(model.state_dict(), ckpt_path)

            # Learning rate step
            if scheduler:
                scheduler.step()

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("!! Reached min LR. Stopping.")
                break

            if time.time() - t0 > params['max_time'] * 3600:
                print(f"!! Reached max training time ({params['max_time']} hours). Stopping.")
                break

    except KeyboardInterrupt:
        print("Training interrupted manually.")

    test_loss = evaluate_epoch(model, params, test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")

    writer.add_text('result', f"Final Test Loss: {test_loss:.4f}\nAvg Epoch Time: {np.mean(per_epoch_time):.2f}s")
    writer.close()

    os.makedirs(os.path.dirname(root_config_dir), exist_ok=True)
    with open(root_config_dir + '.yaml', 'w') as f:
        yaml.dump({'params': params, 'inner_channel': c, 'total_parameters': view_model_param(model)}, f)

if __name__ == '__main__':
    retrain()
