import os
import re
from xmlrpc.client import Boolean  # (kept as in original; appears unused)
import torch
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import make_grid

import model.DT_JSCC as JSCC_model
from model.losses import RIBLoss
from datasets.dataloader import get_data
from engine import train_one_epoch, test
from utils.modulation import QAM, PSK
import time

# ----------------------------- regex helpers ------------------------------ #
# Match dataset base at the beginning: cifar100 / cifar10 / cinic10,
# followed by a word boundary, underscore, hyphen, or end of string.
_DATASET_BASE_RE = re.compile(r'^(cifar100|cifar10|cinic10)(?=\b|[_-]|$)', flags=re.IGNORECASE)

def detect_base(dataset_name: str) -> str:
    """Return canonical base in UPPER (CIFAR10 / CIFAR100 / CINIC10) via regex."""
    ds = str(dataset_name).strip()
    m = _DATASET_BASE_RE.match(ds)
    if not m:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return m.group(1).upper()

def infer_num_classes(dataset_name: str) -> int:
    """Infer number of classes based on regex-detected base."""
    base = detect_base(dataset_name)
    if base == "CIFAR100":
        return 100
    if base in ("CIFAR10", "CINIC10"):
        return 10
    # Unreachable due to detect_base guard
    raise ValueError(f"Unknown dataset for inferring num_classes: {dataset_name}")


def main(args):
    # ---------------- Model and Optimizer ---------------- #
    base = detect_base(args.dataset)  # "CIFAR10" / "CIFAR100" / "CINIC10"

    # Select model by regex base (supports dataset variants like CIFAR10_noise)
    if base == 'CINIC10':
        model = JSCC_model.DTJSCC_CINIC10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif base == 'CIFAR10':
        model = JSCC_model.DTJSCC_CIFAR10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif base == 'CIFAR100':
        model = JSCC_model.DTJSCC_CIFAR100(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    else:
        # Should not happen due to detect_base guard
        raise ValueError(f"No available model for dataset: {args.dataset}, please check model/DT_JSCC.py.")

    model.to(args.device)

    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    # ---------------- Criterion ---------------- #
    criterion = RIBLoss(args.lam)
    criterion.train()

    # ---------------- Dataloaders ---------------- #
    # Keep using the original args.dataset string (may include suffix), so that
    # get_data can route to the corresponding variant.
    dataloader_train = get_data(args.dataset, args.N, n_worker=8)
    dataloader_vali  = get_data(args.dataset, args.N, n_worker=8, train=False)

    # ---------------- Writer ---------------- #
    # Reuse external 'name' variable as in the original script.
    log_writer = SummaryWriter('RT-JSCC/logs/' + name)

    current_epoch = 0
    best_acc = 0.0

    # ---------------- Resume from checkpoint (optional) ---------------- #
    if os.path.isfile(path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_states'])
        optimizer.load_state_dict(checkpoint['optimizer_states'])
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"[Checkpoint] Loaded from {path_to_checkpoint}, reset epoch to 0 for retraining.")

    # ---------------- Training loop ---------------- #
    for epoch in range(args.epoches):
        # Build channel args per selected channel
        channel_args = {}
        if args.channel == 'rician':
            channel_args['K'] = getattr(args, 'K', 3)
        elif args.channel == 'nakagami':
            channel_args['m'] = getattr(args, 'm', 2)

        # Modulator selection
        if args.mod == 'qam':
            mod = QAM(args.num_embeddings, args.psnr, args.channel, channel_args)
        elif args.mod == 'psk':
            mod = PSK(args.num_embeddings, args.psnr, args.channel, channel_args)
        else:
            raise ValueError(f"Unsupported modulation: {args.mod}")

        # Train one epoch
        train_one_epoch(
            dataloader_train, model, optimizer=optimizer, criterion=criterion,
            writer=log_writer, epoch=epoch, mod=mod, args=args
        )
        scheduler.step()

        # Validate after 100 epochs (as in original)
        if epoch > 100:
            acc1 = test(
                dataloader_vali, model, criterion=criterion, writer=log_writer,
                epoch=epoch, mod=mod, args=args
            )

            print('Epoch', epoch)
            print('Best accuracy:', best_acc)

            if acc1 > best_acc:
                best_acc = acc1
                with open(os.path.join(path_to_save, 'best.pt'), 'wb') as f:
                    torch.save({
                        'epoch': epoch,
                        'model_states': model.state_dict(),
                        'optimizer_states': optimizer.state_dict(),
                        'best_acc': best_acc,
                    }, f)

        # Always save epoch checkpoints
        with open(os.path.join(path_to_save, f'model_{epoch}.pt'), 'wb') as f:
            torch.save({
                'epoch': epoch,
                'model_states': model.state_dict(),
                'optimizer_states': optimizer.state_dict(),
                'best_acc': best_acc,
            }, f)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    start_time = time.time()
    print('Number of workers (default: {0})'.format(mp.cpu_count() - 1))

    parser = argparse.ArgumentParser(description='Retrain JSCC')

    parser.add_argument('-d', '--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')

    parser.add_argument('--mod', type=str, default='psk', help='Modulation type')
    parser.add_argument('--channel', type=str, default='rician',
                        choices=['awgn', 'rayleigh', 'rician', 'nakagami'], help='Channel type')
    parser.add_argument('--K', type=float, default=3.0, help='Rician K-factor')
    parser.add_argument('--m', type=float, default=2.0, help='Nakagami m-parameter')

    parser.add_argument('--num_latent', type=int, default=4, help='Number of latent variables')
    parser.add_argument('--latent_d', type=int, default=512, help='Latent vector dimension')
    parser.add_argument('--in_channels', type=int, default=3, help='The image input channel')

    parser.add_argument('--epoches', type=int, default=50, help='Number of epochs')
    parser.add_argument('--N', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--maxnorm', type=float, default=1., help='Max norm')
    parser.add_argument('--num_embeddings', type=int, default=16, help='Codebook size')
    parser.add_argument('--lam', type=float, default=0.0, help='Lambda')
    parser.add_argument('--psnr', type=float, default=8.0, help='PSNR value')

    parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1, help='Number of data loader workers')

    args = parser.parse_args()
    args.n_iter = 0

    # Build a run name; keep original pattern but works with dataset variants


    # You can still override these manually if desired; here we keep the original defaults.
    # Derive a reasonable default checkpoint path using the regex base if needed.
    base_upper = detect_base(args.dataset)       # "CIFAR10" / "CIFAR100" / "CINIC10"
    name = f"{base_upper}-rt-{args.dataset}-{args.channel}"
    # Example: use base 'awgn' best as init (matches original intent)
    # path_to_checkpoint = os.path.join('RT-JSCC/retrained_models/CIFAR10_noise-awgn-rt-CIFAR10_noise-rician', 'model_49.pt')
    path_to_checkpoint = os.path.join('JSCC/trained_models', f'{base_upper}-awgn', 'best.pt')

    path_to_save = os.path.join('RT-JSCC/retrained_models', name)
    if not os.path.exists(path_to_save):
        print(f'Making {path_to_save}...')
        os.makedirs(path_to_save)

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Device:', args.device)

    # Regex-based class inference
    args.num_classes = infer_num_classes(args.dataset)

    main(args)
    end_time = time.time()
    duration = end_time - start_time
    print(f"[Time] Total retraining time: {duration:.2f} seconds")
