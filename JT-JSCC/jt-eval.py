import os
import torch
from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy
import re

def infer_num_classes(dataset_name: str) -> int:
    name = dataset_name.upper()
    if name.startswith("CIFAR100"):
        return 100
    if name.startswith("CIFAR10"):
        return 10
    if name.startswith("CINIC10"):
        return 10
    # fallback: keep user-specified value (or raise)
    raise ValueError(f"Unknown dataset for inferring num_classes: {dataset_name}")


def eval_test(dataloader, model, mod, args, psnr):
    acc1, acc3 = 0., 0.
    with torch.no_grad():
        model.eval()
        for imgs, labs in dataloader:
            imgs = imgs.to(args.device)
            labs = labs.to(args.device)

            outs, dist = model(imgs, mod=mod)
            acc = accuracy(outs, labs, (1, 3))
            acc1 += acc[0].item()
            acc3 += acc[1].item()

    acc1 /= len(dataloader)
    acc3 /= len(dataloader)

    print(f"[Eval] PSNR = {psnr} dB | Acc@1 = {acc1:.4f} | Acc@3 = {acc3:.4f}")

    return acc1, acc3, dist


def main(args):
    print(f"\n[Config] Dataset: {args.dataset} | Modulation: {args.mod.upper()} | Channel: {args.channel.upper()}\n")
    ds = str(args.dataset).strip().lower()
    # Load model
    if re.match(r'^cifar100(\b|[_-])', ds):
        model = JSCC_model.DTJSCC_CIFAR100(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif re.match(r'^cifar10(\b|[_-])', ds):
        model = JSCC_model.DTJSCC_CIFAR10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    elif re.match(r'^cinic10(\b|[_-])', ds):
        model = JSCC_model.DTJSCC_CINIC10(
            args.in_channels, args.latent_d, args.num_classes,
            num_embeddings=args.num_embeddings
        )
    else:
        raise ValueError(
            f"No available model for dataset: {args.dataset}, please check model/DT_JSCC.py."
        )

    dataloader_vali = get_data(args.dataset, 256, n_worker=0, train=False)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_states'])
    model.to(args.device)

    PSNRs = list(range(0, 26, 1))
    acc1s, acc3s = [], []
    dist_re = None

    for psnr in PSNRs:
        channel_args = {}
        if args.channel == 'rician':
            channel_args['K'] = args.K
        elif args.channel == 'nakagami':
            channel_args['m'] = args.m

        if args.mod == 'qam':
            mod = QAM(args.num_embeddings, psnr, args.channel, channel_args)
        elif args.mod == 'psk':
            mod = PSK(args.num_embeddings, psnr, args.channel, channel_args)
        else:
            raise ValueError(f"Unsupported modulation: {args.mod}")

        a1, a3, dist = eval_test(dataloader_vali, model, mod, args, psnr)
        acc1s.append(a1)
        acc3s.append(a3)
        dist_re = dist

    save_dict = {
        'acc1s': acc1s,
        'acc3s': acc3s,
        'dist': dist_re
    }

    # Add channel info to filename
    result_file = os.path.join(
        args.save_dir,
        f"jt-eval-{args.dataset}-{args.channel}.pt"
    )
    torch.save(save_dict, result_file)
    print(f"\n[Done] Results saved to {result_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained JSCC model under varying PSNRs')

    parser.add_argument('--mod', type=str, default='psk', help='Modulation type: psk or qam')
    parser.add_argument('--num_latent', type=int, default=4, help='Number of latent variables')
    parser.add_argument('--latent_d', type=int, default=512, help='Latent dimension')
    parser.add_argument('--dataset', type=str, default='CIFAR100_noise', help='Dataset: MNIST or CIFAR10 or CIFAR10_noise / CIFAR10_blur,,,' )
    parser.add_argument('--model_dir', type=str, default='JT-JSCC/trained_models', help='Root of trained models')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--num_embeddings', type=int, default=16, help='Codebook size')
    parser.add_argument('--in_channels', type=int, default=3, help='The image input channel')
    parser.add_argument('--save_dir', type=str, default='JT-JSCC/eval', help='Directory to save results')
    parser.add_argument('--channel', type=str, default='rician',
                        choices=['awgn', 'rayleigh', 'rician', 'nakagami'],
                        help='Channel type: awgn, rayleigh, rician, nakagami')
    parser.add_argument('--K', type=float, default=3.0, help='Rician K-factor (only used if channel is rician)')
    parser.add_argument('--m', type=float, default=2.0, help='Nakagami m parameter (only used if channel is nakagami)')

    args = parser.parse_args()
    args.num_classes = infer_num_classes(args.dataset)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Device:', args.device)

    model_dir = os.path.join(args.model_dir)
    args.model_path = os.path.join(model_dir, 'cifar100_best.pt')

    if not os.path.exists(args.save_dir):
        print(f'Creating result directory at {args.save_dir}...')
        os.makedirs(args.save_dir)

    args.result_path = os.path.join(args.save_dir)

    main(args)
