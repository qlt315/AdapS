import os
import torch
from datasets.dataloader import get_data
import model.DT_JSCC as JSCC_model
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy
import argparse
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
    correct1 = correct3 = samples = 0
    with torch.no_grad():
        model.eval()
        for imgs, labs in dataloader:
            imgs, labs = imgs.to(args.device), labs.to(args.device)
            outs, _ = model(imgs, mod=mod)
            top1, top3 = accuracy(outs, labs, (1, 3))
            bs = imgs.size(0)
            correct1 += top1.item() / 100 * bs
            correct3 += top3.item() / 100 * bs
            samples += bs
    acc1 = correct1 / samples * 100
    acc3 = correct3 / samples * 100
    print(f"[Eval] SNR = {psnr} dB | Acc@1 = {acc1:.4f} | Acc@3 = {acc3:.4f}")
    return acc1, acc3


def main(args):
    print(f"\n[Config] Dataset: {args.dataset} | Modulation: {args.mod.upper()} | Channel: {args.channel.upper()}\n")
    ds = str(args.dataset).strip().lower()

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

    dataloader_vali = get_data(args.dataset, 256, n_worker=8, train=False)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_states'])
    model.to(args.device)

    PSNRs = list(range(0, 26, 1))
    acc1s, acc3s = [], []

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

        acc1_runs, acc3_runs = [], []
        for _ in range(args.eval_times):
            a1, a3 = eval_test(dataloader_vali, model, mod, args, psnr)
            acc1_runs.append(a1)
            acc3_runs.append(a3)

        mean_acc1 = sum(acc1_runs) / args.eval_times
        mean_acc3 = sum(acc3_runs) / args.eval_times
        acc1s.append(mean_acc1)
        acc3s.append(mean_acc3)
        print(f"[Avg ] SNR = {psnr} dB | Acc@1 = {mean_acc1:.4f} | Acc@3 = {mean_acc3:.4f}\n")

    save_dict = {
        'acc1s': acc1s,
        'acc3s': acc3s,
    }

    result_file = f'{args.result_path}-eval-{args.dataset}-{args.channel}.pt'
    torch.save(save_dict, result_file)
    print(f"\n[Done] Results saved to {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained JSCC model under varying PSNRs')
    parser.add_argument('--mod', type=str, default='psk', help='Modulation type: psk or qam')
    parser.add_argument('--num_latent', type=int, default=4)
    parser.add_argument('--latent_d', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='CINIC10_noise')
    parser.add_argument('--model_dir', type=str, default='JSCC/trained_models')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_embeddings', type=int, default=16)
    parser.add_argument('--name', type=str, default='CINIC10-awgn')
    parser.add_argument('--save_dir', type=str, default='JSCC/eval')
    parser.add_argument('--channel', type=str, default='rician', choices=['awgn', 'rayleigh', 'rician', 'nakagami'])
    parser.add_argument('--in_channels', type=int, default=3, help='The image input channel')
    parser.add_argument('--K', type=float, default=3.0)
    parser.add_argument('--m', type=float, default=2.0)
    parser.add_argument('--eval_times', type=int, default=1, help='Number of evaluations per PSNR point')

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.num_classes = infer_num_classes(args.dataset)

    model_dir = os.path.join(args.model_dir, args.name)
    args.model_path = os.path.join(model_dir, 'best.pt')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.result_path = os.path.join(args.save_dir, args.name)

    main(args)
