import os
from xmlrpc.client import Boolean 
import torch 
from tensorboardX import SummaryWriter
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import make_grid
import re
import model.DT_JSCC as JSCC_model
from model.losses import RIBLoss, VAELoss
from datasets.dataloader import get_data
from engine import train_one_epoch, test 
from utils.modulation import QAM, PSK


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



def main(args):
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

    model.to(args.device)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80,gamma=0.5)    

    """ Criterion """
    criterion = RIBLoss(args.lam)
    criterion.train()
    
    """ dataloader """
    dataloader_train =  get_data(args.dataset, args.N, n_worker= 8)
    dataloader_vali = get_data(args.dataset, args.N, n_worker= 8, train=False)
     
    """ writer """
    log_writer = SummaryWriter('JSCC/logs/'+ name)
    
    # fixed_images, _ = next(iter(dataloader_vali))
    # fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    # log_writer.add_image('original', fixed_grid, 0)
    
    current_epoch = 0
    best_acc = 0.0
    """ Some thing wrong here !!"""
    if os.path.isfile(path_to_backup):
        checkpoint = torch.load(path_to_backup, map_location='cpu')
        model.load_state_dict(checkpoint['model_states'])
        optimizer.load_state_dict(checkpoint['optimizer_states'])
        current_epoch = checkpoint['epoch']  
        
    for epoch in range(current_epoch, args.epoches):
        channel_args = {}
        if args.channel == 'rician':
            channel_args['K'] = getattr(args, 'K', 3)
        elif args.channel == 'nakagami':
            channel_args['m'] = getattr(args, 'm', 2)

        if args.mod == 'qam':
            mod = QAM(args.num_embeddings, args.psnr, args.channel, channel_args)
        elif args.mod == 'psk':
            mod = PSK(args.num_embeddings, args.psnr, args.channel, channel_args)
        else:
            raise ValueError(f"Unsupported modulation: {args.mod}")

        train_one_epoch(dataloader_train, model, optimizer=optimizer, criterion=criterion,
                            writer=log_writer, epoch=epoch, mod=mod, args=args)
        scheduler.step()
        if (epoch >100): 
            acc1 = test(dataloader_vali, model, criterion=criterion, writer=log_writer, epoch=epoch, mod=mod, args=args)
        
            print('Epoch ', epoch)
            print('Best accuracy: ', best_acc)
        
            if (epoch == 0) or (acc1 > best_acc):
                best_acc = acc1
                with open('{0}/best.pt'.format(path_to_backup), 'wb') as f:
                    torch.save(
                    {
                    'epoch': epoch, 
                    'model_states': model.state_dict(), 
                    'optimizer_states': optimizer.state_dict(),
                    }, f
                )
        with open('{0}/model_{1}.pt'.format(path_to_backup, epoch + 1), 'wb') as f:
            torch.save(
            {
                'epoch': epoch, 
                'model_states': model.state_dict(), 
                'optimizer_states': optimizer.state_dict(),
            }, f 
        )
            


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VDL')
    
    parser.add_argument('-d', '--dataset', type=str, default='CINIC10', help='dataset name')
    parser.add_argument('-r', '--root', type=str, default='JSCC/trained_models', help='The root of trained models')
    parser.add_argument('--device', type=str, default='cuda:0', help= 'The device')
    
    parser.add_argument('--mod', type=str, default='psk', help='The modulation')

    parser.add_argument('--num_latent', type=int, default=4, help='The number of latent variable')
    parser.add_argument('--latent_d', type=int, default=512, help='The dimension of latent vector')
    parser.add_argument('--in_channels', type=int, default=3, help='The image input channel')
    
    parser.add_argument('-e', '--epoches', type=int, default=150, help='Number of epoches')
    parser.add_argument('--N', type=int, default=512, help='The batch size of training data')
    parser.add_argument('--lr', type=float, default=1e-3, help='learn rate')
    parser.add_argument('--maxnorm', type=float, default=1., help='The max norm of flip')
    
    parser.add_argument('--num_embeddings', type=int, default=16, help='The size of codebook')

    parser.add_argument('--lam', type=float, default=0.0, help='The lambda' )
    parser.add_argument('--psnr', type=float, default=8.0, help='The psnr' )

    parser.add_argument('--num_workers', type=int, default=0,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))

    parser.add_argument('--channel', type=str, default='awgn',
                        choices=['awgn', 'rayleigh', 'rician', 'nakagami'],
                        help='Channel type: awgn, rayleigh, rician, nakagami')

    args = parser.parse_args()
    args.num_classes = infer_num_classes(args.dataset)
    args.n_iter = 0
    name = args.dataset + '-'+ str(args.channel)
 
    path_to_backup = os.path.join(args.root, name)
    if not os.path.exists(path_to_backup):
        print('Making ', path_to_backup, '...')
        os.makedirs(path_to_backup)


    device = torch.device(args.device if(torch.cuda.is_available()) else "cpu")
    print('Device: ', device)

    main(args)
