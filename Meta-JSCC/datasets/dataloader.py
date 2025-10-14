import torch
import torchvision.datasets as dsets
import os
from torch.utils.data import Dataset
from .transform import simple_transform_mnist, simple_transform, cencrop_teransform, imagenet_transform, simple_transform_test, imagenet_transform_aug
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
root = r'dataset/'

import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class CustomCIFAR10(Dataset):
    def __init__(self, dataset_name, root='dataset/cifar-10-odd-batches-py', transform=None, train=False):
        self.transform = transform
        self.data = []
        self.labels = []

        if dataset_name == 'CIFAR10':
            # Standard CIFAR-10 path
            if train:
                filenames = [f'data_batch_{i}' for i in range(1, 6)]
                subdir = root
            else:
                filenames = ['test_batch']
                subdir = root

        elif dataset_name.startswith('CIFAR10_'):
            variant = dataset_name[len('CIFAR10_'):]  # e.g., 'blur', 'bright', etc.
            if train:
                # Train batches are inside subdir like 'blur', 'bright'
                filenames = [f'data_batch_{i}' for i in range(1, 6)]
                subdir = os.path.join(root, variant)
            else:
                # Test batches are at root, e.g., 'test_batch_blur'
                filenames = [f'test_batch_{variant}']
                subdir = root
        else:
            raise ValueError(f"Unsupported dataset name for CustomCIFAR10: {dataset_name}")

        for fname in filenames:
            filepath = os.path.join(subdir, fname)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            with open(filepath, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                batch_data = entry['data'].reshape((-1, 3, 32, 32)).astype(np.uint8)
                batch_labels = entry.get('labels', entry.get('fine_labels'))
                self.data.append(batch_data)
                self.labels.extend(batch_labels)

        self.data = np.vstack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img, label



class CustomCIFAR100(Dataset):
    """
    Supports:
      - dataset_name='CIFAR100'                  -> root/{train,test}
      - dataset_name='CIFAR100_<variant>'       -> root/<variant>/{train,test}
    Default root matches your generated OOD structure: 'cifar-100-ood-python'
    (works as well with the official 'cifar-100-python' if no variants).
    """
    def __init__(self, dataset_name, root='cifar-100-ood-python', transform=None, train=False):
        self.transform = transform
        self.data = []
        self.labels = []          # fine labels
        self.coarse_labels = []   # optional: kept if present

        if dataset_name == 'CIFAR100':
            # Standard CIFAR-100 path: single 'train' / 'test' file
            filenames = ['train'] if train else ['test']
            subdir = root
        elif dataset_name.startswith('CIFAR100_'):
            # Variant lives in a subdir (e.g., root/blur/train or root/blur/test)
            variant = dataset_name[len('CIFAR100_'):]
            filenames = ['train'] if train else ['test']
            subdir = os.path.join(root, variant)
        else:
            raise ValueError(f"Unsupported dataset name for CustomCIFAR100: {dataset_name}")

        for fname in filenames:
            filepath = os.path.join(subdir, fname)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            with open(filepath, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')  # ensure str keys
                # CIFAR-100 uses 'data' shaped (N, 3072)
                batch_data = entry['data'].reshape((-1, 3, 32, 32)).astype(np.uint8)
                # Prefer fine_labels; fall back to labels if needed
                batch_labels = entry.get('fine_labels', entry.get('labels'))
                if batch_labels is None:
                    raise KeyError("Missing 'fine_labels' (or 'labels') in CIFAR-100 pickle.")
                self.data.append(batch_data)
                self.labels.extend(batch_labels)

                # Keep coarse labels if available (not returned by default)
                if 'coarse_labels' in entry:
                    self.coarse_labels.extend(entry['coarse_labels'])

        self.data = np.vstack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]                     # (C, H, W), uint8
        label = self.labels[idx]                 # fine label
        img = np.transpose(img, (1, 2, 0))      # CHW -> HWC
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img, label



class CustomCINIC10Python(Dataset):
    """
    Read CINIC-10 pickles in CIFAR-like format:
      base root (default): 'dataset/cinic-10-python'
        - files: {train, valid, test, meta}
      OOD root: 'dataset/cinic-10-ood-python/<variant>/{train, valid, test, meta}'

    dataset_name:
      - 'CINIC10'              -> use base root
      - 'CINIC10_<variant>'    -> use OOD root/<variant>
      - optionally 'CINIC10_valid' to force the 'valid' split
    """
    def __init__(self, dataset_name, root='dataset/cinic-10-python', transform=None,
                 train=False, split=None):
        self.transform = transform
        # decide base dir (support variants under OOD root)
        if dataset_name == 'CINIC10':
            base_dir = root
        elif dataset_name.lower() in ('cinic10_valid', 'cinic10_val'):
            base_dir = root
            split = 'valid'
        elif dataset_name.startswith('CINIC10_'):
            variant = dataset_name[len('CINIC10_'):]  # e.g., 'fog'
            base_dir = os.path.join(root, variant)    # when root is the OOD root
        else:
            raise ValueError(f"Unsupported dataset name for CustomCINIC10Python: {dataset_name}")

        # choose split
        if split is not None:
            split_name = split.lower()
            assert split_name in {'train', 'valid', 'test'}, f"Invalid split: {split}"
        else:
            split_name = 'train' if train else 'test'

        file_path = os.path.join(base_dir, split_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Pickle not found: {file_path}")

        # load pickle (CIFAR-like)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')

        data = entry['data'].reshape((-1, 3, 32, 32)).astype(np.uint8)  # (N, C, H, W)
        labels = entry.get('fine_labels', entry.get('labels'))
        if labels is None:
            raise KeyError("Missing 'fine_labels' (or 'labels') in CINIC-10 pickle.")

        self.data = data
        self.labels = labels
        self.filenames = entry.get('filenames', [f'{split_name}_{i:05d}.png' for i in range(len(labels))])
        self.batch_label = entry.get('batch_label', split_name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]                       # (C,H,W)
        img = np.transpose(img, (1,2,0))          # -> (H,W,C)
        img = Image.fromarray(img)
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)
        return img, label

def get_data(data_set, batch_size, shuffle=True, n_worker=0, train = True, add_noise=0):
    if data_set == 'MNIST':
        tran = simple_transform_mnist()
        dataset = dsets.MNIST(root, train=train, transform=tran, target_transform=None, download=False)



    elif data_set == 'CIFAR10':
        if train:
            tran = simple_transform(32)
        else:
            tran = simple_transform_test(32)
        dataset = dsets.CIFAR10(root, train=train, transform=tran, target_transform=None, download=False)



    elif data_set.startswith('CIFAR10_'):
        path = 'dataset/cifar-10-odd-batches-py'
        tran = simple_transform(32) if train else simple_transform_test(32)
        dataset = CustomCIFAR10(data_set, root=path, train=train, transform=tran)



    elif data_set == 'CIFAR100':
        tran = simple_transform(32)
        dataset = dsets.CIFAR100(root, train=train, transform=tran, target_transform=None, download=False)



    elif data_set.startswith('CIFAR100_'):
        path = 'dataset/cifar-100-ood-python'
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Expected CIFAR-100 OOD root at '{path}'. "
                                    f"Please generate it or set the correct path.")
        tran = simple_transform(32) if train else simple_transform_test(32)
        dataset = CustomCIFAR100(data_set, root=path, train=train, transform=tran)



    elif data_set == 'CINIC10':
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        tran = (T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize(cinic_mean, cinic_std),
        ]) if train else
                T.Compose([
                    T.ToTensor(),
                    T.Normalize(cinic_mean, cinic_std),
                ]))
        dataset = CustomCINIC10Python('CINIC10',
                                      root=os.path.join(root, 'cinic-10-python').rstrip('/'),
                                      transform=tran, train=train)

        # optional: explicitly use the validation split
    elif data_set in ('CINIC10_valid', 'CINIC10_val'):
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        tran = T.Compose([T.ToTensor(), T.Normalize(cinic_mean, cinic_std)])
        dataset = CustomCINIC10Python('CINIC10_valid',
                                      root=os.path.join(root, 'cinic-10-python').rstrip('/'),
                                      transform=tran, split='valid')

        # ===== CINIC-10 OOD (python) variants, e.g., CINIC10_fog / CINIC10_blur ... =====
    elif data_set.startswith('CINIC10_'):
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        tran = (T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize(cinic_mean, cinic_std),
        ]) if train else
                T.Compose([
                    T.ToTensor(),
                    T.Normalize(cinic_mean, cinic_std),
                ]))
        # 注意：root 指向 OOD 根：dataset/cinic-10-ood-python
        ood_root = os.path.join(root, 'cinic-10-ood-python')
        if not os.path.isdir(ood_root):
            raise FileNotFoundError(f"Expected CINIC-10 OOD root at '{ood_root}'. "
                                    f"Please generate it or set the correct path.")
        dataset = CustomCINIC10Python(data_set, root=ood_root, transform=tran, train=train)

    # To BE DONE:
    # elif data_set == 'CelebA':
    #     tran = cencrop_teransform(168, resize=(128,128))
    #     split = 'train' if train else 'test'
    #     dataset = dsets.CelebA(root+'CelebA/', split=split, transform=tran, target_transform=None, download=False)
    # elif data_set == 'STL10':
    #     tran = simple_transform(96)
    #     split = 'train+unlabeled' if train else 'test'
    #     folds = None # For valuation
    #     dataset = dsets.STL10(root+'STL10/', split=split, folds=folds, transform=tran, target_transform=None, download=False)
    # elif data_set == 'Caltech101':
    #     tran = cencrop_teransform(300, resize=(256,256))
    #     dataset = dsets.Caltech101(root+'Caltech101', transform=tran, target_transform=None, download=False)
    # elif data_set == 'Caltech256':
    #     tran = cencrop_teransform(168)
    #     dataset = dsets.Caltech256(root+'Caltech256', transform=tran, target_transform=None, download=False)
    # elif data_set == 'Imagenet':
    #     tran = imagenet_transform(64)
    #     split = 'train' if train else 'val'
    #     way = os.path.join(root+'ImageNet/imagenet-mini-100', split)
    #     dataset = dsets.ImageFolder(way, tran)
    else:
        print('Sorry! Cannot support ...')
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_worker)
    return dataloader


if __name__ == '__main__':
    from utils import play_show
    import matplotlib.pyplot as plt
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    cpu = torch.device("cpu")
    dl_1 = get_data('Imagenet', 100, shuffle=True)
    data, _ = next(iter(dl_1))
    print(data.shape)
    print(_)
    play_show(data, device)
    plt.show()
    


        
