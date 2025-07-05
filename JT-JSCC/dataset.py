import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import pickle

class Vanilla(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = os.listdir(root)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.imgs[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0  # 0 is a fake label not important

    def __len__(self):
        return len(self.imgs)



class Cifar10Modified(Dataset):
    def __init__(self, dataset_name, root='dataset/cifar-10-odd-batches-py', transform=None, train=False):
        self.transform = transform
        self.data = []
        self.labels = []

        if dataset_name == 'cifar10':
            if train:
                filenames = [f'data_batch_{i}' for i in range(1, 6)]
                subdir = root
            else:
                filenames = ['test_batch']
                subdir = root
        elif dataset_name.startswith('cifar10_'):
            variant = dataset_name[len('cifar10_'):]  # e.g., 'blur', 'bright'
            if train:
                filenames = [f'data_batch_{i}' for i in range(1, 6)]
                subdir = os.path.join(root, variant)  # e.g., dataset/cifar-10-odd-batches-py/blur
            else:
                filenames = [f'test_batch_{variant}']
                subdir = root
        else:
            raise ValueError(f"Unsupported dataset name for Cifar10Modified: {dataset_name}")

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
def main():
    data_path = './dataset'
    os.makedirs(data_path, exist_ok=True)
    # ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar should be downloaded from https://image-net.org/
    if not os.path.exists('./dataset/ILSVRC2012_img_train.tar') or not os.path.exists('./dataset/ILSVRC2012_img_val.tar'):
        print('ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar should be downloaded from https://image-net.org/')
        print('Please download the dataset from https://image-net.org/challenges/LSVRC/2012/2012-downloads and put it in ./dataset')
        raise Exception('not find dataset')
    phases = ['train', 'val']
    for phase in phases:
        print("extracting {} dataset".format(phase))
        path = './dataset/ImageNet/{}'.format(phase)
        print('path is {}'.format(path))
        os.makedirs(path, exist_ok=True)
        print('tar -xf ./dataset/ILSVRC2012_img_{}.tar -C {}'.format(phase, path))
        os.system('tar -xf ./dataset/ILSVRC2012_img_{}.tar -C {}'.format(phase, path))
        if phase == 'train':
            for tar in os.listdir(path):
                print('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
                os.makedirs('{}/{}'.format(path, tar.split('.')[0]), exist_ok=True)
                os.system('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
                os.remove('{}/{}'.format(path, tar))





if __name__ == '__main__':
    main()
