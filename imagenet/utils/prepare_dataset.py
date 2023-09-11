import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
import os
import torchvision

def prepare_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_corrupt_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform, val_transform, val_corrupt_transform

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(dataset, args, shuffle=False, drop_last=False):
    return torch.utils.data.DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                shuffle=shuffle, 
                                num_workers=args.workers, 
                                worker_init_fn=seed_worker, 
                                pin_memory=True, 
                                drop_last=drop_last)


class ImageNetCorruption(ImageNet):
    def __init__(self, root, corruption_name="gaussian_noise", transform=None, is_carry_index=False):
        super().__init__(root, 'val', transform=transform)
        self.root = root
        self.corruption_name = corruption_name
        self.transform = transform
        self.is_carry_index = is_carry_index
        self.load_data()
    
    def load_data(self):
        self.data = torch.load(os.path.join(self.root, 'corruption', self.corruption_name + '.pth')).numpy()
        self.target = [i[1] for i in self.imgs]
        return
    
    def __getitem__(self, index):
        img = self.data[index, :, :, :]
        target = self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.is_carry_index:
            img = [img, index]
        return img, target
    
    def __len__(self):
        return self.data.shape[0]

class ImageNet_(ImageNet):
    def __init__(self, *args, is_carry_index=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_carry_index = is_carry_index
    
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        if self.is_carry_index:
            if type(img) == list:
                img.append(index)
            else:
                img = [img, index]
        return img, target

 
class noise_dataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self, transform,ratio=1):
        #定义好 image 的路径
        self.number = int(50000*ratio)
        self.transform = transform

    def __getitem__(self, index:int):
        image = torch.randn(3,224,224)
        target = 1000
        # if self.transform is not None:
        #     image = self.transform(image)
        if type(image) == list:
                image.append(index)
        else:
            image = [image, index]

        return image, target

    def __len__(self):

        return self.number
    
class imageneta(torchvision.datasets.ImageFolder):#需要继承data.Dataset
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.is_carry_index = is_carry_index
    
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)

        if type(img) == list:
            img.append(index)
        else:
            img = [img, index]
        return img, target
    

class MNIST_openset(torchvision.datasets.MNIST):
    def __init__(self, *args, ratio = 1 , **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.targets = self.data[:int(50000*ratio)], self.targets[:int(50000*ratio)]
        print(ratio)
        print(len(self.data))
        return

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target = target + 1000
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target


class SVHN_openset(torchvision.datasets.SVHN):
    def __init__(self, *args, ratio = 1 , **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.labels = self.data[:int(50000*ratio)], self.labels[:int(50000*ratio)]
        print(ratio)
        print(len(self.data))
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target = target + 1000
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target


def prepare_ood_test_data_a(root, corruption_name="gaussian_noise", transform=None, is_carry_index=False, OOD = 'noise', OOD_transform=None):
    teset_seen = imageneta(root='/cluster/personal/dataset/imagenet-a', transform=OOD_transform)
    print(len(teset_seen))
    if OOD =='noise':
        teset_unseen = noise_dataset(transform,ratio=0.15)
    elif OOD=='SVHN':
        teset_unseen = SVHN_openset(root="/cluster/personal/dataset/CIFAR-C",
                            split='train', download=True, transform=OOD_transform, ratio=0.15)
    elif OOD=='MNIST':
        te_rize = transforms.Compose([transforms.Grayscale(3), OOD_transform ])
        teset_unseen = MNIST_openset(root="/cluster/personal/dataset/CIFAR-C",
                    train=True, download=True, transform=te_rize, ratio=0.15)
    teset = torch.utils.data.ConcatDataset([teset_seen,teset_unseen])
    return teset

def prepare_ood_test_data_r(root, corruption_name="gaussian_noise", transform=None, is_carry_index=False, OOD = 'noise', OOD_transform=None):
    teset_seen = imageneta(root='/cluster/personal/dataset/imagenet-r', transform=OOD_transform)
    print(len(teset_seen))
    if OOD =='noise':
        teset_unseen = noise_dataset(transform,ratio=0.6)
    elif OOD=='SVHN':
        teset_unseen = SVHN_openset(root="/cluster/personal/dataset/CIFAR-C",
                            split='train', download=True, transform=OOD_transform, ratio=0.6)
    elif OOD=='MNIST':
        te_rize = transforms.Compose([transforms.Grayscale(3), OOD_transform ])
        teset_unseen = MNIST_openset(root="/cluster/personal/dataset/CIFAR-C",
                    train=True, download=True, transform=te_rize, ratio=0.6)
    teset = torch.utils.data.ConcatDataset([teset_seen,teset_unseen])
    return teset

def prepare_test_data_r(root, corruption_name="gaussian_noise", transform=None, is_carry_index=False, OOD = 'noise', OOD_transform=None):
    teset_seen = imageneta(root=root+'/imagenet-r', transform=OOD_transform)
    return teset_seen

def prepare_ood_test_data(root, corruption_name="gaussian_noise", transform=None, is_carry_index=False, OOD = 'noise', OOD_transform=None):
    teset_seen = ImageNetCorruption(root, corruption_name, transform=transform, is_carry_index=is_carry_index)
    if OOD =='noise':
        teset_unseen = noise_dataset(transform)
    elif OOD=='SVHN':
        teset_unseen = SVHN_openset(root="/cluster/personal/dataset/CIFAR-C",
                            split='train', download=True, transform=OOD_transform, ratio=1)
    elif OOD=='MNIST':
        te_rize = transforms.Compose([transforms.Grayscale(3), OOD_transform ])
        teset_unseen = MNIST_openset(root="/cluster/personal/dataset/CIFAR-C",
                    train=True, download=True, transform=te_rize, ratio=1)
    teset = torch.utils.data.ConcatDataset([teset_seen,teset_unseen])
    return teset
