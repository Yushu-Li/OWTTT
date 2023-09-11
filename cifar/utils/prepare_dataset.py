import os
import sys
import torch
import random
import torchvision
import numpy as np
from PIL import Image
import torch.utils.data
import torchvision.transforms as transforms


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target

class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target
    
    
class CIFAR100_openset(torchvision.datasets.CIFAR100):
    def __init__(self,ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.targets = self.data[:int(10000*ratio)], self.targets[:int(10000*ratio)]
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target = target + 1000
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target

class CIFAR10_openset(torchvision.datasets.CIFAR10):
    def __init__(self,ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.targets = self.data[:int(10000*ratio)], self.targets[:int(10000*ratio)]
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target = target + 1000
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target

class noise_dataset(torch.utils.data.Dataset):
    def __init__(self, transform,ratio=1): 
        self.number = int(10000*ratio)
        self.transform = transform

    def __getitem__(self, index:int):
        image = torch.randn(3,32,32)
        target = 1000
        if type(image) == list:
                image.append(index)
        else:
            image = [image, index]

        return image, target

    def __len__(self):

        return self.number

class MNIST_openset(torchvision.datasets.MNIST):
    def __init__(self, *args, ratio = 1 , **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.targets = self.data[:int(10000*ratio)], self.targets[:int(10000*ratio)]
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
        self.data, self.labels = self.data[:int(10000*ratio)], self.labels[:int(10000*ratio)]
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target = target + 1000
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target


class TinyImageNet_OOD_nonoverlap(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None,list=True,ratio=1):
        self.Train = train
        self.list=list
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.ratio = ratio
        
        self.class_list = ['n03544143', 'n03255030', 'n04532106', 'n02669723', 'n02321529', 'n02423022', 'n03854065', 'n02509815', 'n04133789', 'n03970156', 'n01882714', 'n04023962', 'n01768244', 'n04596742', 'n03447447', 'n03617480', 'n07720875', 'n02125311', 'n02793495', 'n04532670']

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                if entry.strip("\n") in self.class_list:
                    self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        temp=[]
        for i in range(20):
            temp.append(0)
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG") and f.split("_")[0] in self.class_list:
                    for i in range(len(self.class_list)):
                        if f.split("_")[0] == self.class_list[i]:
                            
                            
                            if temp[i] < 500:
                                temp[i]+=1
                                num_images = num_images + 1
                            break
        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                if words[1] in self.class_list:
                    self.val_img_to_class[words[0]] = words[1]
                    set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]
        temp=[]
        for i in range(20):
            temp.append(0)
        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG"))and fname.split("_")[0] in self.class_list:
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        for i in range(len(self.class_list)):
                            if fname.split("_")[0] == self.class_list[i]:
                                temp[i]+=1
                                
                                if temp[i] <= 500:
                                    self.images.append(item)
        print('len',len(self.images))

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return int(self.len_dataset*self.ratio)

    def __getitem__(self, idx:int):
        img_path, tgt = self.images[idx]
        tgt+=1000
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        index = idx
        if self.list:
            if type(sample) == list:
                sample.append(index)
            else:
                sample = [sample, index]

        return sample, tgt


def prepare_transforms(dataset):

    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar10+100' or dataset == 'cifar10OOD' :
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100' or dataset == 'cifar100OOD':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise NotImplementedError

    normalize = transforms.Normalize(mean=mean, std=std)

    te_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    tr_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    simclr_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])

    return tr_transforms, te_transforms, simclr_transforms

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, te_transform):
        self.transform = transform
        self.te_transform = te_transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.te_transform(x)]

# -------------------------

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_test_data(args, ttt=False, num_sample=None, align=False):

    tr_transforms, te_transforms, simclr_transforms = prepare_transforms(args.dataset)

    if args.dataset == 'cifar10OOD':
        
        tesize = 10000
        if args.corruption in common_corruptions:
            
            print('Test on %s level %d' %(args.corruption, args.level))
            teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
            teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
            teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
            teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
            teset_10 = CIFAR10(root=args.dataroot,
                            train=False, download=True, transform=te_transforms)
            teset_10.data = teset_raw_10

            if args.strong_OOD == 'MNIST':
                te_rize = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.Grayscale(3), te_transforms ])
                noise = MNIST_openset(root=args.dataroot,
                            train=False, download=True, transform=te_rize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_10,noise])
            
            elif args.strong_OOD == 'noise':
                noise = noise_dataset(te_transforms, args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_10,noise])

            elif args.strong_OOD =='cifar100':
                teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/snow.npy')
                teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
                teset_100 = CIFAR100_openset(root=args.dataroot,
                                train=False, download=True, transform=te_transforms, ratio=args.strong_ratio)
                teset_100.data = teset_raw_100[:int(10000*args.strong_ratio)]
                teset = torch.utils.data.ConcatDataset([teset_10,teset_100])

            elif args.strong_OOD =='SVHN': 
                te_rize = transforms.Compose([te_transforms ])
                noise = SVHN_openset(root=args.dataroot,
                            split='test', download=True, transform=te_rize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_10,noise])
                
            elif args.strong_OOD =='Tiny':

                transform_test = transforms.Compose([transforms.Resize(32), te_transforms ])
                testset_tiny = TinyImageNet_OOD_nonoverlap(args.dataroot +'/tiny-imagenet-200', transform=transform_test, train=True)
                teset = torch.utils.data.ConcatDataset([teset_10,testset_tiny])
                print(len(teset_10),len(testset_tiny),len(teset))
                
            else:
                raise

    elif args.dataset == 'cifar100OOD':
        
        tesize = 10000

        if args.corruption in common_corruptions:
            print('Test on %s level %d' %(args.corruption, args.level))
            teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
            teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
            teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
            teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
            teset_100 = CIFAR100(root=args.dataroot,
                            train=False, download=True, transform=te_transforms)
            teset_100.data = teset_raw_100

            if args.strong_OOD == 'MNIST':
                te_rize = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.Grayscale(3), te_transforms ])
                noise = MNIST_openset(root=args.dataroot,
                            train=False, download=True, transform=te_rize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_100,noise])
            
            elif args.strong_OOD == 'noise':
                noise = noise_dataset(te_transforms, args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_100,noise])

            elif args.strong_OOD =='cifar10':
                teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/snow.npy')
                teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
                teset_10 = CIFAR10_openset(root=args.dataroot,
                                train=False, download=True, transform=te_transforms, ratio=args.strong_ratio)
                teset_10.data = teset_raw_10[:int(10000*args.strong_ratio)]
                teset = torch.utils.data.ConcatDataset([teset_100,teset_10])

            elif args.strong_OOD =='SVHN': 
                te_rize = transforms.Compose([te_transforms ])
                noise = SVHN_openset(root=args.dataroot,
                            split='test', download=True, transform=te_rize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_100,noise])
                
            elif args.strong_OOD =='Tiny':

                transform_test = transforms.Compose([transforms.Resize(32), te_transforms ])
                testset_tiny = TinyImageNet_OOD_nonoverlap(args.dataroot +'/tiny-imagenet-200', transform=transform_test, train=True)
                teset = torch.utils.data.ConcatDataset([teset_100,testset_tiny])

            else:
                raise

    if not hasattr(args, 'workers') or args.workers < 2:
        pin_memory = False
    else:
        pin_memory = True

    if ttt:
        shuffle = True
        drop_last = True
    else:
        shuffle = True
        drop_last = False

    try:
        teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                            shuffle=shuffle, num_workers=args.workers,
                                            worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=drop_last)
    except:
        teloader = None
    
    
    return teset, teloader

def prepare_train_data(args, num_sample=None):
    print('Preparing data...')
    
    tr_transforms, te_transforms, simclr_transforms = prepare_transforms(args.dataset)

    if args.dataset == 'cifar10' or args.dataset == 'cifar10+100' or args.dataset == 'cifar10OOD':

        if hasattr(args, 'ssl') and args.ssl == 'contrastive':
            trset = CIFAR10(root=args.dataroot,
                            train=False, download=True,
                            transform=TwoCropTransform(simclr_transforms, te_transforms))
            if hasattr(args, 'corruption') and args.corruption in common_corruptions:
                print('Contrastive on %s level %d' %(args.corruption, args.level))
                tesize = 10000
                trset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
                trset_raw = trset_raw[(args.level-1)*tesize: args.level*tesize]   
                trset.data = trset_raw
            else:
                print('Contrastive on ciar10 training set')
        else:
            trset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                        train=True, download=True, transform=tr_transforms)
            print('Cifar10 training set')

    elif args.dataset == 'cifar100' or args.dataset == 'cifar100OOD':
        if hasattr(args, 'ssl') and args.ssl == 'contrastive':
            trset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                         train=True, download=True,
                                         transform=TwoCropTransform(simclr_transforms, te_transforms))            
            if hasattr(args, 'corruption') and args.corruption in common_corruptions:
                print('Contrastive on %s level %d' %(args.corruption, args.level))
                tesize = 10000
                trset_raw = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
                trset_raw = trset_raw[(args.level-1)*tesize: args.level*tesize]   
                trset.data = trset_raw
            else:
                print('Contrastive on ciar10 training set')
        else:
            trset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                            train=True, download=True, transform=tr_transforms)
            print('Cifar100 training set')
    else:
        raise Exception('Dataset not found!')

    if not hasattr(args, 'workers') or args.workers < 2:
        pin_memory = False
    else:
        pin_memory = True

    if num_sample and num_sample < trset.data.shape[0]:
        trset.data = trset.data[:num_sample]
        print("Truncate the training set to {:d} samples".format(num_sample))

    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers,
                                            worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=False)
    return trset, trloader
