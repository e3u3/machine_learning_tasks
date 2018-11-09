import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

_NUM_CLASSES = {
    'cifar10': 10,
    'flowers': 17,
}


def cifar10(batch_size, data_root='./cifar10', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("CIFAR-10 training data size: {}".format(len(train_loader.dataset.train_data)))
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("CIFAR-10 testing data size: {}".format(len(test_loader.dataset.test_data)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def flowers(batch_size, data_root='./flowers', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(data_root)
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=os.path.join(data_root, 'train'),
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset.samples)))
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root=os.path.join(data_root, 'val'),
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset.samples)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

