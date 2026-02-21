"""
Specific configuration for CIFAR-100 experiments.
Inherits logic from shared_config.py but overrides dataset and class count.
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import logging

from shared_config import (
    SEED, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, MOMENTUM,
    WIDTH_MULT, MIN_CHANNELS, BLOCK_SETTINGS, VAL_SIZE,
    get_data_dir, setup_logging, get_device, set_seed,
    initialize_weights, count_flops
)

# CIFAR-100 specific
NUM_CLASSES = 100
WIDTH_MULT = 1.0  # Passiamo a 1.0x per CIFAR-100
NORM_MEAN = (0.5071, 0.4867, 0.4408)
NORM_STD = (0.2675, 0.2565, 0.2761)

# Nuovi Block settings potenziati (t, c, n, s)
BLOCK_SETTINGS = [
    [1,  24, 2, 1],  # Stage 1
    [6,  32, 4, 2],  # Stage 2
    [6,  48, 6, 2],  # Stage 3 (Aumentato n=6)
    [6,  56, 4, 1],  # Stage 4 (Aumentato n=4)
    [6,  72, 2, 1],  # Stage 5 (Nuovo stage)
]

def get_transforms_advanced_c100():
    """Advanced transforms for CIFAR-100."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return transform_train, transform_test

def get_dataloaders_c100(transform_train, transform_test):
    """Dataloaders for CIFAR-100."""
    data_dir = get_data_dir()

    full_trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(full_trainset), generator=generator).tolist()

    val_indices = indices[:VAL_SIZE]
    train_indices = indices[VAL_SIZE:]

    trainset = Subset(full_trainset, train_indices)
    val_dataset_raw = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=transform_test
    )
    valset = Subset(val_dataset_raw, val_indices)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valloader = DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    logging.info(f"CIFAR-100 Dataset split: Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}")

    return trainloader, valloader, testloader
