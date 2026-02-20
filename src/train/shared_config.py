"""
Configurazione condivisa per tutti i 4 modelli dell'ablation study.
Garantisce uniformità di seed, normalizzazione, split e iperparametri.
"""
import os
import math
import copy
import json
import time
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ============ Iperparametri unificati ============
SEED = 42
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.05
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
WIDTH_MULT = 0.5
NUM_CLASSES = 10
MIN_CHANNELS = 12   # min channel safeguard (come v3)

# Normalizzazione unificata (stessi valori per tutti e 4 i modelli)
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD  = (0.247, 0.243, 0.261)

# Block settings condivisi (t, c, n, s)
BLOCK_SETTINGS = [
    [1,  20, 2, 1],
    [6,  32, 4, 2],
    [8,  42, 4, 2],
    [8,  52, 2, 1],
]

# Validation split: 45k train / 5k val (dal training set originale)
VAL_SIZE = 5000

# ============ Funzioni di utilità ============

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data_dir():
    """Trova o crea la cartella dati condivisa."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def setup_logging(run_dir):
    """Configura logging su file + console."""
    os.makedirs(run_dir, exist_ok=True)

    # Reset handlers
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filename=os.path.join(run_dir, 'training.log'),
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


def get_transforms_standard():
    """Trasformazioni standard (per modelli A, B, C): RandomCrop + RandomFlip."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return transform_train, transform_test


def get_transforms_advanced():
    """Trasformazioni avanzate (per modello D): AutoAugment + RandomErasing."""
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


def get_dataloaders(transform_train, transform_test):
    """
    Crea train/val/test loaders con validation split corretta.
    Train: 45k, Validation: 5k (split dal training set), Test: 10k (originale).
    """
    data_dir = get_data_dir()

    full_trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # Split deterministico con lo stesso seed
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(full_trainset), generator=generator).tolist()

    val_indices = indices[:VAL_SIZE]
    train_indices = indices[VAL_SIZE:]

    trainset = Subset(full_trainset, train_indices)

    # Per la validazione usiamo le trasformazioni di test (no augmentation)
    val_dataset_raw = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform_test
    )
    valset = Subset(val_dataset_raw, val_indices)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valloader = DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    logging.info(f"Dataset split: Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}")

    return trainloader, valloader, testloader


def initialize_weights(model):
    """Inizializzazione pesi coerente (come v3)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            m.bias.data.zero_()


def count_flops(model, input_size=(1, 3, 32, 32)):
    """Calcola FLOPs (Multiply-Accumulate) tramite forward hooks su Conv2d e Linear."""
    total_flops = [0]
    hooks = []

    def conv_hook(module, inp, out):
        batch, in_c, h, w = inp[0].shape
        out_c = module.out_channels
        kh, kw = module.kernel_size
        groups = module.groups
        flops_per_instance = (in_c // groups) * kh * kw
        total_flops[0] += batch * out_c * out.shape[2] * out.shape[3] * flops_per_instance

    def linear_hook(module, inp, out):
        total_flops[0] += inp[0].shape[0] * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device_param = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_size, device=device_param))
    model.train()

    for h in hooks:
        h.remove()

    return total_flops[0]


def train_and_evaluate(model, run_dir, trainloader, valloader, testloader, device, has_deploy=False):
    """
    Loop di training unificato.
    - Salva best model su validation set (non test!)
    - Alla fine valuta su test set
    - Salva history JSON
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    params = sum(p.numel() for p in model.parameters())
    flops = count_flops(model)
    logging.info(f"Parametri totali: {params:,} ({params/1000:.1f}k)")
    logging.info(f"FLOPs totali: {flops:,} ({flops/1e6:.2f}M)")

    best_val_acc = 0.0
    stats = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_acc": [], "lr": [], "time": []
    }

    start_time = time.time()

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        avg_loss = train_loss / len(trainloader)

        # --- Validation (su val set, NON test set!) ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Save best (basato su VALIDATION, non test)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

        # Stats
        stats["epoch"].append(epoch + 1)
        stats["train_loss"].append(avg_loss)
        stats["train_acc"].append(train_acc)
        stats["val_acc"].append(val_acc)
        stats["lr"].append(current_lr)
        stats["time"].append(time.time() - start_time)

        logging.info(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | "
            f"Val: {val_acc:.2f}% (Best: {best_val_acc:.2f}%) | "
            f"LR: {current_lr:.6f}"
        )

    total_time = (time.time() - start_time) / 60
    logging.info(f"Training completato in {total_time:.1f} min")
    logging.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # --- Valutazione finale su TEST SET (una sola volta) ---
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth")))
    model.eval()
    test_correct = 0
    test_total = 0

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            all_targets.append(targets.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    # Salva predizioni per analisi (confusion matrix, ROC, errori)
    np.savez(
        os.path.join(run_dir, "test_predictions.npz"),
        targets=all_targets,
        predictions=all_preds,
        probabilities=all_probs
    )
    logging.info(f"Predizioni test salvate: targets={all_targets.shape}, probs={all_probs.shape}")

    test_acc = 100. * test_correct / test_total
    logging.info(f"*** Test Accuracy (finale, one-shot): {test_acc:.2f}% ***")
    stats["test_acc_final"] = test_acc
    stats["best_val_acc"] = best_val_acc
    stats["total_params"] = params
    stats["total_flops"] = flops
    stats["total_time_min"] = total_time

    # Deploy mode (per modelli con RepConv)
    if has_deploy:
        model_deploy = copy.deepcopy(model)
        model_deploy.deploy()
        model_deploy.eval()

        # Verifica che deploy non cambi accuratezza
        deploy_correct = 0
        deploy_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_deploy(inputs)
                _, predicted = outputs.max(1)
                deploy_total += targets.size(0)
                deploy_correct += predicted.eq(targets).sum().item()
        deploy_acc = 100. * deploy_correct / deploy_total
        logging.info(f"Deploy Accuracy (verifica): {deploy_acc:.2f}%")
        stats["deploy_acc"] = deploy_acc
        stats["deploy_params"] = sum(p.numel() for p in model_deploy.parameters())

        torch.save(model_deploy.state_dict(), os.path.join(run_dir, "best_model_deploy.pth"))

    # Salva stats
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logging.info(f"Risultati salvati in {run_dir}")

    return stats
