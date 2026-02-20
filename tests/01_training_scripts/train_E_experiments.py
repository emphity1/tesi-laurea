"""
Esperimenti aggiuntivi su MobileNetECA-Rep-AdvAug.
4 tecniche toggle-abili:
  - Label Smoothing (CrossEntropyLoss label_smoothing=0.1)
  - Mixup (alpha=0.2)
  - LR Warmup (5 epoche lineari)
  - EMA (Exponential Moving Average dei pesi, decay=0.999)

Ogni combinazione salva risultati in cartella separata.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
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

from shared_config import (
    SEED, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, MOMENTUM,
    WIDTH_MULT, NUM_CLASSES, MIN_CHANNELS, BLOCK_SETTINGS,
    set_seed, get_device, setup_logging,
    get_transforms_advanced, get_dataloaders,
    initialize_weights, count_flops
)

# =============================================
# TOGGLE FLAGS (modifica qui)
# =============================================
USE_LABEL_SMOOTHING = False  # Esperimento 1
USE_MIXUP           = False  # Esperimento 2
USE_LR_WARMUP       = False  # Esperimento 3
USE_EMA             = True   # Esperimento 4

LABEL_SMOOTHING = 0.1
MIXUP_ALPHA     = 0.2
WARMUP_EPOCHS   = 10
EMA_DECAY       = 0.999

# =============================================
# Architettura (identica a D)
# =============================================

class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = nn.GELU()

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True)
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):
        if hasattr(self, 'reparam_conv'):
            return self.activation(self.reparam_conv(inputs))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        return self.activation(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_conv'):
            return
        k3, b3 = self._get_equivalent(self.rbr_dense)
        k1, b1 = self._get_equivalent(self.rbr_1x1)
        kid, bid = self._get_equivalent(self.rbr_identity)
        k1 = F.pad(k1, [1, 1, 1, 1])

        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True)
        self.reparam_conv.weight.data = k3 + k1 + kid
        self.reparam_conv.bias.data = b3 + b1 + bid

        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        self.deploy = True

    def _get_equivalent(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean, running_var = branch[1].running_mean, branch[1].running_var
            gamma, beta, eps = branch[1].weight, branch[1].bias, branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=branch.weight.dtype, device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            if self.id_tensor.device != branch.weight.device:
                self.id_tensor = self.id_tensor.to(branch.weight.device)
            kernel = self.id_tensor
            running_mean, running_var = branch.running_mean, branch.running_var
            gamma, beta, eps = branch.weight, branch.bias, branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=3, b=12):
        super(ECABlock, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class RepInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=True):
        super(RepInvertedResidual, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ])

        layers.append(RepConv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim))

        if use_eca:
            layers.append(ECABlock(hidden_dim))

        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetECARep(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, width_mult=WIDTH_MULT):
        super(MobileNetECARep, self).__init__()

        input_channel = max(int(32 * width_mult), MIN_CHANNELS)
        last_channel = max(int(144 * width_mult), MIN_CHANNELS)

        self.features = [RepConv(3, input_channel, stride=1)]

        for t, c, n, s in BLOCK_SETTINGS:
            output_channel = max(int(c * width_mult), MIN_CHANNELS)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(RepInvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        ))

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_channel, num_classes))

        initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()


# =============================================
# Mixup utilities
# =============================================

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Mixup: combina coppie di campioni con un coefficiente lambda ~ Beta(alpha, alpha)."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss per Mixup: combinazione pesata delle loss su entrambi i target."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================
# LR Warmup utilities
# =============================================

def get_lr_with_warmup(epoch, warmup_epochs=WARMUP_EPOCHS, base_lr=LR):
    """Linear warmup: LR cresce linearmente da 0 a base_lr nei primi warmup_epochs."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


# =============================================
# EMA (Exponential Moving Average)
# =============================================

class ModelEMA:
    """
    Exponential Moving Average dei pesi del modello.
    Mantiene una copia shadow dei pesi: ema_weight = decay * ema_weight + (1 - decay) * model_weight
    Migliora la generalizzazione senza parametri extra a inference.
    """
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        """Aggiorna i pesi EMA dopo ogni step di ottimizzazione."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        """Applica i pesi EMA al modello (per valutazione)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        """Ripristina i pesi originali (per continuare il training)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


# =============================================
# Training loop personalizzato
# =============================================

def train_and_evaluate_experimental(model, run_dir, trainloader, valloader, testloader, device):
    """
    Training loop con supporto per Label Smoothing, Mixup, LR Warmup, EMA.
    Salva tutto: history, model, predictions.
    """

    # --- Configurazione esperimento ---
    config = {
        "label_smoothing": USE_LABEL_SMOOTHING,
        "label_smoothing_value": LABEL_SMOOTHING if USE_LABEL_SMOOTHING else 0,
        "mixup": USE_MIXUP,
        "mixup_alpha": MIXUP_ALPHA if USE_MIXUP else 0,
        "lr_warmup": USE_LR_WARMUP,
        "warmup_epochs": WARMUP_EPOCHS if USE_LR_WARMUP else 0,
        "ema": USE_EMA,
        "ema_decay": EMA_DECAY if USE_EMA else 0,
        "base_lr": LR,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "momentum": MOMENTUM,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    }

    # Salva configurazione
    with open(os.path.join(run_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"Configurazione esperimento: {config}")

    # --- Loss ---
    if USE_LABEL_SMOOTHING:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        logging.info(f"✓ Label Smoothing attivo: {LABEL_SMOOTHING}")
    else:
        criterion = nn.CrossEntropyLoss()
        logging.info("✗ Label Smoothing disattivo")

    if USE_MIXUP:
        logging.info(f"✓ Mixup attivo: alpha={MIXUP_ALPHA}")
    else:
        logging.info("✗ Mixup disattivo")

    if USE_LR_WARMUP:
        logging.info(f"✓ LR Warmup attivo: {WARMUP_EPOCHS} epoche")
    else:
        logging.info("✗ LR Warmup disattivo")

    if USE_EMA:
        ema = ModelEMA(model, decay=EMA_DECAY)
        logging.info(f"✓ EMA attivo: decay={EMA_DECAY}")
    else:
        ema = None
        logging.info("✗ EMA disattivo")

    # --- Optimizer ---
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # --- Scheduler ---
    if USE_LR_WARMUP:
        # Warmup lineare + Cosine Annealing dopo warmup
        def lr_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            else:
                # Cosine annealing da warmup_epochs a EPOCHS
                progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
                return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logging.info("Scheduler: Linear Warmup + Cosine Annealing")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        logging.info("Scheduler: Cosine Annealing (standard)")

    # --- Info modello ---
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

            if USE_MIXUP:
                # Mixup: mescola i campioni
                mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
                outputs = model(mixed_inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                # Per accuracy tracking: usa i target originali
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (lam * predicted.eq(targets_a).sum().float()
                           + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            # Aggiorna EMA dopo ogni step
            if ema is not None:
                ema.update(model)

            train_loss += loss.item()

        train_acc = 100. * correct / total
        avg_loss = train_loss / len(trainloader)

        # --- Validation (su val set, NON test set!) ---
        # Se EMA attivo, usa i pesi EMA per la validazione
        if ema is not None:
            ema.apply_shadow(model)

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

        # Ripristina pesi originali per continuare il training
        if ema is not None:
            ema.restore(model)

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

    # --- Valutazione finale su TEST SET ---
    # Carica il best model (che ha già i pesi EMA se EMA era attivo)
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
    stats["experiment_config"] = config

    # Deploy mode
    model_deploy = copy.deepcopy(model)
    model_deploy.deploy()
    model_deploy.eval()

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

    # --- Riepilogo finale ---
    logging.info("=" * 60)
    logging.info("RIEPILOGO ESPERIMENTO")
    logging.info("=" * 60)
    logging.info(f"Label Smoothing: {'ON' if USE_LABEL_SMOOTHING else 'OFF'}")
    logging.info(f"Mixup:           {'ON' if USE_MIXUP else 'OFF'}")
    logging.info(f"LR Warmup:       {'ON' if USE_LR_WARMUP else 'OFF'}")
    logging.info(f"EMA:             {'ON' if USE_EMA else 'OFF'}")
    logging.info(f"Best Val Acc:    {best_val_acc:.2f}%")
    logging.info(f"Test Acc:        {test_acc:.2f}%")
    logging.info(f"Deploy Acc:      {deploy_acc:.2f}%")
    logging.info(f"Params:          {params:,} (deploy: {stats['deploy_params']:,})")
    logging.info(f"FLOPs:           {flops:,}")
    logging.info(f"Tempo:           {total_time:.1f} min")
    logging.info("=" * 60)

    return stats


# =============================================
# Main
# =============================================

if __name__ == "__main__":
    set_seed()
    device = get_device()

    # Nome cartella basato sulle feature attive
    features = []
    if USE_LABEL_SMOOTHING:
        features.append("ls")
    if USE_MIXUP:
        features.append("mixup")
    if USE_LR_WARMUP:
        features.append(f"warmup{WARMUP_EPOCHS}")
    if USE_EMA:
        features.append("ema")

    if not features:
        features.append("noop")

    experiment_name = f"results_E_{'_'.join(features)}"
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), experiment_name)
    setup_logging(run_dir)

    logging.info("=" * 60)
    logging.info(f"ESPERIMENTO E: MobileNetECA-Rep-AdvAug + {', '.join(features).upper()}")
    logging.info(f"Risultati in: {run_dir}")
    logging.info("=" * 60)

    # Trasformazioni avanzate (come D)
    transform_train, transform_test = get_transforms_advanced()
    trainloader, valloader, testloader = get_dataloaders(transform_train, transform_test)

    model = MobileNetECARep().to(device)
    train_and_evaluate_experimental(model, run_dir, trainloader, valloader, testloader, device)
