"""
Experiment: Knowledge Distillation + EMA on CIFAR-100.
Renamed copy of train_F_kd.py adapted for CIFAR-100.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))

import argparse
import math
import copy
import json
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import CIFAR-100 specific config
from shared_config_cifar100 import (
    SEED, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, MOMENTUM,
    WIDTH_MULT, NUM_CLASSES, MIN_CHANNELS, BLOCK_SETTINGS,
    set_seed, get_device, setup_logging,
    get_transforms_advanced_c100, get_dataloaders_c100,
    initialize_weights, count_flops
)

# =============================================
# ARCHITETTURA POTENZIATA (CIFAR-100 VERSION)
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
        if hasattr(self, 'reparam_conv'): return
        k3, b3 = self._get_equivalent(self.rbr_dense)
        k1, b1 = self._get_equivalent(self.rbr_1x1)
        kid, bid = self._get_equivalent(self.rbr_identity)
        k1 = F.pad(k1, [1, 1, 1, 1])
        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True)
        self.reparam_conv.weight.data = k3 + k1 + kid
        self.reparam_conv.bias.data = b3 + b1 + bid
        for para in self.parameters(): para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'): self.__delattr__('rbr_identity')
        self.deploy = True

    def _get_equivalent(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean, running_var = branch[1].running_mean, branch[1].running_var
            gamma, beta, eps = branch[1].weight, branch[1].bias, branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=branch.weight.dtype, device=branch.weight.device)
                for i in range(self.in_channels): kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor.to(branch.weight.device) if self.id_tensor.device != branch.weight.device else self.id_tensor
            running_mean, running_var, gamma, beta, eps = branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=3, b=12):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(y))

class RepInvertedResidualC100(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)
        layers = []
        if expand_ratio != 1:
            layers.extend([nn.Conv2d(inp, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.GELU()])
        layers.append(RepConv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim))
        layers.append(ECABlock(hidden_dim))
        layers.append(SpatialAttention()) # Nuova Attention Spaziale
        layers.extend([nn.Conv2d(hidden_dim, oup, 1, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetECARepC100(nn.Module):
    def __init__(self, num_classes=100, width_mult=1.0):
        super().__init__()
        input_channel = max(int(32 * width_mult), MIN_CHANNELS)
        last_channel = max(int(144 * width_mult), MIN_CHANNELS)
        self.features = [RepConv(3, input_channel, stride=1)]
        for t, c, n, s in BLOCK_SETTINGS:
            output_channel = max(int(c * width_mult), MIN_CHANNELS)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(RepInvertedResidualC100(input_channel, output_channel, stride, t))
                input_channel = output_channel
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        ))
        self.features = nn.Sequential(*self.features)
        # Head migliorata per CIFAR-100
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        initialize_weights(self)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
    def deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'): m.switch_to_deploy()

# Import student architecture
from train_E_experiments import ModelEMA

# Import teacher architecture (Ensure you have a CIFAR-100 teacher or use a wrapper)
# For now, we reuse the RepVGG_CIFAR logic if it supports num_classes
from train_repvgg import RepVGG_CIFAR

# =============================================
# HYPERPARAMETERS FOR KNOWLEDGE DISTILLATION
# =============================================
KD_T = 4.0      # Temperature
KD_ALPHA = 0.5  # Weight for distillation loss
USE_EMA = True
EMA_DECAY = 0.999

# PATH TO TEACHER WEIGHTS (Important: needs to be a CIFAR-100 teacher)
TEACHER_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines', 'results_repvgg_a0_c100', 'best_model_deploy.pth')

def kd_loss(student_logits, teacher_logits, targets, temperature, alpha):
    """Calculates Knowledge Distillation Loss"""
    ce_loss = F.cross_entropy(student_logits, targets)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    kd_l = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    return (1. - alpha) * ce_loss + alpha * kd_l

def train_and_evaluate_kd(student, teacher, run_dir, trainloader, valloader, testloader, device):
    """Training loop with Knowledge Distillation and EMA."""
    config = {
        "dataset": "CIFAR-100",
        "kd_temperature": KD_T,
        "kd_alpha": KD_ALPHA,
        "ema": USE_EMA,
        "ema_decay": EMA_DECAY,
        "base_lr": LR,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "momentum": MOMENTUM,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    }

    with open(os.path.join(run_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"KD Configuration: Alpha={KD_ALPHA}, Temp={KD_T}")

    optimizer = optim.SGD(student.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if USE_EMA:
        ema = ModelEMA(student, decay=EMA_DECAY)
        logging.info(f"✓ EMA active: decay={EMA_DECAY}")
    else:
        ema = None

    params = sum(p.numel() for p in student.parameters())
    flops = count_flops(student)
    logging.info(f"Student Parameters: {params:,} ({params/1000:.1f}k)")
    logging.info(f"Student FLOPs: {flops:,} ({flops/1e6:.2f}M)")

    best_val_acc = 0.0
    stats = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_acc": [], "lr": [], "time": []
    }

    start_time = time.time()

    for epoch in range(EPOCHS):
        student.train()
        if teacher: teacher.eval()
        
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            student_logits = student(inputs)
            
            if teacher:
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                loss = kd_loss(student_logits, teacher_logits, targets, temperature=KD_T, alpha=KD_ALPHA)
            else:
                loss = F.cross_entropy(student_logits, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5)
            optimizer.step()

            if ema is not None:
                ema.update(student)

            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        avg_loss = train_loss / len(trainloader)

        # --- Validation ---
        if ema is not None:
            ema.apply_shadow(student)

        student.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), os.path.join(run_dir, "best_model.pth"))

        if ema is not None:
            ema.restore(student)

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
    logging.info(f"Training completed in {total_time:.1f} min")

    # --- Final Test ---
    student.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), weights_only=True))
    student.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_acc = 100. * test_correct / test_total
    logging.info(f"*** Test Accuracy (Final): {test_acc:.2f}% ***")
    
    stats["test_acc_final"] = test_acc
    stats["best_val_acc"] = best_val_acc
    stats["total_params"] = params
    stats["total_flops"] = flops

    # --- Deploy ---
    student_deploy = copy.deepcopy(student)
    student_deploy.deploy()
    student_deploy.eval()
    
    deploy_correct = 0
    deploy_total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student_deploy(inputs)
            _, predicted = outputs.max(1)
            deploy_total += targets.size(0)
            deploy_correct += predicted.eq(targets).sum().item()
            
    deploy_acc = 100. * deploy_correct / deploy_total
    logging.info(f"Deploy Accuracy (check): {deploy_acc:.2f}%")
    
    stats["deploy_acc"] = deploy_acc
    stats["deploy_params"] = sum(p.numel() for p in student_deploy.parameters())
    
    torch.save(student_deploy.state_dict(), os.path.join(run_dir, "best_model_deploy.pth"))
    
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    set_seed()
    device = get_device()

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_cifar100_kd_ema")
    setup_logging(run_dir)

    logging.info("=" * 60)
    logging.info("CIFAR-100 EXPERIMENT: Knowledge Distillation + EMA")
    logging.info("=" * 60)

    # 1. (Teacher rimosso perché addestrato su CIFAR-10)
    teacher = None
    
    # 2. Inizializza lo Studente (Versione Potenziata)
    logging.info(f"Inizializzazione Modello: MobileNetECA-Rep (Versione C100, Classes={NUM_CLASSES})")
    student = MobileNetECARepC100(num_classes=NUM_CLASSES, width_mult=WIDTH_MULT).to(device)

    # 3. Dataloaders (CIFAR-100)
    transform_train, transform_test = get_transforms_advanced_c100()
    trainloader, valloader, testloader = get_dataloaders_c100(transform_train, transform_test)

    # 4. Vai col training (Standard CE)
    train_and_evaluate_kd(student, teacher, run_dir, trainloader, valloader, testloader, device)
