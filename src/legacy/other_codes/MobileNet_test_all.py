import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import random
import numpy as np
import os
import argparse
import time
import json
from datetime import datetime

# ==========================================
# CONFIGURAZIONE DEFAULT (sovrascritta da argparse)
# ==========================================
DEFAULT_SEED = 50
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.05
DEFAULT_WIDTH_MULT = 0.5
DEFAULT_WD = 3e-4
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ========== COMPONENTI MODULARI ==========

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=3, b=12, lr_scale=1.6):
        super(ECABlock, self).__init__()
        self.lr_scale = lr_scale
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = y * self.lr_scale + y.detach() * (1 - self.lr_scale)
        return x * y.expand_as(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=False, use_gelu=True, lr_scale=1.6):
        super(InvertedResidual, self).__init__()
        self.lr_scale = lr_scale
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)
        
        activation_layer = nn.GELU if use_gelu else nn.ReLU

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation_layer()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation_layer()
        ])

        if use_eca:
            layers.append(ECABlock(hidden_dim, lr_scale=self.lr_scale))

        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if hasattr(self, 'lr_scale'): # Apply gradient scaling if ECA is used inside (or apply generally)
             # Note: original code only applied this scaling on ECA block output inside ECA block usually, 
             # but here we keep structure consistent. Let's apply scaling only if ECA is present or if we want it globally.
             # Based on original legacy code, scaling was applied on block output too.
             out = out * self.lr_scale + out.detach() * (1 - self.lr_scale)
             
        if self.use_res_connect:
            return x + out
        else:
            return out

class ModularMobileNet(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5, use_eca=True, use_gelu=True, lr_scale=1.54):
        super(ModularMobileNet, self).__init__()
        
        activation_layer = nn.GELU if use_gelu else nn.ReLU
        
        block_settings = [
            # t, c, n, s
            [1, 20, 2, 1],
            [6, 32, 4, 2],
            [8, 42, 4, 2],
            [8, 52, 2, 1],
        ]
        
        input_channel = max(int(32 * width_mult), 8)
        last_channel = max(int(144 * width_mult), 8)

        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            activation_layer()
        )]

        for t, c, n, s in block_settings:
            output_channel = max(int(c * width_mult), 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, 
                                     use_eca=use_eca, use_gelu=use_gelu, lr_scale=lr_scale)
                )
                input_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            activation_layer(),
            nn.AdaptiveAvgPool2d(1)
        ))

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(last_channel, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

# ========== MAIN EXECUTION ==========

def main():
    parser = argparse.ArgumentParser(description='MobileNet Test All Script')
    parser.add_argument('--name', type=str, required=True, help='Nome identificativo per il report')
    parser.add_argument('--no-eca', action='store_true', help='Disabilita ECA (default: ECA abilitato)')
    parser.add_argument('--use-relu', action='store_true', help='Usa ReLU invece di GELU (default: GELU)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help=f'Numero di epoche (default: {DEFAULT_EPOCHS})')
    
    args = parser.parse_args()

    # Configurazione derivata
    USE_ECA = not args.no_eca
    USE_GELU = not args.use_relu
    RUN_NAME = args.name
    
    # Setup Paths
    REPORT_DIR = os.path.join("/workspace/tesi-laurea/reports/test_all")
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_base = f"{timestamp}_{RUN_NAME}"
    
    print(f"\n{'='*50}")
    print(f"STARTING RUN: {RUN_NAME}")
    print(f"ECA: {USE_ECA} | GELU: {USE_GELU} | Epochs: {args.epochs}")
    print(f"{'='*50}\n")
    
    set_seed(DEFAULT_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data Loading
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Inizializza Modello
    model = ModularMobileNet(
        num_classes=10, 
        width_mult=DEFAULT_WIDTH_MULT, 
        use_eca=USE_ECA, 
        use_gelu=USE_GELU
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=DEFAULT_LR, momentum=0.9, weight_decay=DEFAULT_WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training Loop
    start_time = time.time()
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'loss': [], 'epochs': []}

    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(trainloader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        scheduler.step()
        
        history['epochs'].append(epoch + 1)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['loss'].append(avg_loss)
        
        print(f"Ep {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (Best: {best_acc:.2f}%)")

    total_time = time.time() - start_time
    
    # SALVATAGGIO REPORT
    report_path_txt = os.path.join(REPORT_DIR, f"{report_base}.txt")
    report_path_json = os.path.join(REPORT_DIR, f"{report_base}.json")

    # TXT Report
    with open(report_path_txt, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"TEST RUN REPORT: {RUN_NAME}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Configuration:\n")
        f.write(f"  - ECA Enabled: {USE_ECA}\n")
        f.write(f"  - Activation: {'GELU' if USE_GELU else 'ReLU'}\n")
        f.write(f"  - Epochs: {args.epochs}\n")
        f.write(f"  - Seed: {DEFAULT_SEED}\n")
        f.write(f"  - Parameters: {params}\n")
        f.write(f"\nResults:\n")
        f.write(f"  - Best Validation Acc: {best_acc:.2f}%\n")
        f.write(f"  - Final Loss: {avg_loss:.4f}\n")
        f.write(f"  - Total Time: {total_time/60:.1f} minutes\n")
    
    # JSON Report (per grafici futuri)
    with open(report_path_json, 'w') as f:
        json.dump({
            'name': RUN_NAME,
            'config': {
                'use_eca': USE_ECA,
                'use_gelu': USE_GELU,
                'params': params
            },
            'results': {
                'best_acc': best_acc,
                'history': history,
                'total_time': total_time
            }
        }, f, indent=4)
        
    print(f"\nRun Completed! Reports saved to:")
    print(f"  - {report_path_txt}")
    print(f"  - {report_path_json}")

if __name__ == "__main__":
    main()
