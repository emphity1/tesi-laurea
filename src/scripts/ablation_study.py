
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import os
import time

# --- CONFIGURAZIONE ---
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.05
WIDTH_MULT = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FILE = "ablation_results.txt"

# --- DEFINIZIONE ARCHITETTURA MODULARE ---
# Classe Base MobileNetV2 che possiamo configurare per le 3 varianti
class MobileNetV2_Modular(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5, use_eca=False, use_gelu=False):
        super(MobileNetV2_Modular, self).__init__()
        self.use_eca = use_eca
        self.use_gelu = use_gelu
        self.activation = nn.GELU() if use_gelu else nn.ReLU(inplace=True)
        
        # Struttura Standard MobileNetV2
        # t: expand_ratio, c: out_channels, n: repeat, s: stride
        block_settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 1], # Stride 1 iniziale per CIFAR-10 (vs 2 ImageNet)
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Adaptation for CIFAR-10 Small Scale (simile alla nostra architettura custom)
        # Usiamo i settings della tua tesi per coerenza
        block_settings_custom = [
            [1, 20, 2, 1], 
            [6, 32, 4, 2], 
            [8, 42, 4, 2], 
            [8, 52, 2, 1]
        ]
        
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        
        # Stem
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            self.activation
        )]
        
        # Blocks
        for t, c, n, s in block_settings_custom:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, use_eca=use_eca, activation=self.activation))
                input_channel = output_channel
        
        # Head
        last_channel = int(144 * width_mult) # Adattato alla tua tesi
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            self.activation
        ))

        self.features = nn.Sequential(*self.features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca, activation):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(activation)
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation
        ])
        
        if use_eca:
            layers.append(ECABlock(hidden_dim)) # ECA dopo la depthwise

        layers.append(nn.Conv2d(hidden_dim, oup, 1, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=3, b=12):
        super(ECABlock, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# --- TRAINING HELPER ---
def train_model(model, train_loader, test_loader, epochs, name):
    print(f"\n{'='*20} TRAINING: {name} {'='*20}")
    model = model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        acc = 100. * val_correct / val_total
        if acc > best_acc: best_acc = acc
        scheduler.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Val Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
            
    total_time = (time.time() - start_time) / 60
    params = sum(p.numel() for p in model.parameters())
    return best_acc, params, total_time


# --- MAIN ABLATION STUDY ---
if __name__ == "__main__":
    # Data Augmentation (Standard)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    results = []
    
    # 1. Baseline: ReLU, No ECA
    model_baseline = MobileNetV2_Modular(use_eca=False, use_gelu=False, width_mult=WIDTH_MULT)
    acc, params, t_time = train_model(model_baseline, trainloader, testloader, EPOCHS, "Baseline (ReLU, No ECA)")
    results.append(f"Baseline (ReLU): Acc={acc:.2f}%, Params={params}, Time={t_time:.1f}m")
    
    # 2. Variant A: GELU, No ECA
    model_gelu = MobileNetV2_Modular(use_eca=False, use_gelu=True, width_mult=WIDTH_MULT)
    acc, params, t_time = train_model(model_gelu, trainloader, testloader, EPOCHS, "Variant A (GELU, No ECA)")
    results.append(f"Variant A (GELU): Acc={acc:.2f}%, Params={params}, Time={t_time:.1f}m")
    
    # 3. Variant B: ReLU + ECA
    model_eca = MobileNetV2_Modular(use_eca=True, use_gelu=False, width_mult=WIDTH_MULT)
    acc, params, t_time = train_model(model_eca, trainloader, testloader, EPOCHS, "Variant B (ReLU + ECA)")
    results.append(f"Variant B (ReLU+ECA): Acc={acc:.2f}%, Params={params}, Time={t_time:.1f}m")
    
    # Save Results
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(results))
        
    print("\n--- FINAL RESULTS ---")
    for r in results: print(r)
