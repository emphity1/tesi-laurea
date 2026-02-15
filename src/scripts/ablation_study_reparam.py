
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import numpy as np
import random
import os
import json
import time

"""
MobileNetECA Rep - ABLATION STUDY
Script modificato per eseguire 3 varianti in sequenza:
1. Baseline (ReLU, No ECA)
2. Variant A (GELU, No ECA)
3. Variant B (ReLU + ECA)
"""

# ========== Parametri Base ===================
SEED = 42
n_classi = 10
tasso_iniziale = 0.05
epoche = 50  # 50 epoche bastano per vedere il trend relativo
dimensione_batch = 128
fattore_larghezza = 0.5
lr_scale = 1.54 # Usato solo se c'è ECA
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = "reports/ablation_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Reparameterized Convolution Block (Modular) ---
class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, deploy=False, use_gelu=True):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = nn.GELU() if use_gelu else nn.ReLU(inplace=True)

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
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(in_channels)
            else:
                self.rbr_identity = None

    def forward(self, inputs):
        if hasattr(self, 'reparam_conv'):
            return self.activation(self.reparam_conv(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.activation(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    # ... (Il resto delle funzioni RepConv per deploy è identico, omesso per brevità ma necessario se si volesse fare deploy) ...

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=3, b=12):
        super(ECABlock, self).__init__()
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
        return x * y.transpose(-1, -2).unsqueeze(-1).expand_as(x)

class RepInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=True, use_gelu=True):
        super(RepInvertedResidual, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)
        layers = []
        activation = nn.GELU() if use_gelu else nn.ReLU(inplace=True)
        
        if expand_ratio != 1:
            layers.extend([nn.Conv2d(inp, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), activation])
        
        # RepConv ora accetta parametro use_gelu
        layers.append(RepConv(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, use_gelu=use_gelu))
        
        if use_eca: layers.append(ECABlock(hidden_dim))
        
        layers.extend([nn.Conv2d(hidden_dim, oup, 1, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetECARep_Modular(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5, use_eca=True, use_gelu=True):
        super(MobileNetECARep_Modular, self).__init__()
        block_settings = [[1, 20, 2, 1], [6, 32, 4, 2], [8, 42, 4, 2], [8, 52, 2, 1]]
        input_channel = max(int(32 * width_mult), 12)
        last_channel = max(int(144 * width_mult), 12)
        activation = nn.GELU() if use_gelu else nn.ReLU(inplace=True)

        self.features = [RepConv(3, input_channel, stride=1, use_gelu=use_gelu)]
        for t, c, n, s in block_settings:
            output_channel = max(int(c * width_mult), 12)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(RepInvertedResidual(input_channel, output_channel, stride, expand_ratio=t, use_eca=use_eca, use_gelu=use_gelu))
                input_channel = output_channel
        
        self.features.append(nn.Sequential(nn.Conv2d(input_channel, last_channel, 1, bias=False), nn.BatchNorm2d(last_channel), activation, nn.AdaptiveAvgPool2d(1)))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(last_channel, num_classes)
        
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

# --- TRAINING FUNCTION ---
def train_variant(name, use_eca, use_gelu, train_loader, test_loader):
    print(f"\n{'='*70}")
    print(f"START TRAINING: {name}")
    print(f"Config: ECA={use_eca}, GELU={use_gelu}")
    print(f"{'='*70}")
    
    set_seed(SEED) # Reset seed for fair comparison
    
    model = MobileNetECARep_Modular(num_classes=n_classi, width_mult=fattore_larghezza, use_eca=use_eca, use_gelu=use_gelu).to(dispositivo)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params}")
    
    optimizer = optim.SGD(model.parameters(), lr=tasso_iniziale, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoche)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    history = {'epochs': [], 'val_acc': [], 'train_acc': [], 'loss': []}
    start_time = time.time()

    for epoch in range(epoche):
        model.train()
        correct, total = 0, 0
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(dispositivo), targets.to(dispositivo)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(dispositivo), targets.to(dispositivo)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        
        val_acc = 100. * val_correct / val_total
        scheduler.step()
        
        if val_acc > best_acc: best_acc = val_acc
        
        history['epochs'].append(epoch+1)
        history['val_acc'].append(val_acc)
        history['train_acc'].append(train_acc)
        history['loss'].append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1}/{epoche} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (Best: {best_acc:.2f}%)")

    total_time = (time.time() - start_time) / 60
    print(f"Finished {name} in {total_time:.1f} min. Best Acc: {best_acc:.2f}%")
    
    # Save results
    res_file = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}_results.json")
    with open(res_file, 'w') as f:
        json.dump({'config': {'use_eca': use_eca, 'use_gelu': use_gelu}, 'best_acc': best_acc, 'params': params, 'history': history}, f)
        
    return best_acc, params

if __name__ == "__main__":
    
    # Dataset
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
    trainloader = DataLoader(trainset, batch_size=dimensione_batch, shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=dimensione_batch, shuffle=False, num_workers=2)

    results = []

    # 1. Baseline: ReLU, No ECA
    acc, p = train_variant("1_Baseline_ReLU_NoECA", use_eca=False, use_gelu=False, train_loader=trainloader, test_loader=testloader)
    results.append(f"Baseline (ReLU, No ECA): {acc:.2f}% ({p} params)")

    # 2. Variant A: GELU, No ECA
    acc, p = train_variant("2_GELU_NoECA", use_eca=False, use_gelu=True, train_loader=trainloader, test_loader=testloader)
    results.append(f"Variant A (GELU, No ECA): {acc:.2f}% ({p} params)")

    # 3. Variant B: ReLU + ECA
    acc, p = train_variant("3_ReLU_ECA", use_eca=True, use_gelu=False, train_loader=trainloader, test_loader=testloader)
    results.append(f"Variant B (ReLU + ECA): {acc:.2f}% ({p} params)")

    print("\n\n=== FINAL ABLATION RESULTS ===")
    for r in results: print(r)
    
    # Save Summary
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), 'w') as f:
        for r in results: f.write(r + "\n")
