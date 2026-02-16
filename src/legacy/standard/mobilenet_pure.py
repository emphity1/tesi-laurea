import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_Mimir_Pure(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5):
        super(MobileNetV2_Mimir_Pure, self).__init__()
        # Configurazione Mimir (senza ECA)
        # t: expand_ratio, c: output_channels, n: num_blocks, s: stride
        self.block_settings = [
            [1, 20, 2, 1],
            [6, 32, 4, 2],
            [8, 42, 4, 2],
            [8, 52, 2, 1],
        ]
        
        input_channel = max(int(32 * width_mult), 8)
        last_channel = max(int(144 * width_mult), 8)
        
        # Primo layer
        self.features = [conv_bn(3, input_channel, 1)] 
        
        # Blocchi Inverted Residual
        for t, c, n, s in self.block_settings:
            output_channel = max(int(c * width_mult), 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                
        # Ultimo layer conv 1x1 prima del pooling
        # Nota: Nel codice originale Mimir c'è un Conv2d 1x1 + BN + GELU/ReLU
        # Qui usiamo la struttura standard MobileNetV2 che ha un layer di "feature mixing" finale
        self.features.append(conv_1x1_bn(input_channel, last_channel))
        
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

import time
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Path Locale
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Hyperparameters
    epochs = 200
    batch_size = 128
    lr = 0.05
    
    # Data
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
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = MobileNetV2_Mimir_Pure(num_classes=10, width_mult=0.5).to(device)
    print(f"Parametri totali: {sum(p.numel() for p in model.parameters())}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Start Training: {epochs} epochs, Device: {device}")
    
    start_time = time.time()
    best_acc = 0.0
    
    # History Dict
    history = {'train_acc': [], 'val_acc': [], 'loss': [], 'epochs': [], 'lr': [], 'time': []}
    
    for epoch in range(epochs):
        epoch_start = time.time()
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
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_pure.pth'))
            
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        remaining_time = (epochs - epoch - 1) * epoch_time
        
        # Update History
        history['epochs'].append(epoch + 1)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['loss'].append(avg_loss)
        history['lr'].append(current_lr)
        history['time'].append(time.time() - start_time)
        
        print(f"Ep {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (Best: {best_acc:.2f}%) | LR: {current_lr:.5f} | Time: {epoch_time:.1f}s (ETA: {remaining_time/60:.1f}m)")

    # Save History JSON
    with open(os.path.join(OUTPUT_DIR, 'MobileNetPure_50_history.json'), 'w') as f:
        json.dump(history, f)

    total_time = (time.time()-start_time)/60
    print(f"\n{'='*50}")
    print(f"Training Finito in {total_time:.1f} min")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Log salvati in: {OUTPUT_DIR}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()