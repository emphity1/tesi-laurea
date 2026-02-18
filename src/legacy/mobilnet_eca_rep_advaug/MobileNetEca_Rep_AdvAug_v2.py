
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json
import logging
import os

# --- Path Setup ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_run_dir = os.path.join(_script_dir, "v2_results")
os.makedirs(_run_dir, exist_ok=True)

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=os.path.join(_run_dir, 'training.log'), filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# --- Architettura ---

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
            self.rbr_dense = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False), nn.BatchNorm2d(out_channels))
            self.rbr_1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False), nn.BatchNorm2d(out_channels))
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):
        if hasattr(self, 'reparam_conv'):
            return self.activation(self.reparam_conv(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.activation(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_conv'):
            return
        
        k3, b3 = self.get_equivalent_kernel_bias(self.rbr_dense)
        k1, b1 = self.get_equivalent_kernel_bias(self.rbr_1x1)
        kid, bid = self.get_equivalent_kernel_bias(self.rbr_identity)

        # Pad 1x1 to 3x3
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

    def get_equivalent_kernel_bias(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel, running_mean, running_var, gamma, beta, eps = branch[0].weight, branch[1].running_mean, branch[1].running_var, branch[1].weight, branch[1].bias, branch[1].eps
        else:
            # Identity branch: batchnorm. Generating identity kernel dynamically to avoid device mismatch.
            assert isinstance(branch, nn.BatchNorm2d)
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=branch.weight.dtype, device=branch.weight.device)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            
            kernel, running_mean, running_var, gamma, beta, eps = kernel_value, branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class ECABlock(nn.Module):
    # FIXED: gamma=2, b=1 (Standard ECA Paper settings)
    def __init__(self, channels, gamma=2, b=1):
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
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU()) # GELU activation
        
        # RepDepthwise
        layers.append(RepConv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim))
        
        if use_eca:
            layers.append(ECABlock(hidden_dim))
        
        # Pointwise Linear
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetECARep_v2(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5):
        super(MobileNetECARep_v2, self).__init__()
        
        # Settings: [expand_ratio, output_channel, num_blocks, stride]
        self.cfgs = [
            [1,  20, 2, 1], # t=1 light start
            [6,  32, 4, 2], # t=6 standard
            [8,  42, 4, 2], # t=8 rich exp
            [8,  52, 2, 1], # t=8 refinement
        ]

        input_channel = int(32 * width_mult)
        if input_channel < 12: input_channel = 12 # Minimum width safeguard
        
        last_channel = int(144 * width_mult)
        if last_channel < 12: last_channel = 12

        # Stride 1 iniziale per CIFAR-10 (Cruciale!)
        self.features = [RepConv(3, input_channel, stride=1)] 
        
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            if output_channel < 12: output_channel = 12
            
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(RepInvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1) # Global Pool
        ))
        
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_channel, num_classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
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

    def deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()

# --- Configurazione Training ---
BATCH_SIZE = 128
EPOCHS = 200
# FIXED: LR effectively used. Base 0.05 * Scale 1.5 ~ 0.075 to be aggressive
LR_BASE = 0.05 
LR_SCALE = 1.0 # Keep standard 0.05 to compare fairly with v1. Or bump to 1.5 if brave.
LR = LR_BASE * LR_SCALE
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset & Loader
# Use a shared data directory relative to script if possible, or local
_data_dir = os.path.join(os.path.dirname(os.path.dirname(_script_dir)), "data") 
if not os.path.exists(_data_dir):
    _data_dir = './data'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

trainset = torchvision.datasets.CIFAR10(root=_data_dir, train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root=_data_dir, train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Modello
model = MobileNetECARep_v2(width_mult=0.5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - 1))

# --- Training Loop ---
best_acc = 0.0
stats = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}

logging.info("Starting Training V2 (Optimized)...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in tepoch:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            tepoch.set_postfix(loss=train_loss/len(trainloader), acc=100.*correct/total)
    
    # Validation
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
    val_acc = 100. * test_correct / test_total
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step() # Correct placement
    
    # Save best
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(_run_dir, "best_model_v2.pth"))
        
    # Logging
    logging.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(trainloader):.4f}, Train Acc={100.*correct/total:.2f}%, Val Acc={val_acc:.2f}%, LR={current_lr:.6f}")
    
    stats["train_loss"].append(train_loss/len(trainloader))
    stats["train_acc"].append(100.*correct/total)
    stats["val_acc"].append(val_acc)
    stats["lr"].append(current_lr)

total_time = time.time() - start_time
print(f"Total Training Time: {total_time/60:.2f} min")
print(f"Best Validation Accuracy: {best_acc:.2f}%")

# Add params count
total_params = sum(p.numel() for p in model.parameters())
stats["params"] = total_params
print(f"Total Parameters: {total_params}")

# Save stats
with open(os.path.join(_run_dir, "stats_v2.json"), "w") as f:
    json.dump(stats, f)

# Deploy Export
model_deploy = copy.deepcopy(model)
model_deploy.load_state_dict(torch.load(os.path.join(_run_dir, "best_model_v2.pth")))
model_deploy.deploy()
model_deploy.eval()
torch.save(model_deploy.state_dict(), os.path.join(_run_dir, "best_model_v2_deploy.pth"))
print(f"Model deployed and saved in {_run_dir}")
