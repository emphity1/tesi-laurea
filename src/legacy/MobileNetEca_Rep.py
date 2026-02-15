import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
MobileNetECA Rep (Final Thesis Run)
- Reparameterized Structure
- Deterministic Seed
- Full Metrics Logging
- Checkpointing
- Deploy Mode Saving
"""

# ========== Parametri =======================
SEED = 42
n_classi = 10
tasso_iniziale = 0.05
epoche = 200
dimensione_batch = 128
fattore_larghezza = 0.5
lr_scale = 1.54
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = "reports/final_run_reparam"
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

# --- Reparameterized Convolution Block ---
class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
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

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_conv'): return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, 'rbr_identity'): del self.rbr_identity

    def get_equivalent_kernel_bias(self):
        k3x3, b3x3 = self._fuse_bn_tensor(self.rbr_dense)
        k1x1, b1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        k1x1 = F.pad(k1x1, [1, 1, 1, 1])
        k_id, b_id = 0, 0
        if self.rbr_identity is not None:
            k_id, b_id = self._fuse_bn_tensor(self.rbr_identity)
        return k3x3 + k1x1 + k_id, b3x3 + b1x1 + b_id

    def _fuse_bn_tensor(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel, running_mean, running_var, gamma, beta, eps = branch[0].weight, branch[1].running_mean, branch[1].running_var, branch[1].weight, branch[1].bias, branch[1].eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels): kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel, running_mean, running_var, gamma, beta, eps = self.id_tensor, branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

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
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=True):
        super(RepInvertedResidual, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)
        layers = []
        if expand_ratio != 1:
            layers.extend([nn.Conv2d(inp, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.GELU()])
        layers.append(RepConv(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))
        if use_eca: layers.append(ECABlock(hidden_dim))
        layers.extend([nn.Conv2d(hidden_dim, oup, 1, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetECARep(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5):
        super(MobileNetECARep, self).__init__()
        block_settings = [[1, 20, 2, 1], [6, 32, 4, 2], [8, 42, 4, 2], [8, 52, 2, 1]]
        input_channel = max(int(32 * width_mult), 12)
        last_channel = max(int(144 * width_mult), 12)
        self.features = [RepConv(3, input_channel, stride=1)]
        for t, c, n, s in block_settings:
            output_channel = max(int(c * width_mult), 12)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(RepInvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features.append(nn.Sequential(nn.Conv2d(input_channel, last_channel, 1, bias=False), nn.BatchNorm2d(last_channel), nn.GELU(), nn.AdaptiveAvgPool2d(1)))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(last_channel, num_classes)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
    def deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'): m.switch_to_deploy()

if __name__ == "__main__":
    
    set_seed(SEED)
    print(f"Seed set to {SEED}")

    def formatta_numero(num): return f'{num / 1000:.1f}k'

    # Dataset
    trasformazioni_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    trasformazioni_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=trasformazioni_train)
    dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=trasformazioni_test)
    caricatore_train = DataLoader(dataset_train, batch_size=dimensione_batch, shuffle=True, num_workers=2)
    caricatore_test = DataLoader(dataset_test, batch_size=dimensione_batch, shuffle=False, num_workers=2)

    # Modello
    modello = MobileNetECARep(num_classes=n_classi, width_mult=fattore_larghezza).to(dispositivo)
    
    params_train = sum(p.numel() for p in modello.parameters())
    print(f"\n{'='*70}")
    print(f"MODELLO REPARAMETERIZED FINAL RUN")
    print(f"Parametri Training: {formatta_numero(params_train)}")
    print(f"{'='*70}\n")

    ottimizzatore = optim.SGD(modello.parameters(), lr=tasso_iniziale, momentum=0.9, weight_decay=5e-4)
    schedulatore = CosineAnnealingLR(ottimizzatore, T_max=epoche)
    criterio = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {'epochs': [], 'val_acc': [], 'train_acc': [], 'loss': [], 'lr': [], 'time': []}

    start_time = time.time()

    for epoca in range(epoche):
        modello.train()
        corretti_train, totale_train = 0, 0
        running_loss = 0.0
        
        for input, obiettivi in caricatore_train:
            input, obiettivi = input.to(dispositivo), obiettivi.to(dispositivo)
            ottimizzatore.zero_grad()
            uscite = modello(input)
            perdita = criterio(uscite, obiettivi)
            perdita.backward()
            nn.utils.clip_grad_norm_(modello.parameters(), max_norm=5)
            ottimizzatore.step()
            
            running_loss += perdita.item()
            _, predetti = uscite.max(1)
            corretti_train += predetti.eq(obiettivi).sum().item()
            totale_train += obiettivi.size(0)
            
        acc_train = 100. * corretti_train / totale_train
        avg_loss = running_loss / len(caricatore_train)
        
        # Validation
        modello.eval()
        corretti_val, totale_val = 0, 0
        with torch.no_grad():
            for input, obiettivi in caricatore_test:
                input, obiettivi = input.to(dispositivo), obiettivi.to(dispositivo)
                uscite = modello(input)
                _, predetti = uscite.max(1)
                corretti_val += predetti.eq(obiettivi).sum().item()
                totale_val += obiettivi.size(0)
        
        acc_valid = 100. * corretti_val / totale_val
        schedulatore.step()
        
        # Logging
        current_lr = schedulatore.get_last_lr()[0]
        history['epochs'].append(epoca+1)
        history['val_acc'].append(acc_valid)
        history['train_acc'].append(acc_train)
        history['loss'].append(avg_loss)
        history['lr'].append(current_lr)
        history['time'].append(time.time() - start_time) # Cumulative time
        
        # Checkpointing
        if acc_valid > best_acc:
            best_acc = acc_valid
            torch.save(modello.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_training.pth'))
            # Salva anche le metriche correnti
            with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
                json.dump(history, f)
        
        print(f'Epoca {epoca+1:03d}/{epoche} - Loss: {avg_loss:.4f} | Train: {acc_train:.2f}% | Val: {acc_valid:.2f}% (Best: {best_acc:.2f}%) | LR: {current_lr:.6f}')

    total_time = (time.time() - start_time) / 60
    print(f"\nTraining Completato in {total_time:.1f} min.")
    
    # Reload Best Model per Deploy
    print("Caricamento Best Model per Deploy...")
    modello.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model_training.pth')))
    
    print("Deploying...")
    modello.deploy()
    params_deploy = sum(p.numel() for p in modello.parameters())
    print(f"Parametri Finali (Deploy): {formatta_numero(params_deploy)}")
    
    # Final Validation & Save
    modello.eval()
    corretti_val, totale_val = 0, 0
    with torch.no_grad():
        for input, obiettivi in caricatore_test:
            input, obiettivi = input.to(dispositivo), obiettivi.to(dispositivo)
            uscite = modello(input)
            _, predetti = uscite.max(1)
            corretti_val += predetti.eq(obiettivi).sum().item()
            totale_val += obiettivi.size(0)
    final_acc = 100. * corretti_val / totale_val
    print(f"Accuratezza Finale Deployed: {final_acc:.2f}%")
    
    # Save Deployed Model (Quello che userai per Evaluation)
    save_path = os.path.join(OUTPUT_DIR, 'mobilenet_eca_reparam_deployed.pth')
    torch.save(modello.state_dict(), save_path) # Salva state_dict standard
    
    # Save JIT version (Opzionale, più portabile)
    inputs = torch.randn(1, 3, 32, 32).to(dispositivo)
    traced_model = torch.jit.trace(modello, inputs)
    traced_model.save(os.path.join(OUTPUT_DIR, 'mobilenet_eca_reparam_deployed_jit.pt'))
    
    print(f"Modelli salvati in {OUTPUT_DIR}")
