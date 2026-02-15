import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

"""
MobileNetECA Ghost - Training su CIFAR-10 con approccio Ghost Module
Obiettivo: Ridurre i parametri senza perdere accuratezza.
"""

# ========== Parametri Semplificati ==========
n_classi = 10
tasso_iniziale = 0.05       # Come da richiesta (Best Config)
epoche = 50                 # Come da richiesta
dimensione_batch = 128      # Come da richiesta
fattore_larghezza = 0.5     # Come da richiesta
lr_scale = 1.6              # Ottimizzato per Ghost
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
# ============================================

# --- Ghost Module Implementation ---
# Dimezza i canali generati da Conv2d e genera il resto con depthwise cheap operations
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary convolution (genera metà delle map)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.GELU() if relu else nn.Sequential(),
        )

        # Cheap operation (genera l'altra metà)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.GELU() if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# --- ECA Block (Identico a prima) ---
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
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


# --- Ghost Bottleneck con MobileNetV2 structure ---
class GhostBottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=True):
        super(GhostBottleneck, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        # Ghost Expansion
        if expand_ratio != 1:
            layers.append(GhostModule(inp, hidden_dim, kernel_size=1, relu=True))
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()  # GELU come da tua specifica originale
        ])

        if use_eca:
            layers.append(ECABlock(hidden_dim))

        # Ghost Reduction (Linear)
        layers.append(GhostModule(hidden_dim, oup, kernel_size=1, relu=False))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# --- Architettura Principale ---
class MobileNetECAGhost(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5):
        super(MobileNetECAGhost, self).__init__()
        
        # Stessa struttura di blocchi della tua MobileNetECA
        block_settings = [
            # t, c, n, s
            [1, 20, 2, 1],   
            [6, 32, 4, 2],   
            [8, 42, 4, 2],   
            [8, 52, 2, 1],   
        ]
        
        input_channel = max(int(32 * width_mult), 12)
        last_channel = max(int(144 * width_mult), 12)

        # Stem Layer
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.GELU()
        )]

        # Building Blocks
        for t, c, n, s in block_settings:
            output_channel = max(int(c * width_mult), 12)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    GhostBottleneck(input_channel, output_channel, stride, expand_ratio=t, use_eca=True)
                )
                input_channel = output_channel

        # Last Layer
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        ))

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(last_channel, num_classes)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

def formatta_numero(num):
    return f'{num / 1000:.1f}k'

# =============================================================================
# MAIN SCRIPT
# =============================================================================
if __name__ == "__main__":
    
    # 1. Dataset (Data Augmentation Base)
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

    print("Caricamento dataset CIFAR-10...")
    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=trasformazioni_train)
    dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=trasformazioni_test)
    
    caricatore_train = DataLoader(dataset_train, batch_size=dimensione_batch, shuffle=True, num_workers=2)
    caricatore_test = DataLoader(dataset_test, batch_size=dimensione_batch, shuffle=False, num_workers=2)

    # 2. Modello
    modello = MobileNetECAGhost(num_classes=n_classi, width_mult=fattore_larghezza).to(dispositivo)
    
    params = sum(p.numel() for p in modello.parameters())
    print(f"\n{'='*60}")
    print(f"MODELLO GHOST (Compression Optimized)")
    print(f"Parametri Totali: {formatta_numero(params)} (vs Originale ~54k-77k)")
    print(f"{'='*60}\n")

    # 3. Training Setup (Stessi iperparametri richiesti)
    ottimizzatore = optim.SGD(modello.parameters(), lr=tasso_iniziale, momentum=0.9, weight_decay=5e-4) # 5e-4 come Best Config
    schedulatore = CosineAnnealingLR(ottimizzatore, T_max=epoche)
    criterio = nn.CrossEntropyLoss()

    # 4. Loop
    for epoca in range(epoche):
        modello.train()
        corretti_train = 0
        totale_train = 0
        
        for input, obiettivi in caricatore_train:
            input, obiettivi = input.to(dispositivo), obiettivi.to(dispositivo)
            ottimizzatore.zero_grad()
            uscite = modello(input)
            perdita = criterio(uscite, obiettivi)
            perdita.backward()
            nn.utils.clip_grad_norm_(modello.parameters(), max_norm=5) 
            ottimizzatore.step()
            
            _, predetti = uscite.max(1)
            corretti_train += predetti.eq(obiettivi).sum().item()
            totale_train += obiettivi.size(0)
            
        acc_train = 100. * corretti_train / totale_train
        
        # Validazione
        modello.eval()
        corretti_val = 0
        totale_val = 0
        with torch.no_grad():
            for input, obiettivi in caricatore_test:
                input, obiettivi = input.to(dispositivo), obiettivi.to(dispositivo)
                uscite = modello(input)
                _, predetti = uscite.max(1)
                corretti_val += predetti.eq(obiettivi).sum().item()
                totale_val += obiettivi.size(0)
        
        acc_valid = 100. * corretti_val / totale_val
        schedulatore.step()
        
        print(f'Epoca {epoca+1:02d}/{epoche} - Train: {acc_train:.2f}% | Val: {acc_valid:.2f}%')

    print(f"\nRisultato Finale Ghost Architecture:")
    print(f"Params: {formatta_numero(params)}")
    print(f"Validation Acc: {acc_valid:.2f}%")
