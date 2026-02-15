import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

"""
MobileNetECA v2 - Fused-MBConv & Hard-Swish
Ottimizzazioni mirate per aumentare l'accuratezza (>92%) mantenendo <100k parametri.
1. Fused-MBConv nei primi stadi (EfficientNetV2 style) per feature extraction più ricca.
2. Hard-Swish activation (MobileNetV3 style) per migliore non-linearità.
3. Stem leggermente potenziato (24 canali).
"""

# ========== Parametri =======================
n_classi = 10
tasso_iniziale = 0.05
epoche = 50
dimensione_batch = 128
fattore_larghezza = 0.5
lr_scale = 1.54
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
# ============================================


# --- Attivazione Hard-Swish (MobileNetV3) ---
class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu(x + 3.) / 6.


# --- ECA Block (Standard) ---
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


# --- Fused-MBConv Block (EfficientNetV2 Style) ---
# Sostituisce la sequenza (1x1 -> 3x3 dw -> 1x1) con (3x3 piena -> 1x1)
# Estremamente efficiente nei primi stadi dove la risoluzione spaziale è alta.
class FusedMBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=False):
        super(FusedMBConv, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        # Fused Expansion + Depthwise -> diventa una singola Conv3x3 piena
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),  # Conv Piena (non depthwise!)
                nn.BatchNorm2d(hidden_dim),
                HSwish()
            ])
        else:
            # Fallback se non espande (raro)
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish()
            ])

        if use_eca:
            layers.append(ECABlock(hidden_dim))

        # Pointwise Projection (1x1) se servono meno canali in uscita, altrimenti Identity
        if hidden_dim != oup:
             layers.extend([
                nn.Conv2d(hidden_dim, oup, 1, bias=False),
                nn.BatchNorm2d(oup)
            ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# --- Standard Inverted Residual (MobileNetV2 style) ---
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=True):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish()  # HSwish qui
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            HSwish()  # HSwish qui
        ])

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
        else:
            return self.conv(x)


# --- Architettura Principale ---
class MobileNetECA_v2(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5):
        super(MobileNetECA_v2, self).__init__()
        
        # Struttura ibrida: Primi blocchi Fused, poi Inverted
        # t, c, n, s, block_type (0=Fused, 1=Inverted)
        block_settings = [
            [1, 24, 1, 1, 0],   # Fused-MBConv (Start forte)
            [4, 32, 2, 2, 0],   # Fused-MBConv con stride 2 (Downsample aggressivo ma ricco)
            [6, 48, 4, 2, 1],   # Inverted Standard
            [6, 64, 3, 1, 1],   # Inverted Standard
            [6, 80, 2, 2, 1],   # Inverted Standard
            [6, 96, 2, 1, 1],   # Inverted Standard
        ]
        
        # Stem Potenziato (24 canali fissi per catturare RGB)
        input_channel = 24 
        last_channel = max(int(160 * width_mult), 128) # Head più larga

        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            HSwish()
        )]

        for t, c, n, s, b_type in block_settings:
            output_channel = max(int(c * width_mult), 16)
            for i in range(n):
                stride = s if i == 0 else 1
                
                if b_type == 0:
                    self.features.append(
                        FusedMBConv(input_channel, output_channel, stride=stride, expand_ratio=t, use_eca=True)
                    )
                else:
                    self.features.append(
                        InvertedResidual(input_channel, output_channel, stride=stride, expand_ratio=t, use_eca=True)
                    )
                input_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            HSwish(),
            nn.AdaptiveAvgPool2d(1)
        ))

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
             nn.Dropout(0.2),  # Dropout finale per regolarizzare
             nn.Linear(last_channel, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

def formatta_numero(num): return f'{num / 1000:.1f}k'

if __name__ == "__main__":
    
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

    print("Caricamento dataset...")
    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=trasformazioni_train)
    dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=trasformazioni_test)
    caricatore_train = DataLoader(dataset_train, batch_size=dimensione_batch, shuffle=True, num_workers=2)
    caricatore_test = DataLoader(dataset_test, batch_size=dimensione_batch, shuffle=False, num_workers=2)

    modello = MobileNetECA_v2(num_classes=n_classi, width_mult=fattore_larghezza).to(dispositivo)
    
    params = sum(p.numel() for p in modello.parameters())
    print(f"\n{'='*70}")
    print(f"MODELLO V2 (Fused-MBConv + HSwish)")
    print(f"Parametri Totali: {formatta_numero(params)}")
    print(f"{'='*70}\n")

    ottimizzatore = optim.SGD(modello.parameters(), lr=tasso_iniziale, momentum=0.9, weight_decay=5e-4)
    schedulatore = CosineAnnealingLR(ottimizzatore, T_max=epoche)
    criterio = nn.CrossEntropyLoss()
    
    best_acc = 0.0

    for epoca in range(epoche):
        modello.train()
        corretti_train, totale_train = 0, 0
        
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
        
        if acc_valid > best_acc:
            best_acc = acc_valid
            
        print(f'Epoca {epoca+1:02d}/{epoche} - Train: {acc_train:.2f}% | Val: {acc_valid:.2f}% (Best: {best_acc:.2f}%)')
    
    print(f"\nRisultato Finale V2 Architecture:")
    print(f"Params: {formatta_numero(params)}")
    print(f"Validation Acc: {acc_valid:.2f}%")
