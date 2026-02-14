import torch
import torch.nn as nn
import math
import re
import io
import contextlib

# ==========================================
# 1. DEFINIZIONE MODELLO MIMIR (da mimir1.py)
# ==========================================
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1, lr_scale=1.6):
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
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=False, lr_scale=1.6):
        super(InvertedResidual, self).__init__()
        self.lr_scale = lr_scale
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
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
        out = out * self.lr_scale + out.detach() * (1 - self.lr_scale)
        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileNetECA(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5, lr_scale=1.48):
        super(MobileNetECA, self).__init__()
        block_settings = [
            [2, 18, 1, 1, True],
            [6, 24, 3, 2, True],
            [8, 32, 3, 2, True],
            [8, 56, 2, 1, True],
        ]
        input_channel = max(int(32 * width_mult), 8)
        last_channel = max(int(144 * width_mult), 8)

        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.GELU()
        )]

        for t, c, n, s, use_eca in block_settings:
            output_channel = max(int(c * width_mult), 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, use_eca=use_eca, lr_scale=lr_scale)
                )
                input_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU()
        ))
        self.features = nn.Sequential(*self.features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

# ==========================================
# 2. DEFINIZIONE MODELLO COMPETITOR (Simil MobileNet Scaler)
# ==========================================
# (Solo per avere un confronto dimensionale)
class BaselineMobileNet(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(BaselineMobileNet, self).__init__()
        # Semplificata per confronto
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
    
    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. CALCOLO METRICHE
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, input_size=(1, 3, 32, 32)):
    # Stima molto grezza o wrapper per thop/ptflops se installati
    # Qui usiamo solo il conteggio parametri come proxy principale per ora
    # poiché library esterne potrebbero non essere installate
    return "N/A (requires thop/ptflops)"

print("=== ANALISI MODELLI PER TESI ===")
print("Nota: Questa analisi gira sulla CPU del tuo laptop solo per calcolare la complessità.\n")

# 1. Mimir Model
mimir = MobileNetECA(width_mult=0.5) # Configurazione da mimir1.py
params_mimir = count_parameters(mimir)
print(f"Modello [Mimir/MobileNetECA]:")
print(f"  - Input shape: 32x32")
print(f"  - Width Multiplier: 0.5")
print(f"  - Parametri Totali: {params_mimir:,}")
print(f"  - Note: Include blocchi ECA e attivazione GELU.")

# 2. Baseline MobileNetV2 (Standard)
# Nota: MobileNetV2 standard è pensata per 224x224, su 32x32 è molto ridondante ma funge da baseline "pesante"
baseline = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
baseline.classifier[1] = nn.Linear(1280, 10)
params_baseline = count_parameters(baseline)
print(f"\nModello [Baseline MobileNetV2 Standard]:")
print(f"  - Parametri Totali: {params_baseline:,}")
print(f"  - Note: Architettura standard non ottimizzata per CIFAR.")

print(f"\n=== RISULTATO CONFRONTO ===")
reduction = (1 - params_mimir / params_baseline) * 100
print(f"Il tuo modello Mimir è circa il {reduction:.2f}% più leggero della baseline standard MobileNetV2.")
print("Questo è un ottimo punto di partenza per il capitolo 'Risultati/Efficienza'.")
