

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

# ======================================================
# PARAMETRI FISSI O CONFIGURABILI
# ======================================================
USE_CUTOUT = True            # Attiva/Disattiva Cutout
LABEL_SMOOTHING = 0.1        # Label smoothing
WEIGHT_DECAY = 3e-4          # Weight decay (fisso)
BATCH_SIZE = 128             # Batch size (fisso)
EPOCHS = 50                  # Numero di epoche (fisso)
LR = 0.1                     # Learning rate iniziale
DROPOUT_START_EPOCH = 0     # Epoca dalla quale abilitare il dropout
DROPOUT_P = 0.5              # Probabilità di dropout quando attivato
USE_DROPOUT = True           # Se vuoi sperimentare il dropout



def print_param_distribution(model):
    """
    Stampa la distribuzione dei parametri di un modello PyTorch,
    mostrando per ogni livello:
      - nome del parametro
      - numero di parametri (numel)
      - dimensioni (shape)
    e calcola il totale complessivo.
    """
    print("=== Param Distribution ===")
    total_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Salta i parametri che non sono aggiornati
        nump = param.numel()
        total_params += nump
        print(f"{name:60s} | shape={list(param.shape)} | params={nump}")
    
    print(f"Total Trainable Params: {total_params}")
    print("=========================\n")



# ======================================================
# A) CLASSE CUTOUT (OPZIONALE)
# ======================================================
class Cutout(object):
    """
    Applica un 'buco' (rettangolo di zeri) casuale nell'immagine.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img: Tensor [C, H, W]
        h = img.size(1)
        w = img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)

        for _ in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.

        mask = mask.unsqueeze(0).expand_as(img)  # [C, H, W]
        img = img * mask
        return img

# ======================================================
# B) EFFICIENT CHANNEL ATTENTION (ECA)
# ======================================================
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # Kernel size dinamico da formula ECA-Net
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

# ======================================================
# C) COORDINATE ATTENTION (COORDATT)
# ======================================================
class CoordAtt(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(mip)
        self.act   = nn.GELU()
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = torch.mean(x, dim=3, keepdim=True)   # [B, C, H, 1]
        x_w = torch.mean(x, dim=2, keepdim=True)   # [B, C, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)              # [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)           # [B, C, H+W, 1]

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h0, x_w0 = torch.split(y, [H, W], dim=2)
        x_w0 = x_w0.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h0)
        a_w = self.conv_w(x_w0)
        a_h = self.sigmoid(a_h)
        a_w = self.sigmoid(a_w)
        out = x * a_h.expand(-1, -1, -1, W) * a_w.expand(-1, -1, H, -1)
        return out

# ======================================================
# D) FUNZIONE PER IMPOSTARE IL DROPOUT
# ======================================================
def set_dropout_p(module: nn.Module, p: float):
    """
    Visita ricorsivamente tutti i sub-module
    e se trova un 'nn.Dropout', imposta 'p = p'.
    """
    for child in module.children():
        if isinstance(child, nn.Dropout):
            child.p = p
        else:
            set_dropout_p(child, p)

# ======================================================
# E) INVERTED RESIDUAL (MBConv)
# ======================================================
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4.0,
                 use_eca=False, use_ca=False, dropout_p=0.0):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []
        # 1) Espansione
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
            # Dropout facoltativo dopo l'espansione
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))

        # 2) Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())
        # Dropout facoltativo dopo la depthwise
        if dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))

        # 3) ECA + CA
        self.eca = ECABlock(hidden_dim) if use_eca else nn.Identity()
        self.ca  = CoordAtt(hidden_dim, hidden_dim) if use_ca else nn.Identity()

        # 4) Proiezione
        layers.append(nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        idx = 0

        # Se c'è la parte di espansione (8 layer totali nel seq)
        # altrimenti se expand=1 (6 layer totali)
        if len(self.conv) >= 8:
            out = self.conv[0](out)   # conv pw expand
            out = self.conv[1](out)   # BN
            out = self.conv[2](out)   # GELU
            idx = 3
            if isinstance(self.conv[3], nn.Dropout):
                out = self.conv[3](out)
                idx = 4

        # Depthwise
        out = self.conv[idx](out)       # conv dw
        out = self.conv[idx+1](out)     # BN
        out = self.conv[idx+2](out)     # GELU
        next_idx = idx+3
        # Se c'è dropout
        if next_idx < len(self.conv) and isinstance(self.conv[next_idx], nn.Dropout):
            out = self.conv[next_idx](out)
            next_idx += 1

        # ECA + CA
        out = self.eca(out)
        out = self.ca(out)

        # Proiezione
        out = self.conv[next_idx](out)
        out = self.conv[next_idx+1](out)

        if self.use_res_connect:
            out = x + out
        return out

# ======================================================
# F) MODELLO ESEMPIO PER CIFAR-100
#    (CON OPZIONE DROPOUT NEI BLOCCHI)
# ======================================================
class MyCifar100Net(nn.Module):
    def __init__(self, num_classes=100, dropout_p=0.0):
        super(MyCifar100Net, self).__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Eventuale dropout
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        )

        # Stage1
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 64,  stride=1, expand_ratio=2.0, use_eca=True,  use_ca=False, dropout_p=dropout_p),
            InvertedResidual(64, 64,  stride=1, expand_ratio=2.0, use_eca=True,  use_ca=True, dropout_p=dropout_p),
        )
        # Stage2
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 128, stride=2, expand_ratio=4.0, use_eca=True, use_ca=True,  dropout_p=dropout_p),
            InvertedResidual(128,128, stride=1, expand_ratio=4.0, use_eca=True, use_ca=True,  dropout_p=dropout_p),
        )
        # Stage3
        self.stage3 = nn.Sequential(
            InvertedResidual(128,256, stride=2, expand_ratio=5.0, use_eca=True,  use_ca=True,  dropout_p=dropout_p),
            InvertedResidual(256,256, stride=1, expand_ratio=5.0, use_eca=True,  use_ca=True,  dropout_p=dropout_p),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(384),
            nn.GELU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ======================================================
# G) PRINT MODEL INFO (PARAMS E FLOPs)
# ======================================================
try:
    from ptflops import get_model_complexity_info
    USE_PTFLOPS = True
except ImportError:
    USE_PTFLOPS = False

def print_model_info(model, input_res=(3,32,32)):
    # Parametri totali
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Totale parametri: {params_count}")

    # FLOPs se ptflops è disponibile
    if USE_PTFLOPS:
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(
                model, input_res, as_strings=True, print_per_layer_stat=False
            )
        print(f"FLOPs: {macs}")
    else:
        print("FLOPs: (ptflops non installato)")

# ======================================================
# H) FUNZIONI TRAIN E VALIDAZIONE
# ======================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct, total, running_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, 100.0 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

    return val_loss / total, 100.0 * correct / total

# ======================================================
# I) FUNZIONE DI TRAINING
#    (NON USIAMO MAIN, CHIAMIAMO run_training() DIRETTAMENTE)
# ======================================================
def run_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Trasformazioni base
    train_transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
    if USE_CUTOUT:
        train_transform.append(Cutout(n_holes=1, length=16))

    train_transform = transforms.Compose(train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Inizializziamo il modello
    # Di default impostiamo dropout_p = 0 all'inizio
    # Lo abiliteremo a runtime, dopo epoca >= 20, se USE_DROPOUT è True
    model = MyCifar100Net(num_classes=100, dropout_p=0.0 if not USE_DROPOUT else 0.0).to(device)

    print_param_distribution(model)

    # Stampa parametri e FLOPs
    print_model_info(model, input_res=(3,32,32))

    # Ottimizzatore e scheduler
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        # Se vogliamo attivare il dropout a partire dalla 20-esima epoca
        if USE_DROPOUT and epoch == DROPOUT_START_EPOCH:
            print(f"--> Attivo il dropout con p={DROPOUT_P} a partire dall'epoca {epoch}.")
            set_dropout_p(model, DROPOUT_P)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch}/{EPOCHS}]  "
              f"TrainLoss: {train_loss:.4f} | TrainAcc: {train_acc:.2f}% || "
              f"ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.2f}%")

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc

    print(f"Best Val Accuracy: {best_acc:.2f}%")
    scripted_model = torch.jit.script(model)
    scripted_model.save("prova.pt")

# ======================================================
# LANCIO DIRETTO DEL TRAINING (SENZA MAIN)
# ======================================================
if __name__ == "__main__":
    run_training()


import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

# ======================================================
# PARAMETRI FISSI O CONFIGURABILI
# ======================================================
USE_CUTOUT = True            # Attiva/Disattiva Cutout
LABEL_SMOOTHING = 0.1        # Label smoothing
WEIGHT_DECAY = 3e-4          # Weight decay (fisso)
BATCH_SIZE = 128             # Batch size (fisso)
EPOCHS = 50                  # Numero di epoche (fisso)
LR = 0.1                     # Learning rate iniziale
DROPOUT_START_EPOCH = 0     # Epoca dalla quale abilitare il dropout
DROPOUT_P = 0.5              # Probabilità di dropout quando attivato
USE_DROPOUT = True           # Se vuoi sperimentare il dropout



def print_param_distribution(model):
    """
    Stampa la distribuzione dei parametri di un modello PyTorch,
    mostrando per ogni livello:
      - nome del parametro
      - numero di parametri (numel)
      - dimensioni (shape)
    e calcola il totale complessivo.
    """
    print("=== Param Distribution ===")
    total_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Salta i parametri che non sono aggiornati
        nump = param.numel()
        total_params += nump
        print(f"{name:60s} | shape={list(param.shape)} | params={nump}")
    
    print(f"Total Trainable Params: {total_params}")
    print("=========================\n")



# ======================================================
# A) CLASSE CUTOUT (OPZIONALE)
# ======================================================
class Cutout(object):
    """
    Applica un 'buco' (rettangolo di zeri) casuale nell'immagine.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img: Tensor [C, H, W]
        h = img.size(1)
        w = img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)

        for _ in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.

        mask = mask.unsqueeze(0).expand_as(img)  # [C, H, W]
        img = img * mask
        return img

# ======================================================
# B) EFFICIENT CHANNEL ATTENTION (ECA)
# ======================================================
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # Kernel size dinamico da formula ECA-Net
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

# ======================================================
# C) COORDINATE ATTENTION (COORDATT)
# ======================================================
class CoordAtt(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1   = nn.BatchNorm2d(mip)
        self.act   = nn.GELU()
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = torch.mean(x, dim=3, keepdim=True)   # [B, C, H, 1]
        x_w = torch.mean(x, dim=2, keepdim=True)   # [B, C, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)              # [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)           # [B, C, H+W, 1]

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h0, x_w0 = torch.split(y, [H, W], dim=2)
        x_w0 = x_w0.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h0)
        a_w = self.conv_w(x_w0)
        a_h = self.sigmoid(a_h)
        a_w = self.sigmoid(a_w)
        out = x * a_h.expand(-1, -1, -1, W) * a_w.expand(-1, -1, H, -1)
        return out

# ======================================================
# D) FUNZIONE PER IMPOSTARE IL DROPOUT
# ======================================================
def set_dropout_p(module: nn.Module, p: float):
    """
    Visita ricorsivamente tutti i sub-module
    e se trova un 'nn.Dropout', imposta 'p = p'.
    """
    for child in module.children():
        if isinstance(child, nn.Dropout):
            child.p = p
        else:
            set_dropout_p(child, p)

# ======================================================
# E) INVERTED RESIDUAL (MBConv)
# ======================================================
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4.0,
                 use_eca=False, use_ca=False, dropout_p=0.0):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []
        # 1) Espansione
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
            # Dropout facoltativo dopo l'espansione
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))

        # 2) Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())
        # Dropout facoltativo dopo la depthwise
        if dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))

        # 3) ECA + CA
        self.eca = ECABlock(hidden_dim) if use_eca else nn.Identity()
        self.ca  = CoordAtt(hidden_dim, hidden_dim) if use_ca else nn.Identity()

        # 4) Proiezione
        layers.append(nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        idx = 0

        # Se c'è la parte di espansione (8 layer totali nel seq)
        # altrimenti se expand=1 (6 layer totali)
        if len(self.conv) >= 8:
            out = self.conv[0](out)   # conv pw expand
            out = self.conv[1](out)   # BN
            out = self.conv[2](out)   # GELU
            idx = 3
            if isinstance(self.conv[3], nn.Dropout):
                out = self.conv[3](out)
                idx = 4

        # Depthwise
        out = self.conv[idx](out)       # conv dw
        out = self.conv[idx+1](out)     # BN
        out = self.conv[idx+2](out)     # GELU
        next_idx = idx+3
        # Se c'è dropout
        if next_idx < len(self.conv) and isinstance(self.conv[next_idx], nn.Dropout):
            out = self.conv[next_idx](out)
            next_idx += 1

        # ECA + CA
        out = self.eca(out)
        out = self.ca(out)

        # Proiezione
        out = self.conv[next_idx](out)
        out = self.conv[next_idx+1](out)

        if self.use_res_connect:
            out = x + out
        return out

# ======================================================
# F) MODELLO ESEMPIO PER CIFAR-100
#    (CON OPZIONE DROPOUT NEI BLOCCHI)
# ======================================================
class MyCifar100Net(nn.Module):
    def __init__(self, num_classes=100, dropout_p=0.0):
        super(MyCifar100Net, self).__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Eventuale dropout
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        )

        # Stage1
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 64,  stride=1, expand_ratio=2.0, use_eca=True,  use_ca=False, dropout_p=dropout_p),
            InvertedResidual(64, 64,  stride=1, expand_ratio=2.0, use_eca=True,  use_ca=True, dropout_p=dropout_p),
        )
        # Stage2
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 128, stride=2, expand_ratio=4.0, use_eca=True, use_ca=True,  dropout_p=dropout_p),
            InvertedResidual(128,128, stride=1, expand_ratio=4.0, use_eca=True, use_ca=True,  dropout_p=dropout_p),
        )
        # Stage3
        self.stage3 = nn.Sequential(
            InvertedResidual(128,256, stride=2, expand_ratio=5.0, use_eca=True,  use_ca=True,  dropout_p=dropout_p),
            InvertedResidual(256,256, stride=1, expand_ratio=5.0, use_eca=True,  use_ca=True,  dropout_p=dropout_p),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(384),
            nn.GELU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ======================================================
# G) PRINT MODEL INFO (PARAMS E FLOPs)
# ======================================================
try:
    from ptflops import get_model_complexity_info
    USE_PTFLOPS = True
except ImportError:
    USE_PTFLOPS = False

def print_model_info(model, input_res=(3,32,32)):
    # Parametri totali
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Totale parametri: {params_count}")

    # FLOPs se ptflops è disponibile
    if USE_PTFLOPS:
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(
                model, input_res, as_strings=True, print_per_layer_stat=False
            )
        print(f"FLOPs: {macs}")
    else:
        print("FLOPs: (ptflops non installato)")

# ======================================================
# H) FUNZIONI TRAIN E VALIDAZIONE
# ======================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct, total, running_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, 100.0 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

    return val_loss / total, 100.0 * correct / total

# ======================================================
# I) FUNZIONE DI TRAINING
#    (NON USIAMO MAIN, CHIAMIAMO run_training() DIRETTAMENTE)
# ======================================================
def run_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Trasformazioni base
    train_transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
    if USE_CUTOUT:
        train_transform.append(Cutout(n_holes=1, length=16))

    train_transform = transforms.Compose(train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Inizializziamo il modello
    # Di default impostiamo dropout_p = 0 all'inizio
    # Lo abiliteremo a runtime, dopo epoca >= 20, se USE_DROPOUT è True
    model = MyCifar100Net(num_classes=100, dropout_p=0.0 if not USE_DROPOUT else 0.0).to(device)

    print_param_distribution(model)

    # Stampa parametri e FLOPs
    print_model_info(model, input_res=(3,32,32))

    # Ottimizzatore e scheduler
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        # Se vogliamo attivare il dropout a partire dalla 20-esima epoca
        if USE_DROPOUT and epoch == DROPOUT_START_EPOCH:
            print(f"--> Attivo il dropout con p={DROPOUT_P} a partire dall'epoca {epoch}.")
            set_dropout_p(model, DROPOUT_P)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch}/{EPOCHS}]  "
              f"TrainLoss: {train_loss:.4f} | TrainAcc: {train_acc:.2f}% || "
              f"ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.2f}%")

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc

    print(f"Best Val Accuracy: {best_acc:.2f}%")
    scripted_model = torch.jit.script(model)
    scripted_model.save("prova.pt")

# ======================================================
# LANCIO DIRETTO DEL TRAINING (SENZA MAIN)
# ======================================================
if __name__ == "__main__":
    run_training()
