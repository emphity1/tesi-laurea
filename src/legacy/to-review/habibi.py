import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import re
import onnx_tool
import torch.onnx

# ========== Parametri Modificabili ==========
n_classi = 10  # Numero di classi per CIFAR-10
tasso_iniziale = 0.025  # Learning rate iniziale
epoche = 50  # Numero di epoche
dimensione_batch = 128  # Dimensione del batch
fattore_larghezza = 0.42  # Fattore per regolare la capacità del modello
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'  # Uso di CUDA se disponibile
lr_scale = 1.44  # Valore di lr_scale utilizzato nel modello
# ============================================


#20K-5M-86 hhabibi


# Funzione per formattare i numeri
def formatta_numero(num):
    if abs(num) >= 1_000_000:
        return f'{num / 1_000_000:.1f}M'
    elif abs(num) >= 1_000:
        return f'{num / 1_000:.1f}k'
    else:
        return str(num)

# Funzione per arrotondare al numero significativo più vicino
def arrotonda_significativo(x, cifre=2):
    if x == 0:
        return 0
    else:
        return round(x, -int(math.floor(math.log10(abs(x))) - (cifre - 1)))

# Funzione per calcolare i FLOPs utilizzando ONNX
def calcola_flops_onnx(model):
    input_dummy = torch.randn(1, 3, 32, 32).to(dispositivo)
    onnx_path = "tmp.onnx"
    profile_path = "profile.txt"
    torch.onnx.export(model,
                      input_dummy,
                      onnx_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=None)
    onnx_tool.model_profile(onnx_path, save_profile=profile_path)
    with open(profile_path, 'r') as file:
        profilo = file.read()

    match = re.search(r'Total\s+_\s+([\d,]+)\s+100%', profilo)

    if match:
        total_macs = match.group(1)
        total_macs = int(total_macs.replace(',', ''))
        total_macs = arrotonda_significativo(total_macs)
        return total_macs
    else:
        return None





# Implementazione dell'ECABlock
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=6, lr_scale=1.6):
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

# Blocco Inverso aggiornato con ECABlock
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

# Definizione dell'architettura MobileNetECA
class MobileNetECA(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.2, lr_scale=1.44, grayscale=False, in_middle=False):
        super(MobileNetECA, self).__init__()

        # Impostazioni per i blocchi
        block_settings = [
            # t, c, n, s, use_eca
            [2, 24, 1, 1, True],
            [6, 32, 1, 2, True],
            [8, 42, 2, 2, True],
            [8, 56, 1, 1, True],
        ]
        input_channel = max(int(28 * width_mult), 8)
        last_channel = max(int(144 * width_mult), 8)

        # Primo strato
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.GELU()
        )]

        # Costruzione dei blocchi
        for idx, (t, c, n, s, use_eca) in enumerate(block_settings):
            output_channel = max(int(c * width_mult), 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, use_eca=use_eca, lr_scale=lr_scale)
                )
                input_channel = output_channel


        # Ultimo strato
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        ))

        self.features = nn.Sequential(*self.features)

        # Classificatore
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

        # Inizializzazione dei pesi
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)






# Dataset CIFAR-10
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

# Creazione dei DataLoader
dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=trasformazioni_train)
caricatore_train = DataLoader(dataset_train, batch_size=dimensione_batch, shuffle=True, num_workers=2, pin_memory=True)

dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=trasformazioni_test)
caricatore_test = DataLoader(dataset_test, batch_size=dimensione_batch, shuffle=False, num_workers=2, pin_memory=True)

# Creazione del modello e trasferimento su GPU
modello = MobileNetECA(num_classes=n_classi, width_mult=fattore_larghezza, lr_scale=lr_scale).to(dispositivo)

# Calcolo dei parametri
params = sum(p.numel() for p in modello.parameters())
params_formattati = formatta_numero(params)
print(f"Numero totale di parametri: {params_formattati}")

# Ottimizzatore e Scheduler
ottimizzatore = optim.SGD(modello.parameters(), lr=tasso_iniziale, momentum=0.9, weight_decay=3e-4)
schedulatore = CosineAnnealingLR(ottimizzatore, T_max=epoche)

criterio = nn.CrossEntropyLoss()

# Funzione di allenamento
def allenamento():
    modello.train()
    corretti = 0
    totale = 0

    for input, obiettivi in caricatore_train:
        input, obiettivi = input.to(dispositivo), obiettivi.to(dispositivo)
        ottimizzatore.zero_grad()
        uscite = modello(input)
        perdita = criterio(uscite, obiettivi)
        perdita.backward()
        nn.utils.clip_grad_norm_(modello.parameters(), max_norm=5)
        ottimizzatore.step()

        _, predetti = uscite.max(1)
        totale += obiettivi.size(0)
        corretti += predetti.eq(obiettivi).sum().item()

    accuratezza = 100. * corretti / totale
    return accuratezza

# Funzione di validazione
def validazione():
    modello.eval()
    corretti = 0
    totale = 0

    with torch.no_grad():
        for input, obiettivi in caricatore_test:
            input, obiettivi = input.to(dispositivo), obiettivi.to(dispositivo)
            uscite = modello(input)
            perdita = criterio(uscite, obiettivi)

            _, predetti = uscite.max(1)
            totale += obiettivi.size(0)
            corretti += predetti.eq(obiettivi).sum().item()

    accuratezza = 100. * corretti / totale
    return accuratezza

# Calcolo e stampa dei parametri e MACs
print("------ Parametri arrotondati ------")
params = sum(param.numel() for param in modello.parameters())
params = arrotonda_significativo(params)
macs = calcola_flops_onnx(modello)
params_formattati = formatta_numero(params)
macs_formattati = formatta_numero(macs)
print(f"Params: {params_formattati}  MACS: {macs_formattati}")

# Ciclo di allenamento
for epoca in range(epoche):
    acc_train = allenamento()
    acc_valid = validazione()
    schedulatore.step()
    print(f'Epoca {epoca+1} - Accuratezza Allenamento: {acc_train:.2f}% - Accuratezza Validazione: {acc_valid:.2f}%')

# Salvataggio del modello utilizzando TorchScript
modello_scriptato = torch.jit.script(modello)
modello_scriptato.save('/workspace/Dima/models/modello_habibi.pt')
print("Modello salvato come 'modello_alternativo.pt'")
