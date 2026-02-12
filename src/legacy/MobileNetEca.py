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

"""
MobileNetECA - Training su CIFAR-10

Questo script implementa una versione custom di MobileNet con meccanismo di attenzione ECA
(Efficient Channel Attention) per la classificazione su CIFAR-10.

NOTA IMPORTANTE - Perché GELU invece di ReLU:
- GELU (Gaussian Error Linear Unit) è preferita a ReLU per i seguenti motivi:
  1. Gradienti più smooth: GELU è differenziabile ovunque, evitando il "dying ReLU problem"
  2. Migliori prestazioni in modelli compatti: studi recenti dimostrano che GELU funziona meglio
     in architetture con pochi parametri come questa (54k params)
  3. Usata in architetture moderne: BERT, GPT, Vision Transformers usano tutte GELU
  4. Apprendimento più stabile: la transizione smooth aiuta l'ottimizzazione
"""


# ========== Parametri Modificabili ==========
n_classi = 10  # Numero di classi per CIFAR-10
tasso_iniziale = 0.025  # Learning rate iniziale
epoche = 50  # Numero di epoche
dimensione_batch = 128  # Dimensione del batch
fattore_larghezza = 0.42  # Fattore per regolare la capacità del modello
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'  # Uso di CUDA se disponibile
lr_scale = 1.54  # Valore di lr_scale utilizzato nel modello
# ============================================


#Params: 30K  MACS: 7M - 88% mimir7


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




# ========== ECA Block (Efficient Channel Attention) ==========
# ECA è un meccanismo di attenzione molto leggero che migliora le prestazioni del modello
# con un overhead computazionale minimo (pochi parametri).
# 
# Come funziona:
# 1. Calcola l'importanza di ogni canale tramite Global Average Pooling
# 2. Applica una convoluzione 1D per catturare le dipendenze tra canali vicini
# 3. Genera pesi di attenzione con Sigmoid
# 4. Moltiplica l'input per questi pesi (ricalibrazione dei canali)
#
# Vantaggio rispetto a SE (Squeeze-and-Excitation):
# - ECA usa solo una Conv1D invece di due FC layers → meno parametri
# - La dimensione del kernel è adattiva in base al numero di canali
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=3, b=12, lr_scale=1.6):
        """
        Args:
            channels: numero di canali del feature map in input
            gamma: parametro per calcolare kernel_size adattivo (default=3)
            b: bias per la formula del kernel_size (default=12)
            lr_scale: fattore di scaling per il gradiente (vedi forward)
        """
        super(ECABlock, self).__init__()
        self.lr_scale = lr_scale
        
        # Formula adattiva per determinare kernel_size basata sul numero di canali
        # Più canali → kernel più grande per catturare dipendenze a lungo raggio
        # Formula: k = |log2(C) + b| / gamma, dove C = numero di canali
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1  # Forza kernel_size dispari per padding simmetrico
        
        # Global Average Pooling: riduce spatial dimensions a 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Conv1D per modellare le interazioni tra canali
        # kernel_size adattivo permette di catturare dipendenze locali tra canali
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        
        # Sigmoid per generare pesi di attenzione nell'intervallo [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, channels, height, width]
        
        # Step 1: Global Average Pooling → [batch, channels, 1, 1]
        y = self.avg_pool(x)
        
        # Step 2: Reshape per Conv1D: [batch, channels, 1, 1] → [batch, 1, channels]
        y = y.squeeze(-1).transpose(-1, -2)
        
        # Step 3: Convoluzione 1D per catturare dipendenze tra canali
        y = self.conv(y)
        
        # Step 4: Sigmoid per ottenere pesi di attenzione [0, 1]
        y = self.sigmoid(y)
        
        # Step 5: Reshape back: [batch, 1, channels] → [batch, channels, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        
        # Step 6: Gradient scaling trick con lr_scale
        # Questa tecnica permette di controllare quanto i gradienti fluiscono attraverso l'attenzione:
        # - y * lr_scale: parte con gradienti
        # - y.detach() * (1 - lr_scale): parte senza gradienti (freezata)
        # Questo stabilizza il training e previene che l'attenzione domini troppo l'apprendimento
        y = y * self.lr_scale + y.detach() * (1 - self.lr_scale)
        
        # Step 7: Moltiplica input per i pesi di attenzione (channel-wise)
        # Canali importanti vengono amplificati, quelli meno importanti vengono attenuati
        return x * y.expand_as(x)



# ========== Inverted Residual Block ==========
# Questo è il blocco fondamentale di MobileNetV2, noto come "Inverted Bottleneck".
# 
# Architettura standard vs Inverted:
# - Bottleneck standard (ResNet): largo → stretto → largo (es. 256→64→256)
# - Inverted bottleneck: stretto → largo → stretto (es. 24→144→24)
#
# Struttura del blocco (3 fasi):
# 1. EXPANSION: Conv 1x1 per espandere i canali (se expand_ratio > 1)
# 2. DEPTHWISE: Conv 3x3 depthwise (una conv per canale, molto efficiente)
# 3. PROJECTION: Conv 1x1 per ridurre i canali al numero desiderato
#
# Residual connection: se input e output hanno stessa dimensione, si somma l'input (skip connection)
# Questo aiuta il flusso dei gradienti e permette al modello di apprendere funzioni residuali.
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=False, lr_scale=1.6):
        """
        Args:
            inp: numero di canali in input
            oup: numero di canali in output
            stride: stride della depthwise conv (1=mantiene dimensione, 2=dimezza)
            expand_ratio: fattore di espansione (es. 6 significa 6x canali nella fase intermedia)
            use_eca: se True, aggiunge ECA attention dopo la depthwise conv
            lr_scale: fattore di scaling per i gradienti (vedi ECABlock)
        """
        super(InvertedResidual, self).__init__()
        self.lr_scale = lr_scale
        
        # Calcola il numero di canali nella fase espansa (hidden dimension)
        hidden_dim = int(inp * expand_ratio)
        
        # Residual connection è possibile solo se:
        # 1. stride=1 (nessun downsampling, dimensioni spaziali uguali)
        # 2. inp=oup (stesso numero di canali)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion (1x1 conv)
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # GELU invece di ReLU: fornisce gradienti più smooth e migliori prestazioni
                # in modelli compatti. GELU è una funzione non-lineare più sofisticata che
                # permette al modello di catturare pattern più complessi rispetto a ReLU.
                nn.GELU()
            ])

        # Depthwise convolution (3x3 conv per gruppo)
        # groups=hidden_dim significa che ogni canale ha il suo proprio filtro (depthwise)
        # Questo riduce drasticamente i parametri rispetto a una conv standard
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()  # GELU per coerenza e migliori prestazioni
        ])

        if use_eca:
            # Opzionale: aggiunge attenzione ECA per ricalibrazione dei canali
            layers.append(ECABlock(hidden_dim, lr_scale=self.lr_scale))

        # Pointwise projection (1x1 conv)
        # Riduce i canali dalla dimensione espansa (hidden_dim) a quella finale (oup)
        # Nota: nessuna attivazione dopo questa conv (linear bottleneck)
        # Questo preserva l'informazione e aiuta il gradient flow
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # Applica la sequenza di convoluzioni
        out = self.conv(x)
        
        # Gradient scaling trick (stesso concetto di ECABlock)
        # Controlla quanto i gradienti fluiscono attraverso il blocco
        out = out * self.lr_scale + out.detach() * (1 - self.lr_scale)
        
        if self.use_res_connect:
            # Se possibile, aggiunge skip connection: out = F(x) + x
            # Questo è il cuore dell'architettura residual: il modello impara
            # la differenza (residuo) invece della funzione completa
            return x + out
        else:
            # Altrimenti ritorna solo l'output della trasformazione
            return out


# Definizione dell'architettura MobileNetECA
class MobileNetECA(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.2, lr_scale=1.44, grayscale=False, in_middle=False):
        super(MobileNetECA, self).__init__()

        # Impostazioni per i blocchi - Ogni blocco è configurato con:
        # t (expansion_ratio): fattore di espansione dei canali (es. 6 = espande 6x)
        # c (output_channels): numero di canali di output dopo il blocco
        # n (num_blocks): numero di volte che il blocco viene ripetuto
        # s (stride): stride della prima convoluzione (1=no downsampling, 2=dimezza risoluzione)
        # use_eca: se True, applica il meccanismo di attenzione ECA al blocco
        block_settings = [
            # t, c, n, s, use_eca
            [1, 20, 2, 1, True],   # Blocco 1: no expansion, 20 canali out, ripetuto 2x, stride=1
            [6, 32, 4, 2, True],   # Blocco 2: 6x expansion, 32 canali out, ripetuto 4x, stride=2 (dimezza risoluzione)
            [8, 42, 4, 2, True],   # Blocco 3: 8x expansion, 42 canali out, ripetuto 4x, stride=2 (dimezza risoluzione)
            [8, 52, 2, 1, True],   # Blocco 4: 8x expansion, 52 canali out, ripetuto 2x, stride=1
        ]
        input_channel = max(int(32 * width_mult), 8)
        last_channel = max(int(144 * width_mult), 8)

        # Primo strato - Stem layer che processa l'input RGB
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),  # 3 canali RGB -> input_channel
            nn.BatchNorm2d(input_channel),
            nn.GELU()  # GELU per migliore apprendimento rispetto a ReLU
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
modello_scriptato.save('/workspace/tesi-laurea/models/modello_mimir1.pt')
print("Modello salvato come 'modello_mimir1.pt'")

# Generazione report di allenamento
print("\n" + "="*50)
print("REPORT DI ALLENAMENTO")
print("="*50)
print(f"\nModello: MobileNetECA")
print(f"Dataset: CIFAR-10")
print(f"Parametri: {params_formattati} ({params})")
print(f"MACs: {macs_formattati} ({macs})")
print(f"\nConfigurazione:")
print(f"  - Epoche: {epoche}")
print(f"  - Batch size: {dimensione_batch}")
print(f"  - Learning rate iniziale: {tasso_iniziale}")
print(f"  - Width multiplier: {fattore_larghezza}")
print(f"  - LR scale: {lr_scale}")
print(f"  - Dispositivo: {dispositivo}")
print(f"\nAccuratezza finale:")
print(f"  - Training: {acc_train:.2f}%")
print(f"  - Validation: {acc_valid:.2f}%")
print("\n" + "="*50)

# Salva report su file
with open('/workspace/tesi-laurea/reports/training_report.txt', 'w') as f:
    f.write("="*50 + "\n")
    f.write("REPORT DI ALLENAMENTO\n")
    f.write("="*50 + "\n\n")
    f.write(f"Modello: MobileNetECA\n")
    f.write(f"Dataset: CIFAR-10\n")
    f.write(f"Parametri: {params_formattati} ({params})\n")
    f.write(f"MACs: {macs_formattati} ({macs})\n")
    f.write(f"\nConfigurazione:\n")
    f.write(f"  - Epoche: {epoche}\n")
    f.write(f"  - Batch size: {dimensione_batch}\n")
    f.write(f"  - Learning rate iniziale: {tasso_iniziale}\n")
    f.write(f"  - Width multiplier: {fattore_larghezza}\n")
    f.write(f"  - LR scale: {lr_scale}\n")
    f.write(f"  - Dispositivo: {dispositivo}\n")
    f.write(f"\nAccuratezza finale:\n")
    f.write(f"  - Training: {acc_train:.2f}%\n")
    f.write(f"  - Validation: {acc_valid:.2f}%\n")
    f.write("\n" + "="*50 + "\n")
print("\nReport salvato in 'reports/training_report.txt'")
