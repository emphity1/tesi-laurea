import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import math
import numpy as np

"""
MobileNetECA KD - Knowledge Distillation
Teacher: ResNet-18 (ImageNet Pre-trained + CIFAR-10 Adaptation)
Student: MobileNetECA Reparameterized (v1.5)
Obiettivo: >93% Accuracy trasferendo conoscenza da ResNet a MobileNet.
"""

# ========== Parametri =======================
n_classi = 10
tasso_iniziale = 0.05
epoche = 50  # Mettiamo 50 per test veloce, ma ideally 200
dimensione_batch = 128
fattore_larghezza = 0.5
lr_scale = 1.54
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'

# KD Parameters
TEMP = 4.0      # Temperatura per ammorbidire le probability distribution
ALPHA = 0.5     # Peso: 0.5 KLDiv (Teacher) + 0.5 CrossEntropy (Labels)
# ============================================


# --- STUDENT ARCHITECTURE (Reparameterized + ECA) ---
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


# --- DISTILLATION LOSS ---
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    # Hard Loss (Standard CrossEntropy)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft Loss (KL Divergence)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    return alpha * hard_loss + (1. - alpha) * soft_loss


# --- MAIN ---
if __name__ == "__main__":
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

    # 1. SETUP TEACHER (ResNet-18)
    print(f"\n{'='*70}")
    print(f"PREPARAZIONE TEACHER (ResNet-18)")
    print(f"{'='*70}")
    # Scarica ResNet18 pre-allenata ImageNet
    teacher_model = models.resnet18(weights='DEFAULT')
    # Adatta ultimo layer a CIFAR-10
    num_ftrs = teacher_model.fc.in_features
    teacher_model.fc = nn.Linear(num_ftrs, 10)
    teacher_model = teacher_model.to(dispositivo)
    
    # Fine-tuning rapido del Teacher (solo 1 epoca se serve, o usiamo così com'è se già forte)
    # NOTA: Una ResNet18 da ImageNet su CIFAR-10 senza fine-tuning performa MALE (random output su classi diverse).
    # Dobbiamo fare un "Teacher Warmup" veloce (es. 5 epoche) per dargli dignità.
    print("Fine-tuning rapido del Teacher su CIFAR-10 (5 Epoche)...")
    optimizer_teacher = optim.SGD(teacher_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # LR basso
    criterion_teacher = nn.CrossEntropyLoss()
    
    teacher_model.train()
    for ep in range(5):
        corr, tot = 0, 0
        for inp, lbl in caricatore_train:
            inp, lbl = inp.to(dispositivo), lbl.to(dispositivo)
            optimizer_teacher.zero_grad()
            out = teacher_model(inp)
            loss = criterion_teacher(out, lbl)
            loss.backward()
            optimizer_teacher.step()
            _, pred = out.max(1)
            corr += pred.eq(lbl).sum().item()
            tot += lbl.size(0)
        print(f"Teacher Warmup Ep {ep+1}/5: Acc {100.*corr/tot:.1f}%")
        
    print("Teacher Pronto! Congelamento pesi...")
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Validate Teacher
    t_corr, t_tot = 0, 0
    with torch.no_grad():
        for inp, lbl in caricatore_test:
            inp, lbl = inp.to(dispositivo), lbl.to(dispositivo)
            out = teacher_model(inp)
            _, pred = out.max(1)
            t_corr += pred.eq(lbl).sum().item()
            t_tot += lbl.size(0)
    print(f"Teacher Validation Accuracy: {100.*t_corr/t_tot:.2f}% (Target per lo Student)")
    print(f"{'='*70}\n")


    # 2. SETUP STUDENT
    student_model = MobileNetECARep(num_classes=n_classi, width_mult=fattore_larghezza).to(dispositivo)
    params_student = sum(p.numel() for p in student_model.parameters())
    print(f"STUDENT MODEL: MobileNetECARep")
    print(f"Parametri Student (Train): {formatta_numero(params_student)}")
    print(f"Inizio Distillation Training...")

    ottimizzatore = optim.SGD(student_model.parameters(), lr=tasso_iniziale, momentum=0.9, weight_decay=5e-4)
    schedulatore = CosineAnnealingLR(ottimizzatore, T_max=epoche)

    best_acc = 0.0

    for epoca in range(epoche):
        student_model.train()
        corretti_train, totale_train = 0, 0
        running_loss = 0.0
        
        for input, obiettivi in caricatore_train:
            input, obiettivi = input.to(dispositivo), obiettivi.to(dispositivo)
            
            # Forward Teacher (No Grad)
            with torch.no_grad():
                teacher_logits = teacher_model(input)
            
            # Forward Student
            ottimizzatore.zero_grad()
            student_logits = student_model(input)
            
            # KD Loss
            loss = distillation_loss(student_logits, teacher_logits, obiettivi, T=TEMP, alpha=ALPHA)
            loss.backward()
            
            nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=5)
            ottimizzatore.step()
            
            running_loss += loss.item()
            _, predetti = student_logits.max(1)
            corretti_train += predetti.eq(obiettivi).sum().item()
            totale_train += obiettivi.size(0)
            
        acc_train = 100. * corretti_train / totale_train
        avg_loss = running_loss / len(caricatore_train)
        
        # Validation
        student_model.eval()
        corretti_val, totale_val = 0, 0
        with torch.no_grad():
            for input, obiettivi in caricatore_test:
                input, obiettivi = input.to(dispositivo), obiettivi.to(dispositivo)
                uscite = student_model(input)
                _, predetti = uscite.max(1)
                corretti_val += predetti.eq(obiettivi).sum().item()
                totale_val += obiettivi.size(0)
        
        acc_valid = 100. * corretti_val / totale_val
        schedulatore.step()
        
        if acc_valid > best_acc:
            best_acc = acc_valid
            
        print(f'Epoca {epoca+1:02d}/{epoche} - Loss: {avg_loss:.4f} | Train: {acc_train:.2f}% | Val: {acc_valid:.2f}% (Best: {best_acc:.2f}%)')

    print(f"\nDeploying Student Model...")
    student_model.deploy()
    params_final = sum(p.numel() for p in student_model.parameters())
    print(f"Parametri Finali Student: {formatta_numero(params_final)}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
