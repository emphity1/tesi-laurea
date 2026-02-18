
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --- Architettura da Importare (Ricopiata per brevità in standalone) ---
# (In uno scenario reale importeremmo da MobileNetEca_Rep_AdvAug)
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
            self.rbr_dense = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False), nn.BatchNorm2d(out_channels))
            self.rbr_1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False), nn.BatchNorm2d(out_channels))
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):
        if hasattr(self, 'reparam_conv'): return self.activation(self.reparam_conv(inputs))
        id_out = self.rbr_identity(inputs) if self.rbr_identity else 0
        return self.activation(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_conv'): return
        k3, b3 = self._fuse_bn(self.rbr_dense)
        k1, b1 = self._fuse_bn(self.rbr_1x1)
        kid, bid = self._fuse_bn(self.rbr_identity) if self.rbr_identity else (0,0)
        
        # Pad k1 to 3x3
        k1 = F.pad(k1, [1, 1, 1, 1])
        
        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True)
        self.reparam_conv.weight.data = k3 + k1 + kid
        self.reparam_conv.bias.data = b3 + b1 + bid
        
        for p in self.parameters(): p.detach_()
        del self.rbr_dense, self.rbr_1x1, self.rbr_identity

    def _fuse_bn(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, nn.Sequential):
            k, rm, rv, gamma, beta, eps = branch[0].weight, branch[1].running_mean, branch[1].running_var, branch[1].weight, branch[1].bias, branch[1].eps
        else: # Identity BN
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                k_val = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels): k_val[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(k_val).to(branch.weight.device)
            k, rm, rv, gamma, beta, eps = self.id_tensor, branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        
        std = (rv + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return k * t, beta - rm * gamma / std

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=3, b=12):
        super(ECABlock, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

class RepInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=True):
        super().__init__()
        hidden = int(inp * expand_ratio)
        self.use_res = (stride == 1 and inp == oup)
        layers = []
        if expand_ratio != 1: layers.extend([nn.Conv2d(inp, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.GELU()])
        layers.append(RepConv(hidden, hidden, 3, stride, 1, groups=hidden))
        if use_eca: layers.append(ECABlock(hidden))
        layers.extend([nn.Conv2d(hidden, oup, 1, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)
    def forward(self, x): return x + self.conv(x) if self.use_res else self.conv(x)

class MobileNetECARep(nn.Module):
    def __init__(self, num_classes=10, width_mult=0.5):
        super().__init__()
        block_settings = [[1, 20, 2, 1], [6, 32, 4, 2], [8, 42, 4, 2], [8, 52, 2, 1]]
        input_channel = max(int(32 * width_mult), 12)
        last_channel = max(int(144 * width_mult), 12)
        feats = [RepConv(3, input_channel, stride=1)]
        for t, c, n, s in block_settings:
            output_channel = max(int(c * width_mult), 12)
            for i in range(n):
                feats.append(RepInvertedResidual(input_channel, output_channel, s if i==0 else 1, t))
                input_channel = output_channel
        feats.append(nn.Sequential(nn.Conv2d(input_channel, last_channel, 1, bias=False), nn.BatchNorm2d(last_channel), nn.GELU(), nn.AdaptiveAvgPool2d(1)))
        self.features = nn.Sequential(*feats)
        self.classifier = nn.Linear(last_channel, num_classes)
        self.deploy_mode = False
    
    def forward(self, x): 
        return self.classifier(self.features(x).flatten(1))
    
    def deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'): m.switch_to_deploy()
        self.deploy_mode = True

# --- Funzioni di Analisi ---

def occlusion_sensitivity(model, image, label, patch_size=8, stride=4, device='cpu'):
    """
    Scorre un quadrato nero (occlusione) sull'immagine e misura il calo di confidenza
    per la classe corretta.
    """
    model.eval()
    width, height = image.shape[2], image.shape[1]
    output_height = int((height - patch_size) / stride) + 1
    output_width = int((width - patch_size) / stride) + 1
    
    heatmap = np.zeros((output_height, output_width))
    
    # 1. Get baseline confidence
    with torch.no_grad():
        baseline_out = F.softmax(model(image.unsqueeze(0).to(device)), dim=1)
        baseline_prob = baseline_out[0, label].item()
    
    # 2. Iterate occlusion
    for h in range(output_height):
        for w in range(output_width):
            h_start = h * stride
            w_start = w * stride
            h_end = min(height, h_start + patch_size)
            w_end = min(width, w_start + patch_size)
            
            # Create occluded image
            img_occ = image.clone()
            img_occ[:, h_start:h_end, w_start:w_end] = 0 # Black patch (normalized value approx -2) 
            # Note: 0 in tensor depends on normalization. Let's use mean value approx 0.
            
            with torch.no_grad():
                out = F.softmax(model(img_occ.unsqueeze(0).to(device)), dim=1)
                prob = out[0, label].item()
            
            # Heatmap value: Drop in probability (Higher = Sensitive)
            heatmap[h, w] = baseline_prob - prob

    return heatmap, baseline_prob

def run_analysis(checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Analysis on {device}...")
    
    # 1. Load Model
    model = MobileNetECARep(width_mult=0.5).to(device)
    # Prova a caricare state dict (gestendo il fatto che potrebbe essere salvato intero o dict)
    try:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print("Model state dict loaded.")
    except:
        print("Warning: Could not load state dict directly. Training script might save full model.")
        return

    # Switch to deploy for analysis (faster)
    model.deploy()
    model.eval()
    
    # 2. Load Test Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 3. Occlusion Sensitivity Test (Su 5 immagini casuali corrette)
    print("\nGenerazione Mappe Sensibilità Occlusione...")
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    idxs = np.random.choice(len(testset), 50) # Prendine 50 a caso, filtreremo i primi 5 corretti
    
    count = 0
    for idx in idxs:
        if count >= 5: break
        
        img, label = testset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
        
        if pred == label:
            heatmap, base_prob = occlusion_sensitivity(model, img, label, device=device)
            
            # Unnormalize image for display
            img_disp = img.permute(1, 2, 0).numpy()
            img_disp = img_disp * np.array([0.247, 0.243, 0.261]) + np.array([0.4914, 0.4822, 0.4465])
            img_disp = np.clip(img_disp, 0, 1)
            
            # Upsample heatmap to image size
            import cv2
            heatmap_resized = cv2.resize(heatmap, (32, 32))
            heatmap_resized = np.maximum(heatmap_resized, 0) # Clip negative drops
            heatmap_resized = heatmap_resized / (heatmap_resized.max() + 1e-6) # Normalize
            
            axes[0, count].imshow(img_disp)
            axes[0, count].set_title(f"Orig: {classes[label]}")
            axes[0, count].axis('off')
            
            axes[1, count].imshow(heatmap_resized, cmap='jet')
            axes[1, count].set_title(f"Sensitivity Map")
            axes[1, count].axis('off')
            
            # Overlay
            axes[2, count].imshow(img_disp)
            axes[2, count].imshow(heatmap_resized, cmap='jet', alpha=0.5)
            axes[2, count].set_title(f"Overlaid")
            axes[2, count].axis('off')
            
            count += 1
            
    os.makedirs('reports/analysis', exist_ok=True)
    plt.savefig('reports/analysis/occlusion_sensitivity.png')
    print("Saved occlusion_sensitivity.png")
    
    # 4. Find Top Confusions (High Confidence Errors)
    print("\nRicerca Errori ad Alta Confidenza...")
    high_conf_errors = []
    
    dataloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = F.softmax(model(data), dim=1)
            probs, preds = output.max(1)
            
            # Trova errori
            incorrect_mask = preds.ne(target)
            high_conf_mask = probs > 0.9 # Errori sicuri (>90%)
            mask = incorrect_mask & high_conf_mask
            
            if mask.sum() > 0:
                err_idxs = mask.nonzero(as_tuple=True)[0]
                for err_i in err_idxs:
                    # Salva (img_tensor, true_label, pred_label, confidence)
                    high_conf_errors.append((data[err_i].cpu(), target[err_i].cpu().item(), preds[err_i].cpu().item(), probs[err_i].cpu().item()))
                    if len(high_conf_errors) >= 10: break
            if len(high_conf_errors) >= 10: break
            
    # Visualize Errors
    if high_conf_errors:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, (img, true_l, pred_l, conf) in enumerate(high_conf_errors[:10]):
            img_disp = img.permute(1, 2, 0).numpy()
            img_disp = img_disp * np.array([0.247, 0.243, 0.261]) + np.array([0.4914, 0.4822, 0.4465])
            img_disp = np.clip(img_disp, 0, 1)
            
            row = i // 5
            col = i % 5
            axes[row, col].imshow(img_disp)
            axes[row, col].set_title(f"True: {classes[true_l]}\nPred: {classes[pred_l]}\nConf: {conf:.2f}", color='red')
            axes[row, col].axis('off')
            
        plt.tight_layout()
        plt.savefig('reports/analysis/top_errors.png')
        print("Saved top_errors.png")
    else:
        print("Nessun errore ad alta confidenza trovato! (Ottimo modello o soglia troppo alta)")

if __name__ == "__main__":
    # Assicurarsi che il path punti al checkpoint corretto
    CKPT_PATH = '/workspace/tesi-laurea/reports/eca_vs_eca_parametrized/best_model_training.pth' 
    # Fallback path if running locally
    if not os.path.exists(CKPT_PATH):
        CKPT_PATH = 'reports/eca_vs_eca_parametrized/best_model_training.pth'
        
    if os.path.exists(CKPT_PATH):
        run_analysis(CKPT_PATH)
    else:
        print(f"Checkpoint non trovato in: {CKPT_PATH}. Esegui prima il training.")
