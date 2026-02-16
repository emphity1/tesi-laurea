import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ========== Model Definition (copied for standalone execution) ==========
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

# ========================================================================

def evaluate_model():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/workspace/tesi-laurea/reports/adv_aug_test/mobilenet_eca_reparam_deployed.pth'
    output_dir = '/workspace/tesi-laurea/reports/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load Model
    model = MobileNetECARep(num_classes=10, width_mult=0.5).to(device)
    # Since the saved model is already deployed (RepConv switched to Conv2d), we need to handle loading
    # Ideally, we instantiate the model, call deploy(), then load state_dict
    model.deploy() 
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Metrics Collection
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 1. Classification Report
    print("\nClassification Report:")
    report = classification_report(all_targets, all_preds, target_names=classes)
    print(report)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 2. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Saved confusion_matrix.png to {output_dir}")

    # 3. ROC Curve
    # Binarize output
    y_test_bin = label_binarize(all_targets, classes=range(10))
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {classes[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    print(f"Saved roc_curve.png to {output_dir}")

    # 3b. Zoomed ROC Curve
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {classes[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 0.2])
    plt.ylim([0.8, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Zoomed Top-Left)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve_zoomed.png'))
    plt.close()
    print(f"Saved roc_curve_zoomed.png to {output_dir}")

    # 4. FLOPs and Params
    try:
        from thop import profile
        input_dummy = torch.randn(1, 3, 32, 32).to(device)
        macs, params = profile(model, inputs=(input_dummy, ), verbose=False)
        print(f"\nComputational Metrics:")
        print(f"Params: {params/1000:.2f}k")
        print(f"MACs: {macs/1e6:.2f}M") # FLOPs ~ 2 * MACs
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Params: {params/1000:.2f}k\n")
            f.write(f"MACs: {macs/1e6:.2f}M\n")
    except ImportError:
        print("\nthop not installed. Skipping FLOPs calculation.")
        params = sum(p.numel() for p in model.parameters())
        print(f"Params (manual count): {params/1000:.2f}k")
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Params: {params/1000:.2f}k\n")
            f.write("MACs: N/A (thop missing)\n")

if __name__ == "__main__":
    evaluate_model()
