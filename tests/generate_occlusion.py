"""
Genera la Occlusion Sensitivity Map per il modello D (MobileNetECA-Rep-AdvAug).
Seleziona 5 immagini ben classificate e mostra dove il modello "guarda".
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_D_eca_rep_advaug import MobileNetECARep
from shared_config import NORM_MEAN, NORM_STD, SEED

CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carica modello deploy
model = MobileNetECARep().to(device)
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_D_eca_rep_advaug', 'best_model.pth')
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Modello caricato da {ckpt_path}")

# Dataset test (normalizzato per il modello)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

# Dataset test RAW (per visualizzazione)
testset_raw = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

# Seleziona 5 immagini classificate correttamente, una per classe diversa
np.random.seed(SEED)
selected = []
target_classes = [0, 1, 3, 5, 8]  # Airplane, Automobile, Cat, Dog, Ship

for cls in target_classes:
    class_indices = [i for i in range(len(testset)) if testset.targets[i] == cls]
    for idx in class_indices:
        img, label = testset[idx]
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))
            pred = out.argmax(1).item()
        if pred == label:
            selected.append(idx)
            break

print(f"Immagini selezionate: {selected}")

# Occlusion sweep
def occlusion_sensitivity(model, img_tensor, true_class, patch_size=4, stride=2):
    """Calcola la mappa di sensibilità all'occlusione."""
    _, H, W = img_tensor.shape
    heatmap = np.zeros((H, W))
    count = np.zeros((H, W))

    # Confidenza originale
    with torch.no_grad():
        orig_out = model(img_tensor.unsqueeze(0).to(device))
        orig_conf = F.softmax(orig_out, dim=1)[0, true_class].item()

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = img_tensor.clone()
            occluded[:, y:y+patch_size, x:x+patch_size] = 0  # Patch nero

            with torch.no_grad():
                out = model(occluded.unsqueeze(0).to(device))
                conf = F.softmax(out, dim=1)[0, true_class].item()

            drop = orig_conf - conf  # Quanto cala la confidenza
            heatmap[y:y+patch_size, x:x+patch_size] += drop
            count[y:y+patch_size, x:x+patch_size] += 1

    count[count == 0] = 1
    heatmap /= count
    return heatmap, orig_conf

# Genera le mappe
print("Generazione mappe di occlusione...")
fig, axes = plt.subplots(3, len(selected), figsize=(3*len(selected), 9))

for col, idx in enumerate(selected):
    img_tensor, label = testset[idx]
    img_raw, _ = testset_raw[idx]
    img_np = img_raw.permute(1, 2, 0).numpy()

    heatmap, orig_conf = occlusion_sensitivity(model, img_tensor, label, patch_size=4, stride=1)

    # Riga 1: Immagine originale
    axes[0, col].imshow(img_np)
    axes[0, col].set_title(f'{CLASSES[label]} ({orig_conf*100:.1f}%)', fontsize=10)
    axes[0, col].axis('off')

    # Riga 2: Heatmap
    im = axes[1, col].imshow(heatmap, cmap='hot', interpolation='bilinear')
    axes[1, col].set_title('Sensitivity', fontsize=10)
    axes[1, col].axis('off')

    # Riga 3: Overlay
    axes[2, col].imshow(img_np)
    axes[2, col].imshow(heatmap, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[2, col].set_title('Overlay', fontsize=10)
    axes[2, col].axis('off')

    print(f"  {CLASSES[label]}: done")

axes[0, 0].set_ylabel('Originale', fontsize=12, rotation=90, labelpad=10)
axes[1, 0].set_ylabel('Heatmap', fontsize=12, rotation=90, labelpad=10)
axes[2, 0].set_ylabel('Overlay', fontsize=12, rotation=90, labelpad=10)

fig.suptitle('Mappe di Sensibilità all\'Occlusione - MobileNetECA-Rep-AdvAug', fontsize=14, y=1.02)
fig.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'occlusion_sensitivity.png')
fig.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\n✓ Salvato: {output_path}")
print("Copia in Tesi_Dmytro_Kozak/figure/")
