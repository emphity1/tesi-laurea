
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Set seed per riproducibilità
torch.manual_seed(42)

def unnormalize(tensor):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = tensor * std + mean
    return np.clip(tensor, 0, 1)

# 1. Definizione Trasformazioni (Tua Pipeline Esatta)
# Pipeline "Base" (Solo Resize/Normalize per visualizzazione pulita)
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # Valori corretti dal tuo main.py
])

# Pipeline "Advanced Augmentation" (Quella aggressiva del tuo training)
transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0), # Forziamo il flip per vederlo
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.RandomErasing(p=1.0, scale=(0.10, 0.33), ratio=(0.3, 3.3), value=0) # Forziamo Erasing p=1.0 per vederlo
])

# 2. Caricamento Dataset (Solo poche immagini)
# Usiamo transform=None qui per avere le immagini PIL originali
dataset_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

# Indici scelti manualmente per classi diverse (Nave, Auto, Cane, Rana)
indices = [0, 1, 3, 4] # Modifica se vuoi altre immagini
class_names = ['Frog', 'Truck', 'Deer', 'Car'] # CIFAR classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

# 3. Creazione Griglia
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
plt.subplots_adjust(wspace=0.1, hspace=0.2)

for i, idx in enumerate(indices):
    img_pil, label = dataset_raw[idx]
    
    # Riga 1: Originale (pulita)
    # Applichiamo transform base solo per avere il tensore normalizzato coerente col codice, poi denormalizziamo per plot
    img_tensor_base = transform_base(img_pil)
    img_show_base = unnormalize(img_tensor_base)
    
    axes[0, i].imshow(img_show_base)
    axes[0, i].set_title(f"Original: {class_names[i]}")
    axes[0, i].axis('off')
    
    # Riga 2: Augmentata (Aggressiva)
    # Nota: AutoAugment lavora su PIL, RandomErasing su Tensor. La pipeline gestisce tutto.
    # Per visualizzare l'effetto random, potremmo dover provare più volte, ma con seed fisso sarà deterministico.
    # Qui applichiamo la pipeline completa all'immagine PIL originale
    img_tensor_aug = transform_aug(img_pil)
    img_show_aug = unnormalize(img_tensor_aug)
    
    axes[1, i].imshow(img_show_aug)
    axes[1, i].set_title(f"Augmented\n(Cutout + AutoAug)")
    axes[1, i].axis('off')

# 4. Salvataggio
os.makedirs('figure_generated', exist_ok=True)
save_path = 'figure_generated/augmentation_comparison.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Immagine salvata in: {save_path}")

# Opzionale: Mostra a schermo se possibile (in notebook)
# plt.show()
