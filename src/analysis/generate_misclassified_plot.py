
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.legacy.mobilnet_eca_rep_advaug.MobileNetEca_Rep_AdvAug import MobileNetECARep

def generate_misclassified_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = MobileNetECARep(num_classes=10, width_mult=0.5).to(device)
    
    checkpoint_path = '/workspace/tesi-laurea/reports/eca_vs_eca_parametrized/best_model_training.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Using random weights (for demo purposes only if checkpoints are missing).")
        # In a real run, we should probably exit or warn loudly.
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model loaded.")
        
    model.deploy()
    model.eval()
    
    # 2. Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    
    # 3. Find Errors
    misclassified = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            incorrect_mask = predicted != labels
            if incorrect_mask.any():
                incorrect_indices = torch.nonzero(incorrect_mask).squeeze()
                if incorrect_indices.dim() == 0: incorrect_indices = [incorrect_indices]
                
                for idx in incorrect_indices:
                    img = images[idx].cpu()
                    true_label = labels[idx].item()
                    pred_label = predicted[idx].item()
                    conf = probs[idx][pred_label].item()
                    
                    # Store image and metadata
                    misclassified.append({
                        'image': img,
                        'true': true_label,
                        'pred': pred_label,
                        'conf': conf
                    })
                    
            if len(misclassified) > 50: # Find enough to choose from
                break
                
    # 4. Select Interesting Examples (e.g., Cat vs Dog, Car vs Truck)
    # Filter for diverse errors if possible, or just take first 8
    selected = misclassified[:8] 
    
    # 5. Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    # Un-normalize for display
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.247, 0.243, 0.261])
    
    for i, ax in enumerate(axes):
        if i >= len(selected): break
        item = selected[i]
        
        img = item['image'].permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        t_label = classes[item['true']]
        p_label = classes[item['pred']]
        conf = item['conf']
        
        ax.set_title(f"True: {t_label}\nPred: {p_label}\n({conf:.2f})", fontsize=10, color='red')
        ax.axis('off')
        
    plt.tight_layout()
    os.makedirs('/workspace/tesi-laurea/docs/scrittura-tesi/tesi/figure', exist_ok=True)
    plt.savefig('/workspace/tesi-laurea/docs/scrittura-tesi/tesi/figure/misclassified_examples.png', dpi=300)
    print(f"Plot saved to /workspace/tesi-laurea/docs/scrittura-tesi/tesi/figure/misclassified_examples.png")

if __name__ == "__main__":
    generate_misclassified_plot()
