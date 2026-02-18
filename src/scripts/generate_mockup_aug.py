
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import requests
import io
import os
import random

def get_real_cifar_sample(id=0):
    # Usiamo URL diretti di CIFAR10 sample per realismo
    # ID: 0 (Frog), 1 (Truck), 3 (Deer), 4 (Car)
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/European_Common_Frog_Rana_temporaria.jpg/320px-European_Common_Frog_Rana_temporaria.jpg", # Frog (simulato)
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Fire_Truck_ladder.jpg/320px-Fire_Truck_ladder.jpg", # Truck
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Peacock_on_fence.jpg/320px-Peacock_on_fence.jpg", # Bird/Deer like
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Neckarsulm_2016_Audi_R8_V10_Plus_5.2_FSI_quattro_Front_2.jpg/320px-Neckarsulm_2016_Audi_R8_V10_Plus_5.2_FSI_quattro_Front_2.jpg" # Car
    ]
    try:
        response = requests.get(urls[id])
        img = Image.open(io.BytesIO(response.content)).resize((32, 32)) # Simuliamo la bassa risoluzione
        return img
    except:
        # Fallback a noise se internet manca
        return Image.new('RGB', (32, 32), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))

def apply_mock_cutout(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Cutout size random tra 8 e 16 pixel (aggressivo, ~33%)
    cut_w = random.randint(8, 14)
    cut_h = random.randint(8, 14)
    
    # Posizione random
    x = random.randint(0, w - cut_w)
    y = random.randint(0, h - cut_h)
    
    # Disegna quadrato nero (o grigio medio dataset)
    draw.rectangle([x, y, x+cut_w, y+cut_h], fill=(0,0,0)) # Nero come Cutout standard
    return img

def apply_mock_autoaugment(img):
    # Simuliamo trasformazioni tipiche di AutoAugment: 
    # Shear, Rotate, Contrast, Solarize
    
    # Scelta random per varietà
    ops = [
        lambda i: ImageOps.solarize(i, threshold=128),
        lambda i: ImageEnhance.Contrast(i).enhance(2.0), # Forte contrasto
        lambda i: i.rotate(15),
        lambda i: ImageEnhance.Color(i).enhance(0.5) # Desaturazione
    ]
    
    op = random.choice(ops)
    img = op(img)
    
    # Flip orizzontale (molto comune)
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
        
    return img

# Creazione Griglia
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
# Titoli
axes[0,0].set_ylabel("Original", fontsize=12, fontweight='bold')
axes[1,0].set_ylabel("Augmented\n(Cutout + AutoAug)", fontsize=12, fontweight='bold')

titles = ['Immagine 1', 'Immagine 2', 'Immagine 3', 'Immagine 4']

for i in range(4):
    # Originale
    img_orig = get_real_cifar_sample(i)
    axes[0, i].imshow(img_orig)
    axes[0, i].set_title(titles[i])
    axes[0, i].axis('off') # Nascondi assi
    
    # Augmentata (Copia per non modificare originale)
    img_aug = img_orig.copy()
    img_aug = apply_mock_autoaugment(img_aug) # Prima distorsione colore/geom
    img_aug = apply_mock_cutout(img_aug)      # Poi Cutout (mascheramento)
    
    # Simuliamo il resize/crop (zoom leggero)
    w, h = img_aug.size
    crop_pixels = 2
    img_aug = img_aug.crop((crop_pixels, crop_pixels, w-crop_pixels, h-crop_pixels)).resize((32,32))
    
    axes[1, i].imshow(img_aug)
    axes[1, i].axis('off')

plt.tight_layout()
os.makedirs('figure_generated', exist_ok=True)
save_path = 'figure_generated/mockup_augmentation.png'
plt.savefig(save_path, dpi=300)
print(f"Salvata mockup in: {save_path}")
