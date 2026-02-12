import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn

"""
ResNetLight - Modello compatto per CIFAR-10
Architettura: 16→32→64 canali con BasicBlock standard
Target: ~100k parametri
"""

class Cutout(object):
    """Data augmentation: rimuove patch quadrata random dall'immagine"""
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class BasicBlock(nn.Module):
    """Blocco residuale standard con skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNetLight(nn.Module):
    """ResNet leggero: 16→32→64 canali, 2 blocchi per stage"""
    def __init__(self, num_classes=10):
        super(ResNetLight, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

class ResNetTrainer:
    def __init__(self, epochs=50, batch_size=128, base_lr=0.001, max_lr=0.01, weight_decay=1e-4, cutout_length=16):
        self.epochs = epochs
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.cutout_length = cutout_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0.0
        self.train_acc_history = []
        self.val_acc_history = []

        self.set_seed(42)

        # Data augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(self.cutout_length)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        self.model = ResNetLight(num_classes=10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.max_lr, steps_per_epoch=len(self.trainloader), epochs=self.epochs)

        self.print_config()

    def print_config(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("CONFIGURAZIONE RESNETLIGHT")
        print("="*60)
        print(f"Modello: ResNetLight (16→32→64)")
        print(f"Dataset: CIFAR-10")
        print(f"Parametri totali: {total_params:,}")
        print(f"Parametri addestrabili: {trainable_params:,}")
        print(f"\nTraining:")
        print(f"  - Epoche: {self.epochs}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Base LR: {self.base_lr}")
        print(f"  - Max LR: {self.max_lr}")
        print(f"  - Dispositivo: {self.device}")
        print(f"\nAugmentation:")
        print(f"  - RandomCrop + Flip + Rotation + ColorJitter + Cutout({self.cutout_length})")
        print("="*60 + "\n")

    def set_seed(self, seed=0):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.use_deterministic_algorithms(False)

    def train(self):
        print("🚀 Inizio training...\n")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            epoch_loss = 0.0
            num_batches = 0
            
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            avg_loss = epoch_loss / num_batches
            self.train_acc_history.append(train_acc)
            
            val_acc = self.test()
            self.val_acc_history.append(val_acc)
            
            print(f'Epoca {epoch+1} - Accuratezza Allenamento: {train_acc:.2f}% - Accuratezza Validazione: {val_acc:.2f}% - Loss: {avg_loss:.4f}')
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model('/workspace/tesi-laurea/models/resnet_light_check1.pt')
                print(f'  ✅ Nuovo miglior modello salvato! Accuracy: {val_acc:.2f}%')

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_scripted = torch.jit.script(self.model)
        model_scripted.save(path)

    def generate_report(self, output_path='/workspace/tesi-laurea/reports/resnet_light_check1_report.txt'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        with open(output_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("REPORT DI TRAINING - ResNetLight (check1)\n")
            f.write("="*50 + "\n\n")
            f.write(f"Architettura: ResNetLight (16→32→64 canali)\n")
            f.write(f"Dataset: CIFAR-10\n")
            f.write(f"Parametri totali: {total_params:,}\n")
            f.write(f"\nConfigurazione Training:\n")
            f.write(f"  - Epoche: {self.epochs}\n")
            f.write(f"  - Batch size: {self.batch_size}\n")
            f.write(f"  - Base LR: {self.base_lr}\n")
            f.write(f"  - Max LR: {self.max_lr}\n")
            f.write(f"  - Weight decay: {self.weight_decay}\n")
            f.write(f"  - Cutout length: {self.cutout_length}\n")
            f.write(f"  - Dispositivo: {self.device}\n")
            f.write(f"\nRisultati Finali:\n")
            if self.train_acc_history:
                f.write(f"  - Training Accuracy: {self.train_acc_history[-1]:.2f}%\n")
            if self.val_acc_history:
                f.write(f"  - Validation Accuracy: {self.val_acc_history[-1]:.2f}%\n")
            f.write(f"  - Best Validation Accuracy: {self.best_acc:.2f}%\n")
            f.write("\n" + "="*50 + "\n")
        
        print(f"\n✅ Report salvato in: {output_path}")
        print("\n" + "="*50)
        print("REPORT FINALE")
        print("="*50)
        print(f"Modello: ResNetLight (check1)")
        print(f"Parametri: {total_params:,}")
        print(f"Best Validation Accuracy: {self.best_acc:.2f}%")
        print("="*50 + "\n")

    def get_model(self):
        return self.model


if __name__ == "__main__":
    trainer = ResNetTrainer(
        epochs=50,
        batch_size=128,
        base_lr=0.001,
        max_lr=0.01,
        weight_decay=1e-4,
        cutout_length=16
    )
    
    trainer.train()
    trainer.generate_report()
    
    print("\n🎉 Training completato!")
