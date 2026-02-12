import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import torch.backends.cudnn as cudnn

class Cutout(object):
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
    def __init__(self, num_classes=10):
        super(ResNetLight, self).__init__()
        self.in_channels = 16  # Inizia con 16 canali
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 1, stride=1)  # Ridotto a 1 blocco
        self.layer2 = self._make_layer(32, 1, stride=2)  # Ridotto a 32 canali e 1 blocco
        self.layer3 = self._make_layer(64, 1, stride=2)  # Ridotto a 64 canali e 1 blocco
        self.linear = nn.Linear(64, num_classes)  # Adattato per il layer finale

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

class WrappedModel(nn.Module):
    def __init__(self, model, num_classes, device):
        super(WrappedModel, self).__init__()
        self.model = model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Spostare l'input di esempio sul dispositivo corretto
        example_input = torch.randn(1, 3, 32, 32).to(device)
        self.model = self.model.to(device)
        
        # Calcolare la dimensione dell'output
        with torch.no_grad():
            example_output = self.model(example_input)
        print("Output size after model:", example_output.size())
        
        self.fc = nn.Linear(example_output.size(1), num_classes)

    def forward(self, x):
        x = self.model(x)
        if x.dim() == 4:  # Controlla se l'output è un tensore 4D
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # Appiattire il tensor
        x = self.fc(x)
        return x

class DummyTrainer:
    def __init__(self, model_path='/workspace/NASChain/model/model.pt', epochs=50, batch_size=512, base_lr=0.001, max_lr=0.01, weight_decay=1e-4, cutout_length=16):
        self.epochs = epochs
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.cutout_length = cutout_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.set_seed(42)

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
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=50)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=50)

        # Caricare il modello pre-addestrato
        try:
            torchscript_model = torch.jit.load(model_path)
            self.model = WrappedModel(torchscript_model, num_classes=10, device=self.device).to(self.device)
        except Exception as e:
            print(f"Error loading the model: {e}")
            # Se il caricamento del modello fallisce, utilizzare il modello standard
            self.model = ResNetLight(num_classes=10).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
        
        # Warmup scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.max_lr, steps_per_epoch=len(self.trainloader), epochs=self.epochs)
        
        # Stampa del numero dei parametri
        self.print_model_parameters()

    def print_model_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total number of parameters: {total_params}')

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
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
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
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            self.test()

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

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

    def get_model(self):
        return self.model
