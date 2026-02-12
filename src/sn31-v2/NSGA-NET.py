import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import json
import random

#Codice completo per 31
# prendere e creare un file json da wandb con params,flops,accuracy.



# Definizione di ResNetLight
class ResNetLight(nn.Module):
    def __init__(self, num_classes=10, num_blocks=[2, 2, 2], num_channels=[16, 32, 64]):
        super(ResNetLight, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(num_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(num_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(num_channels[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(num_channels[2], num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
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

# Definizione di ResNetBlock
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
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

# DummyTrainer per l'addestramento e la valutazione
class DummyTrainer:
    def __init__(self, model=None, epochs=50, batch_size=128, learning_rate=0.01, weight_decay=1e-4):
        # Se il modello non è fornito, avvia NSGA-Net per trovare un'architettura
        if model is None:
            nsga = NSGA_Net(population_size=20, generations=20, mutation_rate=0.4)
            model = nsga.evolve()[0]  # Prendi la prima architettura migliore trovata

        self.model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Caricamento dei dati e normalizzazione con semplice augmentation
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=15)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=15)

        # Ottimizzatore e scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)

        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.epochs):
            self.scheduler.step()
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

                running_loss += loss.item()
                if i % 100 == 99:
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


class NSGA_Net:
    def __init__(self, population_size=8, generations=6, mutation_rate=0.4):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        self.architetture_esistenti = self.load_architectures()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            model = ResNetLight(num_classes=10)
            population.append(model)
        return population

    def load_architectures(self):
        with open('/workspace/Dima/models/wandb.json', 'r') as f:
            return json.load(f)

    def save_architectures(self):
        with open('/workspace/Dima/models/wandb.json', 'w') as f:
            json.dump(self.architetture_esistenti, f, indent=4)

    def arrotonda_parametri(self, parametri):
        return (parametri // 1000) * 1000

    def is_better_architecture(self, new_arch, existing_arch):
        return (
            new_arch['accuracy'] > existing_arch['accuracy'] and
            new_arch['flops'] < existing_arch['flops']
        )

    def evaluate(self, model):
        trainer = DummyTrainer(model=model)
        trainer.train()
        accuracy = trainer.test()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        flops = self.compute_flops(model)

        # Arrotonda i parametri
        rounded_params = self.arrotonda_parametri(num_params)

        # Nuova architettura generata
        nuova_architettura = {
            "params": rounded_params,
            "flops": flops,
            "accuracy": accuracy
        }

        # Trova l'indice della riga successiva nel JSON
        indice_successivo = None
        for i, arch_esistente in enumerate(self.architetture_esistenti):
            if rounded_params < int(arch_esistente['params']):
                indice_successivo = i
                break

        architettura_ottima = False

        # Confronto con la riga successiva
        if indice_successivo is not None:
            arch_successiva = self.architetture_esistenti[indice_successivo]
            if self.is_better_architecture(nuova_architettura, arch_successiva):
                architettura_ottima = True

        # Confronto con la riga precedente (se esiste)
        if indice_successivo is not None and indice_successivo > 0:
            arch_precedente = self.architetture_esistenti[indice_successivo - 1]
            if self.is_better_architecture(nuova_architettura, arch_precedente):
                architettura_ottima = True

        # Se l'architettura è ottima, aggiungila
        if architettura_ottima:
            self.architetture_esistenti.append(nuova_architettura)
            self.save_architectures()
            print(f"Nuova architettura ottima trovata: {nuova_architettura}")
        else:
            print("La nuova architettura non migliora le esistenti.")

        return accuracy, num_params, flops

    def compute_flops(self, model):
        # Implementazione per calcolare i FLOPs
        flops = 0
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                output_size = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                flops += output_size
        return flops

    def mutate(self, model):
        if random.random() < self.mutation_rate:
            for layer in model.modules():
                if isinstance(layer, nn.Conv2d):
                    layer.out_channels = random.choice([8, 16, 32, 48])  # Ridotti i canali
                    layer.kernel_size = random.choice([3, 5])
                if isinstance(layer, nn.Dropout):
                    layer.p = random.choice([0.3, 0.5])
        return model

    def crossover(self, parent1, parent2):
        child_model = ResNetLight(num_classes=10)
        # Implementazione avanzata del crossover
        for layer1, layer2 in zip(parent1.modules(), parent2.modules()):
            if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
                if random.random() > 0.5:
                    child_model.add_layer(layer1.out_channels, layer1.kernel_size)
                else:
                    child_model.add_layer(layer2.out_channels, layer2.kernel_size)
        return child_model

    def selection(self, ranked_population):
        return [model for model, _, _, _ in ranked_population[:self.population_size]]

    def evolve(self):
        for generation in range(self.generations):
            print(f"Generation {generation + 1}")
            evaluated_population = [(model, *self.evaluate(model)) for model in self.population]
            ranked_population = self.non_dominated_sort(evaluated_population)
            next_population = self.selection(ranked_population)
            next_population = [self.mutate(model) for model in next_population]
            next_population += [self.crossover(random.choice(next_population), random.choice(next_population)) for _ in range(len(next_population) // 2)]
            self.population = next_population
        return self.population
