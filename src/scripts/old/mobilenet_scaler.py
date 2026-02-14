import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ptflops import get_model_complexity_info

# ========== Configurable Parameters ==========
n_classes = 10  # Number of classes for CIFAR-10
initial_rate = 0.025  # Initial learning rate
epochs = 50  # Number of epochs
batch_size = 128  # Batch size
width_factor = 0.4  # Factor to adjust model capacity
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use CUDA if available
lr_scale = 1.44  # Learning rate scale

# ============================================

# Function to format numbers
def format_number(num):
    if abs(num) >= 1_000_000:
        return f'{num / 1_000_000:.1f}M'
    elif abs(num) >= 1_000:
        return f'{num / 1_000:.1f}k'
    else:
        return str(num)

# Improved Squeeze-Excitation (SE) block
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4, lr_scale=1.6):
        super(SqueezeExcitation, self).__init__()
        self.lr_scale = lr_scale
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def grad_scale(self, x):
        return x * self.lr_scale + x.detach() * (1 - self.lr_scale)

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc(scale)
        scale = self.grad_scale(scale)  # Apply gradient scaling
        return x * scale

# Improved Inverted Residual Block
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expansion, use_se=False, se_ratio=0.25, lr_scale=1.6):
        super(InvertedResidualBlock, self).__init__()
        self.lr_scale = lr_scale
        hidden_dim = int(inp * expansion)
        self.residual_connection = (stride == 1 and inp == oup)

        layers = []
        if expansion != 1:
            layers.append(nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])

        if use_se:
            layers.append(SqueezeExcitation(oup, reduction=int(oup * se_ratio), lr_scale=self.lr_scale))
        self.conv = nn.Sequential(*layers)

    def grad_scale(self, x):
        return x * self.lr_scale + x.detach() * (1 - self.lr_scale)

    def forward(self, x):
        if self.residual_connection:
            return x + self.grad_scale(self.conv(x))
        else:
            return self.grad_scale(self.conv(x))

# Definition of the improved combined model
class Network(nn.Module):
    def __init__(self, n_classes=10, width_factor=0.38, lr_scale=1.6):
        super(Network, self).__init__()

        block_settings = [
            # t, c, n, s, use_se
            [2, 16, 1, 1, True],
            [4, 24, 2, 2, True],   
            [8, 34, 2, 2, True],
            [8, 46, 1, 1, True],
        ]
        input_channel = int(24 * width_factor)
        last_channel = int(32 * width_factor)

        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.GELU()
        )]

        for t, c, n, s, use_se in block_settings:
            output_channel = int(c * width_factor)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidualBlock(input_channel, output_channel, stride, expansion=t, use_se=use_se, lr_scale=lr_scale))
                input_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU()
        ))

        self.features = nn.Sequential(*self.features)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# CIFAR-10 Dataset
transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformations)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformations)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Create the model and transfer it to GPU (if available)
model = Network(n_classes=n_classes, width_factor=width_factor).to(device)


# Optimizer and Scheduler
optimizer = optim.SGD(model.parameters(), lr=initial_rate, momentum=0.9, weight_decay=3e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

loss_function = nn.CrossEntropyLoss()

# Training function
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f'Training Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f} '
          f'Accuracy: {100.*correct/total:.2f}%')

# Validation function
def validate(epoch):
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Validation Epoch {epoch} Loss: {valid_loss/total:.6f} '
          f'Accuracy: {100.*correct/total:.2f}%')


# Training loop
for epoch in range(epochs):
    train(epoch)
    validate(epoch)
    scheduler.step()

# Save the model using TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('/workspace/models/improved_model.pt')
print("Improved model saved as 'improved_model.pt'")
