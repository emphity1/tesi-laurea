"""
MobileNetECA Architecture Definition
Modular implementation for CIFAR-10 classification with ECA attention
"""

import torch
import torch.nn as nn
import math


class ECABlock(nn.Module):
    """
    Efficient Channel Attention Block
    
    Lightweight attention mechanism that recalibrates channel importance
    with minimal computational overhead (~100 parameters per block).
    
    Advantages over SE-Net:
    - Uses 1D Conv instead of two FC layers (fewer parameters)
    - Adaptive kernel size based on channel count
    - Better efficiency-accuracy trade-off
    """
    
    def __init__(self, channels, gamma=3, b=12, lr_scale=1.6):
        """
        Args:
            channels: Number of input channels
            gamma: Parameter for adaptive kernel size formula
            b: Bias for kernel size formula
            lr_scale: Gradient scaling factor for training stability
        """
        super(ECABlock, self).__init__()
        self.lr_scale = lr_scale
        
        # Adaptive kernel size: k = |log2(C) + b| / gamma
        # More channels → larger kernel to capture long-range dependencies
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1  # Enforce odd kernel for symmetric padding
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling: [B, C, H, W] → [B, C, 1, 1]
        y = self.avg_pool(x)
        
        # Reshape for 1D conv: [B, C, 1, 1] → [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)
        
        # 1D convolution for cross-channel interaction
        y = self.conv(y)
        
        # Sigmoid activation for attention weights [0, 1]
        y = self.sigmoid(y)
        
        # Reshape back: [B, 1, C] → [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        
        # Gradient scaling for training stability
        y = y * self.lr_scale + y.detach() * (1 - self.lr_scale)
        
        # Apply channel-wise attention weights
        return x * y.expand_as(x)


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (Mobile Inverted Bottleneck)
    
    Architecture: narrow → wide → narrow (e.g., 24 → 144 → 24)
    This is opposite to standard ResNet bottleneck (wide → narrow → wide)
    
    Structure:
    1. Expansion: 1x1 conv to increase channels (if expand_ratio > 1)
    2. Depthwise: 3x3 depthwise conv (efficient spatial filtering)
    3. Projection: 1x1 conv to reduce channels
    4. Skip connection: added if input/output dimensions match
    """
    
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=False, lr_scale=1.6):
        """
        Args:
            inp: Input channels
            oup: Output channels
            stride: Stride for depthwise conv (1=maintain, 2=halve spatial dims)
            expand_ratio: Channel expansion factor (e.g., 6 = 6x expansion)
            use_eca: Whether to add ECA attention after depthwise conv
            lr_scale: Gradient scaling factor
        """
        super(InvertedResidual, self).__init__()
        self.lr_scale = lr_scale
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()  # GELU for smoother gradients vs ReLU
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        ])

        # Optional ECA attention
        if use_eca:
            layers.append(ECABlock(hidden_dim, lr_scale=self.lr_scale))

        # Projection phase (linear bottleneck - no activation)
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = out * self.lr_scale + out.detach() * (1 - self.lr_scale)
        
        if self.use_res_connect:
            return x + out
        else:
            return out


class MobileNetECA(nn.Module):
    """
    MobileNetECA: Efficient CNN with Inverted Residuals and ECA Attention
    
    Designed for CIFAR-10 (32x32 RGB images, 10 classes)
    Combines efficiency of MobileNetV2 with lightweight ECA attention
    """
    
    def __init__(self, num_classes=10, width_mult=0.42, lr_scale=1.54):
        """
        Args:
            num_classes: Number of output classes (10 for CIFAR-10)
            width_mult: Width multiplier for scaling model capacity
            lr_scale: Gradient scaling factor for training stability
        """
        super(MobileNetECA, self).__init__()

        # Block configuration: [expand_ratio, out_channels, num_blocks, stride, use_eca]
        block_settings = [
            [1, 20, 2, 1, True],   # Block 1: no expansion
            [6, 32, 4, 2, True],   # Block 2: 6x expansion, downsample
            [8, 42, 4, 2, True],   # Block 3: 8x expansion, downsample
            [8, 52, 2, 1, True],   # Block 4: 8x expansion
        ]
        
        input_channel = max(int(32 * width_mult), 8)
        last_channel = max(int(144 * width_mult), 8)

        # Stem layer
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.GELU()
        )]

        # Build inverted residual blocks
        for t, c, n, s, use_eca in block_settings:
            output_channel = max(int(c * width_mult), 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, 
                                   expand_ratio=t, use_eca=use_eca, lr_scale=lr_scale)
                )
                input_channel = output_channel

        # Final conv + pooling
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        ))

        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Kaiming initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_number(num):
    """Format numbers with K/M suffixes"""
    if abs(num) >= 1_000_000:
        return f'{num / 1_000_000:.2f}M'
    elif abs(num) >= 1_000:
        return f'{num / 1_000:.2f}K'
    else:
        return str(num)


if __name__ == "__main__":
    # Test model instantiation
    model = MobileNetECA(num_classes=10, width_mult=0.42, lr_scale=1.54)
    
    # Test forward pass
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    print(f"MobileNetECA Model Summary:")
    print(f"  Total parameters: {format_number(total_params)}")
    print(f"  Trainable parameters: {format_number(trainable_params)}")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
