"""
Training MobileNetV2 (0.5x e 1.0x) su CIFAR-10.
Usa lo stesso protocollo di training del modello proposto per un confronto equo.

Architettura adattata per CIFAR-10 (input 32x32):
- Primo conv con stride 1 (non 2) dato che le immagini sono 32x32
- Rimuove l'ultimo stride 2 per mantenere feature map ragionevole
"""
import sys
import os
import math

# Aggiungi la cartella train al path per importare shared_config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train'))
from shared_config import *

# ============================================================
# Architettura MobileNetV2 per CIFAR-10
# ============================================================

class InvertedResidual(nn.Module):
    """Inverted Residual Block con depthwise separable convolution."""
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden = int(round(inp * expand_ratio))
        self.use_res = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            # Depthwise
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # Pointwise linear
            nn.Conv2d(hidden, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2_CIFAR(nn.Module):
    """
    MobileNetV2 adattato per CIFAR-10.
    - width_mult: 0.5 o 1.0
    - Primo conv stride=1 (CIFAR-10 è 32x32, non 224x224)
    """
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        # Standard MobileNetV2 block settings: [t, c, n, s]
        # t=expansion, c=output_channels, n=repeat, s=stride
        self.cfgs = [
            [1,  16, 1, 1],
            [6,  24, 2, 1],   # stride 1 per CIFAR (originale: 2)
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # Primo conv: stride=1 per CIFAR-10
        features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )]

        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # Last conv
        features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
        ))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.classifier(x)
        return x


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=float, default=1.0, choices=[0.5, 1.0],
                        help='Width multiplier (0.5 o 1.0)')
    args = parser.parse_args()

    set_seed()
    device = get_device()

    width_tag = f"{args.width:.1f}x".replace('.', '')
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'results_mobilenetv2_{width_tag}')

    setup_logging(run_dir)

    logging.info(f"=== MobileNetV2 ({args.width}x) su CIFAR-10 ===")
    logging.info(f"Protocollo: {EPOCHS} epoche, SGD(lr={LR}, momentum={MOMENTUM}), "
                 f"WD={WEIGHT_DECAY}, CosineAnnealing")

    transform_train, transform_test = get_transforms_standard()
    trainloader, valloader, testloader = get_dataloaders(transform_train, transform_test)

    model = MobileNetV2_CIFAR(num_classes=NUM_CLASSES, width_mult=args.width).to(device)

    stats = train_and_evaluate(model, run_dir, trainloader, valloader, testloader, device)

    logging.info(f"\n{'='*60}")
    logging.info(f"MobileNetV2 ({args.width}x) - Risultato finale: {stats['test_acc_final']:.2f}%")
    logging.info(f"Parametri: {stats['total_params']:,}")
    logging.info(f"{'='*60}")
