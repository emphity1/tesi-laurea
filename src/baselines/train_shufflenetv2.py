"""
Training ShuffleNetV2 (0.5x) su CIFAR-10.
Usa lo stesso protocollo di training del modello proposto per un confronto equo.

Architettura adattata per CIFAR-10 (input 32x32):
- Primo conv con stride 1 (non 2)
- Primo maxpool rimosso
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train'))
from shared_config import *


# ============================================================
# Architettura ShuffleNetV2 per CIFAR-10
# ============================================================

def channel_shuffle(x, groups):
    """Channel shuffle operation."""
    b, c, h, w = x.shape
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class ShuffleV2Block(nn.Module):
    """ShuffleNet V2 basic unit."""
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.stride = stride
        branch_features = oup // 2

        if stride == 2:
            # Both branches are used
            self.branch1 = nn.Sequential(
                # Depthwise
                nn.Conv2d(inp, inp, 3, stride=2, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # Pointwise
                nn.Conv2d(inp, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, stride=2, padding=1, groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = None
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_features, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, stride=1, padding=1, groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        else:
            # Channel split
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2_CIFAR(nn.Module):
    """
    ShuffleNetV2 adattato per CIFAR-10.
    - width_mult=0.5: channels = [24, 48, 96, 192, 1024]
    """
    def __init__(self, num_classes=10, width_mult=0.5):
        super().__init__()

        if width_mult == 0.5:
            stages_out = [48, 96, 192]
            output_channels = 1024
        elif width_mult == 1.0:
            stages_out = [116, 232, 464]
            output_channels = 1024
        else:
            raise ValueError(f"Unsupported width_mult: {width_mult}")

        stages_repeats = [4, 8, 4]
        input_channels = 24

        # Primo conv: stride=1 per CIFAR-10 (no maxpool)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        for out_c, repeats in zip(stages_out, stages_repeats):
            stage = []
            stage.append(ShuffleV2Block(input_channels, out_c, stride=2))
            for _ in range(repeats - 1):
                stage.append(ShuffleV2Block(out_c, out_c, stride=1))
            self.stages.append(nn.Sequential(*stage))
            input_channels = out_c

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(output_channels, num_classes)

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
        x = self.conv1(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv_last(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.fc(x)
        return x


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    set_seed()
    device = get_device()

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_shufflenetv2_05x')

    setup_logging(run_dir)

    logging.info("=== ShuffleNetV2 (0.5x) su CIFAR-10 ===")
    logging.info(f"Protocollo: {EPOCHS} epoche, SGD(lr={LR}, momentum={MOMENTUM}), "
                 f"WD={WEIGHT_DECAY}, CosineAnnealing")

    transform_train, transform_test = get_transforms_standard()
    trainloader, valloader, testloader = get_dataloaders(transform_train, transform_test)

    model = ShuffleNetV2_CIFAR(num_classes=NUM_CLASSES, width_mult=0.5).to(device)

    stats = train_and_evaluate(model, run_dir, trainloader, valloader, testloader, device)

    logging.info(f"\n{'='*60}")
    logging.info(f"ShuffleNetV2 (0.5x) - Risultato finale: {stats['test_acc_final']:.2f}%")
    logging.info(f"Parametri: {stats['total_params']:,}")
    logging.info(f"{'='*60}")
