"""
Modello A: MobileNetV2-Micro Baseline (senza ECA, senza Rep)
Architettura pura con Inverted Residual Blocks + GELU.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_config import *

# --- Architettura ---

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        ])

        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetBaseline(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, width_mult=WIDTH_MULT):
        super(MobileNetBaseline, self).__init__()

        input_channel = max(int(32 * width_mult), MIN_CHANNELS)
        last_channel = max(int(144 * width_mult), MIN_CHANNELS)

        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.GELU()
        )]

        for t, c, n, s in BLOCK_SETTINGS:
            output_channel = max(int(c * width_mult), MIN_CHANNELS)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        ))

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(last_channel, num_classes))

        initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    set_seed()
    device = get_device()

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_A_baseline")
    setup_logging(run_dir)

    logging.info("=" * 60)
    logging.info("MODELLO A: MobileNetV2-Micro Baseline")
    logging.info("=" * 60)

    transform_train, transform_test = get_transforms_standard()
    trainloader, valloader, testloader = get_dataloaders(transform_train, transform_test)

    model = MobileNetBaseline().to(device)
    train_and_evaluate(model, run_dir, trainloader, valloader, testloader, device, has_deploy=False)
