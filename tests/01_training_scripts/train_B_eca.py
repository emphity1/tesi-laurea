"""
Modello B: MobileNetECA (Baseline + ECA)
Aggiunge Efficient Channel Attention ai blocchi Inverted Residual.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_config import *

# --- Architettura ---

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=3, b=12):
        super(ECABlock, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class InvertedResidualECA(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=True):
        super(InvertedResidualECA, self).__init__()
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

        if use_eca:
            layers.append(ECABlock(hidden_dim))

        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetECA(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, width_mult=WIDTH_MULT):
        super(MobileNetECA, self).__init__()

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
                self.features.append(InvertedResidualECA(input_channel, output_channel, stride, t, use_eca=True))
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

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_B_eca")
    setup_logging(run_dir)

    logging.info("=" * 60)
    logging.info("MODELLO B: MobileNetECA (Baseline + ECA)")
    logging.info("=" * 60)

    transform_train, transform_test = get_transforms_standard()
    trainloader, valloader, testloader = get_dataloaders(transform_train, transform_test)

    model = MobileNetECA().to(device)
    train_and_evaluate(model, run_dir, trainloader, valloader, testloader, device, has_deploy=False)
