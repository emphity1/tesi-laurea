"""
Modello D: MobileNetECA-Rep-AdvAug (ECA + Rep + Advanced Augmentation)
Stessa architettura di C, ma con AutoAugment + RandomErasing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_config import *

# --- Architettura (identica a C) ---

class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = nn.GELU()

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True)
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):
        if hasattr(self, 'reparam_conv'):
            return self.activation(self.reparam_conv(inputs))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        return self.activation(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_conv'):
            return
        k3, b3 = self._get_equivalent(self.rbr_dense)
        k1, b1 = self._get_equivalent(self.rbr_1x1)
        kid, bid = self._get_equivalent(self.rbr_identity)
        k1 = F.pad(k1, [1, 1, 1, 1])

        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True)
        self.reparam_conv.weight.data = k3 + k1 + kid
        self.reparam_conv.bias.data = b3 + b1 + bid

        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        self.deploy = True

    def _get_equivalent(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean, running_var = branch[1].running_mean, branch[1].running_var
            gamma, beta, eps = branch[1].weight, branch[1].bias, branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=branch.weight.dtype, device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            if self.id_tensor.device != branch.weight.device:
                self.id_tensor = self.id_tensor.to(branch.weight.device)
            kernel = self.id_tensor
            running_mean, running_var = branch.running_mean, branch.running_var
            gamma, beta, eps = branch.weight, branch.bias, branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


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


class RepInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=True):
        super(RepInvertedResidual, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ])

        layers.append(RepConv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim))

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


class MobileNetECARep(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, width_mult=WIDTH_MULT):
        super(MobileNetECARep, self).__init__()

        input_channel = max(int(32 * width_mult), MIN_CHANNELS)
        last_channel = max(int(144 * width_mult), MIN_CHANNELS)

        self.features = [RepConv(3, input_channel, stride=1)]

        for t, c, n, s in BLOCK_SETTINGS:
            output_channel = max(int(c * width_mult), MIN_CHANNELS)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(RepInvertedResidual(input_channel, output_channel, stride, t))
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

    def deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()


if __name__ == "__main__":
    set_seed()
    device = get_device()

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_D_eca_rep_advaug")
    setup_logging(run_dir)

    logging.info("=" * 60)
    logging.info("MODELLO D: MobileNetECA-Rep-AdvAug (Final)")
    logging.info("=" * 60)

    # *** Unica differenza con C: trasformazioni avanzate ***
    transform_train, transform_test = get_transforms_advanced()
    trainloader, valloader, testloader = get_dataloaders(transform_train, transform_test)

    model = MobileNetECARep().to(device)
    train_and_evaluate(model, run_dir, trainloader, valloader, testloader, device, has_deploy=True)
