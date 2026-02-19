"""
Training RepVGG-A0 su CIFAR-10.
Usa lo stesso protocollo di training del modello proposto per un confronto equo.

Architettura adattata per CIFAR-10 (input 32x32):
- Primo conv con stride 1 (non 2)
- Rimuove secondo stride 2 per mantenere feature map ragionevoli
"""
import sys
import os
import math
import copy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train'))
from shared_config import *


# ============================================================
# Architettura RepVGG per CIFAR-10
# ============================================================

class RepVGGBlock(nn.Module):
    """
    RepVGG building block.
    Training: 3x3 conv + 1x1 conv + identity (se inp==oup)
    Deploy: singolo 3x3 conv (fusi)
    """
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if deploy:
            self.reparam = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
        else:
            self.bn_identity = nn.BatchNorm2d(in_channels) if (stride == 1 and in_channels == out_channels) else None
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.relu(self.reparam(x))

        out = self.conv3x3(x) + self.conv1x1(x)
        if self.bn_identity is not None:
            out = out + self.bn_identity(x)
        return self.relu(out)

    def _fuse_bn(self, conv, bn):
        """Fonde conv + batch norm in un singolo conv."""
        kernel = conv.weight
        gamma = bn.weight
        beta = bn.bias
        mu = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = (var + eps).sqrt()
        fused_kernel = kernel * (gamma / std).reshape(-1, 1, 1, 1)
        fused_bias = beta - mu * gamma / std
        return fused_kernel, fused_bias

    def _pad_1x1_to_3x3(self, kernel):
        """Pad a 1x1 kernel to 3x3."""
        return F.pad(kernel, [1, 1, 1, 1])

    def _get_identity_kernel_bias(self):
        """Crea kernel identity per il ramo BatchNorm."""
        kernel = torch.zeros(self.in_channels, self.in_channels, 3, 3,
                            device=self.conv3x3[0].weight.device)
        for i in range(self.in_channels):
            kernel[i, i, 1, 1] = 1
        # Fuse with identity BN
        gamma = self.bn_identity.weight
        beta = self.bn_identity.bias
        mu = self.bn_identity.running_mean
        var = self.bn_identity.running_var
        eps = self.bn_identity.eps
        std = (var + eps).sqrt()
        fused_kernel = kernel * (gamma / std).reshape(-1, 1, 1, 1)
        fused_bias = beta - mu * gamma / std
        return fused_kernel, fused_bias

    def switch_to_deploy(self):
        """Fonde i 3 rami in un singolo conv 3x3."""
        kernel3x3, bias3x3 = self._fuse_bn(self.conv3x3[0], self.conv3x3[1])
        kernel1x1, bias1x1 = self._fuse_bn(self.conv1x1[0], self.conv1x1[1])
        kernel1x1 = self._pad_1x1_to_3x3(kernel1x1)

        kernel = kernel3x3 + kernel1x1
        bias = bias3x3 + bias1x1

        if self.bn_identity is not None:
            kernel_id, bias_id = self._get_identity_kernel_bias()
            kernel = kernel + kernel_id
            bias = bias + bias_id

        self.reparam = nn.Conv2d(self.in_channels, self.out_channels, 3,
                                 stride=self.stride, padding=1, bias=True)
        self.reparam.weight.data = kernel
        self.reparam.bias.data = bias

        # Rimuovi rami di training
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        if hasattr(self, 'bn_identity') and self.bn_identity is not None:
            self.__delattr__('bn_identity')
        self.deploy = True


class RepVGG_CIFAR(nn.Module):
    """
    RepVGG-A0 adattato per CIFAR-10.
    A0 config: num_blocks=[2, 4, 14, 1], width=[0.75, 0.75, 0.75, 2.5]
    Adattato: primo conv stride=1, secondo stage stride=1 per CIFAR-10
    """
    def __init__(self, num_classes=10, deploy=False):
        super().__init__()

        # RepVGG-A0 configuration
        num_blocks = [2, 4, 14, 1]
        width_multiplier = [0.75, 0.75, 0.75, 2.5]
        base_channels = [64, 128, 256, 512]

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        # Stage 0: conv iniziale stride=1 per CIFAR-10
        self.stage0 = RepVGGBlock(3, self.in_planes, stride=1, deploy=deploy)

        # Stage 1-4
        self.stage1 = self._make_stage(int(base_channels[0] * width_multiplier[0]),
                                       num_blocks[0], stride=1, deploy=deploy)  # stride 1 per CIFAR
        self.stage2 = self._make_stage(int(base_channels[1] * width_multiplier[1]),
                                       num_blocks[1], stride=2, deploy=deploy)
        self.stage3 = self._make_stage(int(base_channels[2] * width_multiplier[2]),
                                       num_blocks[2], stride=2, deploy=deploy)
        self.stage4 = self._make_stage(int(base_channels[3] * width_multiplier[3]),
                                       num_blocks[3], stride=2, deploy=deploy)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(base_channels[3] * width_multiplier[3]), num_classes)

        if not deploy:
            self._init_weights()

    def _make_stage(self, planes, num_blocks, stride, deploy):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(RepVGGBlock(self.in_planes, planes, stride=s, deploy=deploy))
            self.in_planes = planes
        return nn.Sequential(*blocks)

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

    def deploy(self):
        """Converte tutti i blocchi in modalità deploy."""
        for module in self.modules():
            if isinstance(module, RepVGGBlock) and not module.deploy:
                module.switch_to_deploy()

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    set_seed()
    device = get_device()

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_repvgg_a0')

    setup_logging(run_dir)

    logging.info("=== RepVGG-A0 su CIFAR-10 ===")
    logging.info(f"Protocollo: {EPOCHS} epoche, SGD(lr={LR}, momentum={MOMENTUM}), "
                 f"WD={WEIGHT_DECAY}, CosineAnnealing")

    transform_train, transform_test = get_transforms_standard()
    trainloader, valloader, testloader = get_dataloaders(transform_train, transform_test)

    model = RepVGG_CIFAR(num_classes=NUM_CLASSES).to(device)

    # RepVGG ha deploy mode, come il nostro modello
    stats = train_and_evaluate(model, run_dir, trainloader, valloader, testloader,
                               device, has_deploy=True)

    logging.info(f"\n{'='*60}")
    logging.info(f"RepVGG-A0 - Risultato finale: {stats['test_acc_final']:.2f}%")
    logging.info(f"Parametri (train): {stats['total_params']:,}")
    if 'deploy_params' in stats:
        logging.info(f"Parametri (deploy): {stats['deploy_params']:,}")
    logging.info(f"{'='*60}")
