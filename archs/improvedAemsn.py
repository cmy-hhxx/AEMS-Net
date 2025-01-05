import torch
from torch import nn
import torch.nn.functional as F
from archs.improvedfastkanconv import ImprovedFastKANConvLayer
from .kan import KANLinear

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FeatureFusionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FeatureFusionAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.se = SEBlock(in_channels, reduction)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            # FastKANConvLayer(in_channels * 2, in_channels, padding=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        se_out = self.se(x)

        # Spatial attention
        q = self.conv1(x)
        k = self.conv2(x)
        v = self.conv3(x)

        attention = self.sigmoid(self.conv4(q * k))
        spatial_out = v * attention

        # Feature fusion
        concat = torch.cat([se_out, spatial_out], dim=1)
        gate = self.gate(concat)

        out = gate * se_out + (1 - gate) * spatial_out
        return out + x  # Residual connection


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.conv1 = ImprovedFastKANConvLayer(in_channels, out_channels // 2, padding=1, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = ImprovedFastKANConvLayer(out_channels // 2, out_channels, padding=1, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ffa = FeatureFusionAttention(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.ffa(out)
        if residual.shape[1] == out.shape[1]:
            out += residual
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, device=device)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, device='cuda'):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, device=device)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, device=device)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = ImprovedFastKANConvLayer(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BrightnessAdaptation(nn.Module):
    def __init__(self, channels):
        super(BrightnessAdaptation, self).__init__()

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]


        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
        #
        # self.fc = KANLinear(
        #     channels,
        #     channels,
        #     grid_size=grid_size,
        #     spline_order=spline_order,
        #     scale_noise=scale_noise,
        #     scale_base=scale_base,
        #     scale_spline=scale_spline,
        #     base_activation=base_activation,
        #     grid_eps=grid_eps,
        #     grid_range=grid_range,
        # )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedAemsn(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, device='cuda'):
        super(ImprovedAemsn, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.device = device

        self.channels = [64, 128, 256, 512, 1024]

        self.inc = DoubleConv(n_channels, 64, device=self.device)
        self.down1 = Down(self.channels[0], self.channels[1], self.device)
        self.down2 = Down(self.channels[1], self.channels[2], self.device)
        self.down3 = Down(self.channels[2], self.channels[3], self.device)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.channels[3], self.channels[4] // factor, self.device)

        self.up1 = Up(self.channels[4], self.channels[3] // factor, bilinear, self.device)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, bilinear, self.device)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, bilinear, self.device)
        self.up4 = Up(self.channels[1], self.channels[0], bilinear, self.device)

        self.outc = OutConv(self.channels[0], n_classes)

        self.brightness_adaptation = BrightnessAdaptation(self.channels[0])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.brightness_adaptation(x)
        logits = self.outc(x)
        return logits