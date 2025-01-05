import torch
from torch import nn
import torch.nn.functional as F
from archs.improvedfastkanconv import ImprovedFastKANConvLayer

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.double_conv = nn.Sequential(
            ImprovedFastKANConvLayer(self.in_channels, self.out_channels // 2, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.out_channels // 2),
            nn.ReLU(inplace=True),
            ImprovedFastKANConvLayer(self.out_channels // 2, self.out_channels, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, device='cuda'):
        super().__init__()
        self.device = device
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, device=self.device)
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

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = ImprovedFastKANConvLayer(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Aemsn(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, device='mps'):
        super(Aemsn, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.device = device

        self.channels = [64, 128, 256, 512, 1024]

        self.inc = (DoubleConv(n_channels, 64, device=self.device))

        self.down1 = (Down(self.channels[0], self.channels[1], self.device))
        self.down2 = (Down(self.channels[1], self.channels[2], self.device))
        self.down3 = (Down(self.channels[2], self.channels[3], self.device))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.channels[3], self.channels[4] // factor, self.device))
        self.up1 = (Up(self.channels[4], self.channels[3] // factor, bilinear, self.device))
        self.up2 = (Up(self.channels[3], self.channels[2] // factor, bilinear, self.device))
        self.up3 = (Up(self.channels[2], self.channels[1] // factor, bilinear, self.device))
        self.up4 = (Up(self.channels[1], self.channels[0], bilinear, self.device))
        self.outc = (OutConv(self.channels[0], n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

