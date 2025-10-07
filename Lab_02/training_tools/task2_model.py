# NYCU IEE Deep Learning Lab 02: Crowd Counting
# BSChen (313510156)
import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    """ Basic Block in Lightweight Dense Block (LDB) """
    def __init__(self, in_channel: int, out_channel: int, stride: int=1):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        out = self.conv(out)
        out = self.relu(out)
        return out


class TransitionBlock(nn.Module):
    """ Transition Block between two Lightweight Dense Blocks (LDBs) """
    def __init__(self, in_channel: int, out_channel: int, stride: int=1):
        super(TransitionBlock, self).__init__()
        if stride == 1:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        else:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class LDB(nn.Module):
    """ Lightweight Dense Block (LDB) """
    def __init__(self, in_channel: int, t: float = 0.5):
        super(LDB, self).__init__()
        self.inter_channel = int(in_channel * t)
        self.conv_1 = nn.Conv2d(in_channel, self.inter_channel, kernel_size=1)
        self.block_1 = BasicBlock(self.inter_channel, self.inter_channel)
        self.block_2 = BasicBlock(self.inter_channel, self.inter_channel)
        self.block_3 = BasicBlock(self.inter_channel, self.inter_channel)
        self.block_4 = BasicBlock(self.inter_channel, self.inter_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_0 = self.conv_1(x)
        out_1 = self.block_1(out_0)
        out_2 = self.block_2(out_1)
        out_3 = self.block_3(out_1 + out_2)
        out_4 = self.block_4(out_1 + out_2 + out_3)
        return torch.cat([x, out_1, out_2, out_3, out_4], dim=1)


class CDenseNet(nn.Module):
    def __init__(self, n: int = 16, t: float = 0.5, in_channel: int = 1, out_dim: int = 3):
        super(CDenseNet, self).__init__()
        print(f"Initializing CDenseNet with {n} LDBs, growth rate t={t}, input channels={in_channel}, output dim={out_dim}")
        # Parameters
        self.num_LDBs = n
        self.growth_rate = t
        self.in_channel = in_channel
        self.out_dim = out_dim
        self.inter_channel = 64

        # Initial convolution layer
        self.enc = nn.Sequential(
            nn.Conv2d(in_channel, self.inter_channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.inter_channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.inter_channel),
            nn.SiLU(inplace=True)
        )

        # Stack LDBs and Transition Blocks
        self.ldb_layers = nn.ModuleList()
        # stride_layers = 5
        for i in range(self.num_LDBs):
            print(f"Building LDB {i+1:02d}/{self.num_LDBs}")
            self.ldb_layers.append(LDB(self.inter_channel, t))
            self.ldb_layers.append(TransitionBlock(int(self.inter_channel * (1 + 4*t)), self.inter_channel))
            # self.ldb_layers.append(TransitionBlock(
            #     int(self.inter_channel * (1 + 4*t)), self.inter_channel,
            #     # stride=2 if i >= self.num_LDBs - stride_two_layers else 1
            #     stride=2 if (i + 1) % stride_layers == 0 and (i + 1) != self.num_LDBs else 1
            # ))

        # Final layers
        self.dec = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.inter_channel, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, out_dim)
        )
        print("CDenseNet initialization complete.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        for layer in self.ldb_layers:
            x = layer(x)
        x = self.dec(x)
        return x


# ------------------------------- Not Used ------------------------------- #

class DoubleConv(nn.Module):
    """ Double Convolution Block """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int=None):
        super(DoubleConv, self).__init__()
        # Define middle channels
        if not mid_channels:
            mid_channels = out_channels

        # Define the double conv layers
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """ Downscaling with maxpool then double conv """
    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_pool(x)


class UpBlock(nn.Module):
    """ Upscaling then double conv """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool=True):
        super(UpBlock, self).__init__()
        if bilinear:  # use normal convolution to reduce the number of channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2
        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 158x238 -> 158x238
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  # 158x238 -> 79x119
            # nn.BatchNorm2d(in_channels),
            # nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int=1, n_classes: int=3, bilinear: bool=True):
        super(UNet, self).__init__()
        # Parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Heatmap storage
        self.heatmaps = None

        # Layers
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.down1 = DownBlock( 64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024 // factor)

        self.up1 = UpBlock(1024, 512 // factor, bilinear)
        self.up2 = UpBlock( 512, 256 // factor, bilinear)
        self.up3 = UpBlock( 256, 128 // factor, bilinear)
        self.up4 = UpBlock( 128,  64, bilinear)

        self.outc = OutConv(64, n_classes)
        # self.dec = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(40*60*n_classes, 256),
        #     nn.SiLU(inplace=True),
        #     nn.Linear(256, n_classes)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2( x, x3)
        x = self.up3( x, x2)
        x = self.up4( x, x1)
        x = self.outc(x)

        # Normalize output
        x = torch.sigmoid(x)
        self.heatmaps = x.detach().cpu()  # Store heatmaps for later use

        # Fully connected layers
        logits_1 = x.sum(dim=(2, 3))
        # logits_2 = self.dec(x)
        return logits_1, x  # return logits, heatmaps

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """ Get intermediate feature maps (heatmaps) """
        if self.heatmaps is None:
            self.forward(x)
        return self.heatmaps

