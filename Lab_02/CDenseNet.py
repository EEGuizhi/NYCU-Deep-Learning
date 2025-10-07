# NYCU IEE Deep Learning Lab 02: Crowd Counting
# BSChen (313510156)
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ Basic Block in Lightweight Dense Block (LDB) """
    def __init__(self, in_channel: int, out_channel: int):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        out = self.conv(out)
        out = self.relu(out)
        return out


class TransitionBlock(nn.Module):
    """ Transition Block between two Lightweight Dense Blocks (LDBs) """
    def __init__(self, in_channel: int, out_channel: int):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

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
        self.inter_channel = 32

        # Initial convolution layer
        self.enc = nn.Sequential(
            nn.Conv2d(in_channel, self.inter_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(inplace=True)
        )

        # Stack LDBs and Transition Blocks
        self.ldb_layers = nn.ModuleList()
        for i in range(self.num_LDBs):
            print(f"Building LDB {i+1:02d}/{self.num_LDBs}")
            self.ldb_layers.append(LDB(self.inter_channel, t))
            self.ldb_layers.append(TransitionBlock(int(self.inter_channel * (1 + 4*t)), self.inter_channel))

        # Final layers
        self.dec = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.inter_channel, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
        )
        print("CDenseNet initialization complete.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        for layer in self.ldb_layers:
            x = layer(x)
        x = self.dec(x)
        return x
