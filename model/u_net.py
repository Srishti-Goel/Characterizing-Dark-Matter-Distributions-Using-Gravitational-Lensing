import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class DownScaling(nn.Module):
    '''Max-pool, convolve 2X with ReLU activation'''
    def __init__(self, in_channels, out_channels, mid_channels = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=mid_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.down(x)

# model = DownScaling(1,5).to(device)
# print(model)

class UpScaling(nn.Module):
    '''Upscale by Transposed Convolution'''
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, 
                                     out_channels=in_channels//2, 
                                     kernel_size=2, 
                                     stride=2)
        mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=mid_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)
    
# model = UpScaling(2,1).to(device)
# print(model)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1)
    def forward(self, x):
        return self.conv(x)

model = OutConv(2,2).to(device=device)
print(model)

class UNet(nn.Module):
    def __init__(self, n_channels) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = 2

        in_channels = n_channels
        mid_channels = 64
        out_channels = 64

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=mid_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.down1 = DownScaling(64,128)
        self.down2 = DownScaling(128, 256)
        self.down3 = DownScaling(256, 512)

        self.down4 = DownScaling(512, 1024)
        
        self.up1 = UpScaling(1024, 512)
        self.up2 = UpScaling(512, 256)
        self.up3 = UpScaling(256, 128)
        self.up4 = UpScaling(128, 64)

        self.outc = OutConv(64, self.n_classes)
        
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

        return self.outc(x)
    
model = UNet(2)
print(model)