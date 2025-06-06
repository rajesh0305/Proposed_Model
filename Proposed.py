import torch
import torch.nn as nn
import torch.nn.functional as F

class BFEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BFEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation block 
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // 4)
        self.fc2 = nn.Linear(out_channels // 4, out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        
        # SE attention
        w = self.global_pool(out).view(out.size(0), -1)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w)).view(out.size(0), out.size(1), 1, 1)
        out = out * w
        return out

class EasyNet(nn.Module):
    def __init__(self, num_classes=1, base_channels=32):
        super(EasyNet, self).__init__()
        # Encoder
        self.bfem1 = BFEM(3, base_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.bfem2 = BFEM(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bfem3 = BFEM(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.bfem4 = BFEM(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.bfem5 = BFEM(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.bfem6 = BFEM(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.bfem7 = BFEM(base_channels * 2, base_channels)
        
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.bfem1(x)
        x2 = self.bfem2(self.pool1(x1))
        x3 = self.bfem3(self.pool2(x2))
        x4 = self.bfem4(self.pool3(x3))
        
        # Decoder
        d3 = self.up3(x4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.bfem5(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.bfem6(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.bfem7(d1)
        
        out = self.final_conv(d1)
        out = torch.sigmoid(out) 
        return out

if __name__ == "__main__":
    model = EasyNet(num_classes=10)
    x = torch.randn(2, 3, 256, 256)  
    y = model(x)
    print(y.shape)  