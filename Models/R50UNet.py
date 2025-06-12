import torchvision
import torch
import torch.nn as nn
from Models.Utils import DoubleConv

class R50UNet(nn.Module):
  def __init__(self, in_channels = 3, out_channels = 32):
    super(R50UNet, self).__init__()

    r50_weights = torchvision.models.ResNet50_Weights.DEFAULT
    resnet50 = torchvision.models.resnet50(weights=r50_weights)
    preconv1 = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )
    maxpool = resnet50.maxpool
    preconv2 = nn.Sequential(
        nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 7,stride = 2,padding = 3, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
    )
    Layer1 = resnet50.layer1
    Layer2 = resnet50.layer2
    Layer3 = resnet50.layer3
    
    preconv3 = nn.Sequential(
        nn.Conv2d(in_channels = 128,out_channels = 64,kernel_size = 3,stride = 1,padding = 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        maxpool
    )
    self.down = nn.ModuleList([preconv1, preconv2, preconv3, Layer1, Layer2, Layer3])
    for down in self.down[-3:]:
      for name, param in down.named_parameters():
        param.requires_grad = False

    self.up = nn.ModuleList()
    shapes = [64, 128, 256, 512]
    for SHAPE in shapes[::-1]:
      self.up.append(nn.ConvTranspose2d(in_channels = SHAPE * 2, out_channels = SHAPE, kernel_size = 2, stride = 2))
      self.up.append(DoubleConv(SHAPE * 2, SHAPE))
    self.FinalConv = nn.Conv2d(in_channels = shapes[0], out_channels= out_channels, kernel_size=1)

  def forward(self, x):
    skip_connections = []
    i = 0
    for down in self.down:
      x = down(x)
      i += 1
      if i == 3 or i == len(self.down):
        continue
      else:
        skip_connections.append(x)

    skip_connections = skip_connections[::-1]

    for idx in range(0,len(self.up), 2):
      x = self.up[idx](x)
      skip_connection = skip_connections[idx // 2]
      x = torch.concat((skip_connection, x), dim=1)
      x = self.up[idx+1](x)

    x = self.FinalConv(x)
    return x