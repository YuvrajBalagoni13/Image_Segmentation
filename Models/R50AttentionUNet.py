import torch
import torchvision
import torch.nn as nn
from Models.Utils import DoubleConv, AttentionGates

class R50_AttentionUNet(nn.Module):
  def __init__(self, in_channels = 3, out_channels = 32):
    super(R50_AttentionUNet, self).__init__()

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
    self.AGs = nn.ModuleList()
    shapes = [64, 128, 256, 512]
    for SHAPE in shapes[::-1]:
      self.up.append(nn.ConvTranspose2d(in_channels = SHAPE * 2, out_channels = SHAPE, kernel_size = 2, stride = 2))
      self.up.append(DoubleConv(SHAPE * 2, SHAPE))
      self.AGs.append(AttentionGates(SHAPE, SHAPE * 2, SHAPE // 2))
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

      skip_connection = skip_connections[idx // 2]
      AG_Output = self.AGs[idx // 2](skip_connection,x)
      x = self.up[idx](x)
      x = torch.concat((AG_Output, x), dim=1)
      x = self.up[idx+1](x)

    x = self.FinalConv(x)
    return x