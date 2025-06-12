import torch.nn.functional as F
import torch.nn as nn

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                  out_channels = out_channels,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels = out_channels,
                  out_channels = out_channels,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.conv(x)

class AttentionGates(nn.Module):
  def __init__(self, in_channels_x, in_channels_g, inter_channels):
    super(AttentionGates, self).__init__()
    self.W_g = nn.Conv2d(in_channels = in_channels_g,
                         out_channels = inter_channels,
                         kernel_size=1,
                         stride=1,
                         bias=False)
    self.W_x = nn.Conv2d(in_channels = in_channels_x,
                         out_channels = inter_channels,
                         kernel_size=1,
                         stride=1,
                         bias=False)
    self.sigma_1 = nn.ReLU(inplace=True)
    self.phi = nn.Conv2d(in_channels = inter_channels,
                         out_channels = 1,
                         kernel_size=1,
                         stride=1,
                         bias=False)
    self.sigma_2 = nn.Sigmoid()
    
  def forward(self, x, g):
    g = F.interpolate(g, scale_factor=2, mode='bilinear', antialias=False)
    summation = self.W_g(g) + self.W_x(x)
    q_att = self.phi(self.sigma_1(summation))
    attentions_coeff = self.sigma_2(q_att)
    return (x * attentions_coeff)