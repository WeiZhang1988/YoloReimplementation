import torch
import torch.nn as nn

class SiLU(nn.Module):
  @staticmethod
  def forward(x):
    return x * torch.sigmoid(x)

def get_activation(name="silu",inplace=True):
  if name == "silu":
    module = SiLU()
  elif name == "relu":
    module = nn.ReLU(inplace=inplace)
  elif name == "lrelu":
    module = nn.LeakyReLU(0.1,inplace=inplace)
  else:
    raise AttributeError(f"unsupported act type: {name}")
  return module

class BaseConv(nn.Module):
  def __init__(self, 
               in_channels, 
               out_channels, 
               kernel_size, 
               stride, 
               groups=1, 
               bias=False, 
               act="silu"):
    super().__init__()
    padding   = (kernel_size - 1) // 2
    self.conv = nn.Conv2d(in_channels, 
                          out_channels, 
                          kernel_size=kernel_size, 
                          stride=stride, 
                          padding=padding, 
                          groups=groups, 
                          bias=bias)
    self.bn   = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
    self.act  = get_activation(act, inplace=True)
  def forward(self, x):
    # x  [..., in_channels,  height,                                               width]
    # -> [..., out_channels, (height+2*((kernel_size-1)//2)-kernel_size)/stride+1, (width+2*((kernel_size-1)//2)-kernel_size)/stride+1]
    return self.act(self.bn(self.conv(x)))
  def fuseforward(self, x):
    # x  [..., in_channels,  height,  width]
    # -> [..., out_channels, (height+2*((kernel_size-1)//2)-kernel_size)/stride+1, (width+2*((kernel_size-1)//2)-kernel_size)/stride+1]
    return self.act(self.conv(x))

class DWConv(nn.Module):
  def __init__(self, 
               in_channels, 
               out_channels, 
               kernel_size, 
               stride=1, 
               act="silu"):
    super().__init__()
    self.dconv = BaseConv(in_channels, 
                          in_channels, 
                          kernel_size=kernel_size, 
                          stride=stride, 
                          groups=in_channels, 
                          act=act)
    self.pconv = BaseConv(in_channels, 
                          out_channels, 
                          kernel_size=1, 
                          stride=1, 
                          groups=1, 
                          act=act)
  def forward(self, x):
    # x  [..., in_channels,  height,   width]
    # -> [..., in_channels,  (height+2*((kernel_size-1)//2)-kernel_size)/stride+1, (width+2*((kernel_size-1)//2)-kernel_size)/stride+1]
    # -> [..., out_channels, (height+2*((kernel_size-1)//2)-kernel_size)/stride+1, (width+2*((kernel_size-1)//2)-kernel_size)/stride+1]
    x = self.dconv(x)
    return self.pconv(x)

class Bottleneck(nn.Module):
  def __init__(self, 
               in_channels, 
               out_channels, 
               shortcut=True, 
               expansion=0.5, 
               depthwise=False, 
               act="silu"):
    super().__init__()
    hidden_channels = int(out_channels*expansion)
    Conv            = DWConv if depthwise else BaseConv
    self.conv1      = BaseConv(in_channels, 
                               hidden_channels, 
                               kernel_size=1, 
                               stride=1, 
                               act=act)
    self.conv2      = Conv(hidden_channels, 
                           out_channels, 
                           kernel_size=3, 
                           stride=1, 
                           act=act)
    self.use_add    = shortcut and in_channels == out_channels
  def forward(self, x):
    # x  [..., in_channels,     height,   width]
    # -> [..., hidden_channels, height,   width]
    # -> [..., out_channels,    height,   width] or if use_add [..., in_channels,     height,   width]
    y = self.conv2(self.conv1(x))
    if self.use_add:
      y = y + x
    return y

class ResLayer(nn.Module):
  def __init__(self, in_channels: int):
    super().__init__()
    mid_channels = in_channels // 2
    self.layer1 = BaseConv(in_channels, 
                           mid_channels, 
                           kernel_size=1, 
                           stride=1, 
                           act="lrelu")
    self.layer2 = BaseConv(mid_channels, 
                           in_channels, 
                           kernel_size=3, 
                           stride=1, 
                           act="lrelu")
  def forward(self, x):
    # x      [..., in_channels,  height, width]
    # -----> [..., mid_channels, height, width]
    # out -> [..., in_channels,  height, width]
    out = self.layer2(self.layer1(x))
    return x + out

class SPPBottleneck(nn.Module):
  def __init__(self, 
               in_channels, 
               out_channels, 
               kernel_sizes=[5, 9, 13], 
               act="silu"):
    super().__init__()
    for kernel_size in kernel_sizes:
      assert 2 * (kernel_size // 2) < kernel_size, "kernel size must be odd"
    hidden_channels = in_channels // 2
    self.conv1      = BaseConv(in_channels, 
                               hidden_channels, 
                               kernel_size=1, 
                               stride=1, 
                               act=act)
    self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=kernel_size, 
                                                  stride=1, 
                                                  padding=kernel_size // 2) 
                                                  for kernel_size in kernel_sizes])
    conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
    self.conv2      = BaseConv(conv2_channels, 
                               out_channels, 
                               kernel_size=1, 
                               stride=1, 
                               act=act)
  def forward(self, x):
    # x  [..., in_channels,                           height, width]
    # -> [..., hidden_channels,                       height, width]
    # -> [..., hidden_channels*(len(kernel_sizes)+1), height, width]
    # -> [..., out_channels,                          height, width]
    x = self.conv1(x)
    x = torch.cat([x] + [m(x) for m in self.m], dim=-3)
    x = self.conv2(x)
    return x

class CSPLayer(nn.Module):
  def __init__(self, 
               in_channels, 
               out_channels, 
               n=1, 
               shortcut=True, 
               expansion=0.5, 
               depthwise=False, 
               act="silu"):
    super().__init__()
    hidden_channels = int(out_channels*expansion)
    self.conv1      = BaseConv(in_channels, 
                               hidden_channels, 
                               kernel_size=1, 
                               stride=1, 
                               act=act)
    self.conv2      = BaseConv(in_channels, 
                               hidden_channels, 
                               kernel_size=1, 
                               stride=1, 
                               act=act)
    self.conv3      = BaseConv(2*hidden_channels, 
                               out_channels, 
                               kernel_size=1, 
                               stride=1, 
                               act=act)
    module_list     = [Bottleneck(hidden_channels, 
                                  hidden_channels, 
                                  shortcut, 
                                  expansion=1.0, 
                                  depthwise=depthwise, 
                                  act=act) for _ in range(n)]
    self.m          = nn.Sequential(*module_list)
  def forward(self, x):
    # x      [..., in_channels,        height, width]
    # x_1--> [..., hidden_channels,    height, width]
    # x_1--> [..., hidden_channels,    height, width]
    # x_2--> [..., hidden_channels,    height, width]
    # x----> [..., 2*hidden_channels,  height, width]
    # -----> [..., out_channels,       height, width]
    x_1 = self.conv1(x)
    x_1 = self.m(x_1)
    x_2 = self.conv2(x)
    x=torch.cat([x_1,x_2],dim=-3)
    return self.conv3(x)

class Focus(nn.Module):
  def __init__(self, 
               in_channels, 
               out_channels, 
               kernel_size=1, 
               stride=1, 
               act="silu"):
    super().__init__()
    self.conv = BaseConv(in_channels*4, 
                         out_channels, 
                         kernel_size, 
                         stride, 
                         act=act)
  def forward(self, x):
    # x                 [..., in_channels,   height, width]
    # patch_top_left--> [..., in_channels,   height/2, width/2]
    # patch_bot_left--> [..., in_channels,   height/2, width/2]
    # patch_top_right-> [..., in_channels,   height/2, width/2]
    # patch_bot_right-> [..., in_channels,   height/2, width/2]
    # x---------------> [..., 4*in_channels, height/2, width/2]
    # ----------------> [..., out_channels,  (height/2+2*((kernel_size-1)//2)-kernel_size)/stride+1, (width/2+2*((kernel_size-1)//2)-kernel_size)/stride+1]
    patch_top_left  = x[...,  ::2,  ::2]
    patch_bot_left  = x[..., 1::2,  ::2]
    patch_top_right = x[...,  ::2, 1::2]
    patch_bot_right = x[..., 1::2, 1::2]
    x = torch.cat([patch_top_left, 
                   patch_bot_left, 
                   patch_top_right, 
                   patch_bot_right], dim=-3)
    return self.conv(x)