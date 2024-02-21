import torch
import torch.nn as nn
from network_blocks import BaseConv, DWConv, SPPBottleneck, ResLayer, CSPLayer, Focus

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(self,
                 in_channels=3,
                 stem_out_channels=32,
                 depth=53,
                 out_features=["stem", "dark2", "dark3", "dark4", "dark5"]):
      super().__init__()
      assert out_features, "output features of Darknet cannot be empty"
      self.out_features = out_features
      self.stem = nn.Sequential(BaseConv(in_channels, 
                                         stem_out_channels, 
                                         kernel_size=3, 
                                         stride=1, 
                                         act="lrelu"),
                                *self.make_group_layer(stem_out_channels, 
                                                       num_blocks=1, 
                                                       stride=2))
      num_blocks = Darknet.depth2blocks[depth]
      self.dark2 = nn.Sequential(*self.make_group_layer(stem_out_channels*2, 
                                                        num_blocks=num_blocks[0], 
                                                        stride=2))
      self.dark3 = nn.Sequential(*self.make_group_layer(stem_out_channels*4, 
                                                        num_blocks=num_blocks[1], 
                                                        stride=2))
      self.dark4 = nn.Sequential(*self.make_group_layer(stem_out_channels*8, 
                                                        num_blocks=num_blocks[2], 
                                                        stride=2))
      self.dark5 = nn.Sequential(*self.make_group_layer(stem_out_channels*16, 
                                                        num_blocks=num_blocks[3], 
                                                        stride=2),
                                 *self.make_spp_block(stem_out_channels*32, 
                                                      stem_out_channels*16, 
                                                      stem_out_channels*32))

    def make_group_layer(self, in_channels, num_blocks, stride=1):
      return [BaseConv(in_channels, 
                       in_channels*2, 
                       kernel_size=3, 
                       stride=stride, 
                       act="lrelu"),
              *[(ResLayer(in_channels*2)) for _ in range(num_blocks)]]

    def make_spp_block(self, in_channels, out_channels, hidden_channels):
      m = nn.Sequential(*[BaseConv(in_channels, 
                                   out_channels, 
                                   kernel_size=1, 
                                   stride=1, 
                                   act="lrelu"),
                          BaseConv(out_channels, 
                                   hidden_channels, 
                                   kernel_size=3, 
                                   stride=1, 
                                   act="lrelu"),
                          SPPBottleneck(in_channels=hidden_channels,
                                        out_channels=out_channels,
                                        act="lrelu"),
                          BaseConv(out_channels, 
                                   hidden_channels, 
                                   kernel_size=3, 
                                   stride=1, 
                                   act="lrelu"),
                          BaseConv(hidden_channels, 
                                   out_channels, 
                                   kernel_size=1, 
                                   stride=1, 
                                   act="lrelu"),])
      return m
    def forward(self, x):
      # x       [..., in_channels,          height,    width]
      # stem -> [..., 2*stem_out_channels,  height/2,  width/2]
      # dark2-> [..., 4*stem_out_channels,  height/4,  width/4]
      # dark3-> [..., 8*stem_out_channels,  height/8,  width/8]
      # dark4-> [..., 16*stem_out_channels, height/16, width/16]
      # dark5-> [..., 16*stem_out_channels, height/32, width/32]
      outputs = {}
      x = self.stem(x)
      outputs["stem"] = x
      x = self.dark2(x)
      outputs["dark2"] = x
      x = self.dark3(x)
      outputs["dark3"] = x
      x = self.dark4(x)
      outputs["dark4"] = x
      x = self.dark5(x)
      outputs["dark5"] = x
      return {k: v for k, v in outputs.items() if k in self.out_features}

class CSPDarknet(nn.Module):
  def __init__(self, 
               in_channels=3, 
               depth=1.0, 
               width=1.0, 
               out_features=["stem", "dark2", "dark3", "dark4", "dark5"],
               depthwise=False, 
               act="silu",):
    super().__init__()
    assert out_features, "output features of Darknet cannot be empty"
    self.out_features = out_features
    Conv              = DWConv if depthwise else BaseConv
    base_channels     = int(width * 64)
    base_depth        = max(round(depth * 3), 1)
    self.stem         = Focus(in_channels, 
                              base_channels, 
                              kernel_size=3, 
                              act=act)
    self.dark2        = nn.Sequential(Conv(base_channels, 
                                           base_channels*2, 
                                           kernel_size=3, 
                                           stride=2, 
                                           act=act),
                                      CSPLayer(base_channels*2, 
                                               base_channels*2, 
                                               n=base_depth, 
                                               depthwise=depthwise,
                                               act=act))
    self.dark3        = nn.Sequential(Conv(base_channels*2, 
                                           base_channels*4, 
                                           kernel_size=3, 
                                           stride=2, 
                                           act=act),
                                      CSPLayer(base_channels*4, 
                                               base_channels*4, 
                                               n=base_depth*3, 
                                               depthwise=depthwise,
                                               act=act))
    self.dark4        = nn.Sequential(Conv(base_channels*4, 
                                           base_channels*8, 
                                           kernel_size=3, 
                                           stride=2, 
                                           act=act),
                                      CSPLayer(base_channels*8, 
                                               base_channels*8, 
                                               n=base_depth*3, 
                                               depthwise=depthwise,
                                               act=act))
    self.dark5        = nn.Sequential(Conv(base_channels*8, 
                                           base_channels*16, 
                                           kernel_size=3, 
                                           stride=2, 
                                           act=act),
                                      SPPBottleneck(base_channels*16, 
                                                    base_channels*16, 
                                                    act=act),
                                      CSPLayer(base_channels*16, 
                                               base_channels*16, 
                                               n=base_depth, 
                                               shortcut=False, 
                                               depthwise=depthwise,
                                               act=act))
  def forward(self, x):
    # x       [..., in_channels, height,    width]
    # stem -> [..., 64*width,    height/2,  width/2]
    # dark2-> [..., 128*width,   height/4,  width/4]
    # dark3-> [..., 256*width,   height/8,  width/8]
    # dark4-> [..., 512*width,   height/16, width/16]
    # dark5-> [..., 1024*width,  height/32, width/32]
    outputs = {}
    x = self.stem(x)
    outputs["stem"] = x
    x = self.dark2(x)
    outputs["dark2"]= x
    x = self.dark3(x)
    outputs["dark3"]= x
    x = self.dark4(x)
    outputs["dark4"]= x
    x = self.dark5(x)
    outputs["dark5"]= x
    return {k: v for k, v in outputs.items() if k in self.out_features}

if __name__ == '__main__':
  x = torch.ones((2,3,128,128))
  print("x",x.shape)
  darknet = Darknet()
  print("Darknet_stem ",darknet(x)["stem"].shape)
  print("Darknet_dark2",darknet(x)["dark2"].shape)
  print("Darknet_dark3",darknet(x)["dark3"].shape)
  print("Darknet_dark4",darknet(x)["dark4"].shape)
  print("Darknet_dark5",darknet(x)["dark5"].shape)
  x = torch.ones((2,3,128,128))
  cspdarknet = CSPDarknet()
  print("CSPDarknet_stem ",cspdarknet(x)["stem"].shape)
  print("CSPDarknet_dark2",cspdarknet(x)["dark2"].shape)
  print("CSPDarknet_dark3",cspdarknet(x)["dark3"].shape)
  print("CSPDarknet_dark4",cspdarknet(x)["dark4"].shape)
  print("CSPDarknet_dark5",cspdarknet(x)["dark5"].shape)