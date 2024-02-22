import torch
import torch.nn as nn
from darknet import Darknet, CSPDarknet
from network_blocks import BaseConv, DWConv, CSPLayer

class FPN(nn.Module):
  def __init__(self, depth=53, in_features=["dark3", "dark4", "dark5"]):
    super().__init__()
    self.backbone = Darknet(depth=depth)
    self.in_features = in_features
    self.out1_cbl = self._make_cbl(512, 256, 1)
    self.out1 = self._make_embedding(512 + 256, 256, 512)
    self.out2_cbl = self._make_cbl(256, 128, 1)
    self.out2 = self._make_embedding(256 + 128, 128, 256)
    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
  def _make_cbl(self, 
                in_channels, 
                out_channels, 
                kernel_size):
    return BaseConv(in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    act="lrelu")
  def _make_embedding(self, in_channels, 
                            out_channels, 
                            hidden_channels):
    m = nn.Sequential(*[self._make_cbl(in_channels, out_channels, 1),
                        self._make_cbl(out_channels, hidden_channels, 3),
                        self._make_cbl(hidden_channels, out_channels, 1),
                        self._make_cbl(out_channels, hidden_channels, 3),
                        self._make_cbl(hidden_channels, out_channels, 1),])
    return m
  def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
    with open(filename, "rb") as f:
      state_dict = torch.load(f, map_location="cpu")
    print("loading pretrained weights...")
    self.backbone.load_state_dict(state_dict)
  def forward(self, x):
    # x                       [..., in_channels,                height,    width]
    # out_features-stem ->    [..., 2*32(==stem_out_channels),  height/2,  width/2]
    # out_features-dark2->    [..., 4*32(==stem_out_channels),  height/4,  width/4]
    # out_features-dark3-x2-> [..., 8*32(==stem_out_channels),  height/8,  width/8]
    # out_features-dark4-x1-> [..., 16*32(==stem_out_channels), height/16, width/16]
    # out_features-dark5-x0-> [..., 16*32(==stem_out_channels), height/32, width/32]
    # x1_in ----------------> [..., 256,                        height/32, width/32]
    # ----------------------> [..., 256,                        height/16, width/16]
    # ----------------------> [..., 256+512,                    height/16, width/16]
    # out_dark4 ------------> [..., 256,                        height/16, width/16]
    # x2_in ----------------> [..., 128,                        height/16, width/16]
    # ----------------------> [..., 128,                        height/8,  width/8]
    # ----------------------> [..., 128+256,                    height/8,  width/8]
    # out_dark3 ------------> [..., 128,                        height/8,  width/8]
    # outputs-out_dark3->     [..., 128,                        height/8,  width/8]
    # outputs-out_dark4->     [..., 256,                        height/16, width/16]
    # outputs-x0------->      [..., 512,                        height/32, width/32]
    out_features = self.backbone(x)
    x2, x1, x0 = [out_features[f] for f in self.in_features]
    x1_in = self.out1_cbl(x0)
    x1_in = self.upsample(x1_in)
    x1_in = torch.cat([x1_in, x1], dim=-3)
    out_dark4 = self.out1(x1_in)
    x2_in = self.out2_cbl(out_dark4)
    x2_in = self.upsample(x2_in)
    x2_in = torch.cat([x2_in, x2], dim=-3)
    out_dark3 = self.out2(x2_in)
    outputs = [out_dark3, out_dark4, x0]
    return outputs

class PAFPN(nn.Module):
  def __init__(self, 
               in_channels=3,
               depth=1.0, 
               width=1.0, 
               in_features=["dark3","dark4","dark5"], 
               in_channelses=[256, 512, 1024], 
               depthwise=False, 
               act="silu"):
    super().__init__()
    self.backbone       = CSPDarknet(in_channels,
                                     depth=depth, 
                                     width=width, 
                                     depthwise=depthwise, 
                                     act=act)
    self.in_features    = in_features
    Conv                = DWConv if depthwise else BaseConv
    self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")
    self.lateral_conv0  = BaseConv(int(in_channelses[2]*width), 
                                   int(in_channelses[1]*width), 
                                   kernel_size=1, 
                                   stride=1, 
                                   act=act)
    self.C3_p4          = CSPLayer(int(2*in_channelses[1]*width),
                                   int(in_channelses[1]*width),
                                   n=round(3*depth),
                                   shortcut=False,
                                   depthwise=depthwise,
                                   act=act)
    self.reduce_conv1   = BaseConv(int(in_channelses[1]*width), 
                                   int(in_channelses[0]*width), 
                                   kernel_size=1, 
                                   stride=1, 
                                   act=act)
    self.C3_p3          = CSPLayer(int(2*in_channelses[0]*width),
                                   int(in_channelses[0]*width),
                                   n=round(3*depth),
                                   shortcut=False,
                                   depthwise=depthwise,
                                   act=act)
    self.bu_conv2       = Conv(int(in_channelses[0]*width), 
                               int(in_channelses[0]*width), 
                               kernel_size=3, 
                               stride=2, 
                               act=act)
    self.C3_n3          = CSPLayer(int(2*in_channelses[0]*width),
                                   int(in_channelses[1]*width),
                                   n=round(3*depth),
                                   shortcut=False,
                                   depthwise=depthwise,
                                   act=act)
    self.bu_conv1       = Conv(int(in_channelses[1]*width), 
                               int(in_channelses[1]*width), 
                               kernel_size=3, 
                               stride=2, 
                               act=act)
    self.C3_n4          = CSPLayer(int(2*in_channelses[1]*width),
                                   int(in_channelses[2]*width),
                                   n=round(3*depth),
                                   shortcut=False,
                                   depthwise=depthwise,
                                   act=act)
  def forward(self, x):
    # x                       [..., in_channels,                     height,    width]
    # out_features-stem ->    [..., 64*width,                        height/2,  width/2]
    # out_features-dark2->    [..., 128*width,                       height/4,  width/4]
    # out_features-dark3-x2-> [..., 256(==in_channelses[0])*width,   height/8,  width/8]
    # out_features-dark4-x1-> [..., 512(==in_channelses[1])*width,   height/16, width/16]
    # out_features-dark5-x0-> [..., 1024(==in_channelses[2])*width,  height/32, width/32]
    # fpn_out0->              [..., in_channelses[1]*width,          height/32, width/32]
    # f_out0--->              [..., in_channelses[1]*width,          height/16, width/16]
    # --------->              [..., 2*in_channelses[1]*width,        height/16, width/16]
    # --------->              [..., in_channelses[1]*width,          height/16, width/16]
    # fpn_out1->              [..., in_channelses[0]*width,          height/32, width/32]
    # f_out1--->              [..., in_channelses[0]*width,          height/8,  width/8]
    # --------->              [..., 2*in_channelses[0]*width,        height/8,  width/8]
    # pan_out2->              [..., in_channelses[0]*width,          height/8,  width/8]
    # p_out1--->              [..., in_channelses[0]*width,          height/16, width/16]
    # --------->              [..., 2*in_channelses[0]*width,        height/16, width/16]
    # pan_out1->              [..., in_channelses[1]*width,          height/16, width/16]
    # p_out0--->              [..., in_channelses[1]*width,          height/32, width/32]
    # pan_out0->              [..., in_channelses[2]*width,          height/32, width/32]
    # outputs-pan_out2->      [..., in_channelses[0]*width,          height/8,  width/8]
    # outputs-pan_out1->      [..., in_channelses[1]*width,          height/16, width/16]
    # outputs-pan_out0->      [..., in_channelses[2]*width,          height/32, width/32]
    out_features = self.backbone(x)
    features = [out_features[f] for f in self.in_features]
    [x2, x1, x0] = features
    fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
    f_out0 = self.upsample(fpn_out0)  # 512/16
    f_out0 = torch.cat([f_out0, x1], dim=-3)  # 512->1024/16
    f_out0 = self.C3_p4(f_out0)  # 1024->512/16
    fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
    f_out1 = self.upsample(fpn_out1)  # 256/8
    f_out1 = torch.cat([f_out1, x2], dim=-3)  # 256->512/8
    pan_out2 = self.C3_p3(f_out1)  # 512->256/8
    p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
    p_out1 = torch.cat([p_out1, fpn_out1], dim=-3)  # 256->512/16
    pan_out1 = self.C3_n3(p_out1)  # 512->512/16
    p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
    p_out0 = torch.cat([p_out0, fpn_out0], dim=-3)  # 512->1024/32
    pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
    outputs = [pan_out2, pan_out1, pan_out0]
    return outputs

if __name__ == '__main__':
  x = torch.ones((2,3,128,128))
  print("x",x.shape)
  fpn = FPN()
  print("fpn_out_dark3",fpn(x)[0].shape)
  print("fpn_out_dark4",fpn(x)[1].shape)
  print("fpn_out_dark5",fpn(x)[2].shape)
  x = torch.ones((2,3,128,128))
  print("x",x.shape)
  pafpn = PAFPN()
  print("pafpn_pan2",pafpn(x)[0].shape)
  print("pafpn_pan1",pafpn(x)[1].shape)
  print("pafpn_pan0",pafpn(x)[2].shape)