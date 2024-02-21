import torch
import torch.nn as nn
from network_blocks import BaseConv, DWConv, CSPLayer
from darknet import CSPDarknet

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
  pafpn = PAFPN()
  print("pafpn_pan2",pafpn(x)[0].shape)
  print("pafpn_pan1",pafpn(x)[1].shape)
  print("pafpn_pan0",pafpn(x)[2].shape)