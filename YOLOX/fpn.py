import torch
import torch.nn as nn

from darknet import Darknet
from network_blocks import BaseConv


class FPN(nn.Module):
    def __init__(self, depth=53, in_features=["dark3", "dark4", "dark5"]):
      super().__init__()
      self.backbone = Darknet(depth)
      self.in_features = in_features
      self.out1_cbl = self._make_cbl(512, 256, 1)
      self.out1 = self._make_embedding([256, 512], 512 + 256)
      self.out2_cbl = self._make_cbl(256, 128, 1)
      self.out2 = self._make_embedding([128, 256], 256 + 128)
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

    def _make_embedding(self, filters_list, in_filters):
      m = nn.Sequential(*[self._make_cbl(in_filters, filters_list[0], 1),
                          self._make_cbl(filters_list[0], filters_list[1], 3),
                          self._make_cbl(filters_list[1], filters_list[0], 1),
                          self._make_cbl(filters_list[0], filters_list[1], 3),
                          self._make_cbl(filters_list[1], filters_list[0], 1),])
      return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
      with open(filename, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
      print("loading pretrained weights...")
      self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        out_features = self.backbone(x)
        x2, x1, x0 = [out_features[f] for f in self.in_features]
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)
        outputs = [out_dark3, out_dark4, x0]
        return outputs