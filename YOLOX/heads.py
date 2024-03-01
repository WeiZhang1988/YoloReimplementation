import torch
import torch.nn as nn
import torch.nn.functional as F
from network_blocks import BaseConv, DWConv

class Head(nn.Module):
  def __init__(
    self,
    num_classes,
    width=1.0,
    strides=[8, 16, 32],
    in_channelses=[256, 512, 1024],
    act="silu",
    depthwise=False,
  ):
    assert (3==len(strides)) == (3==len(in_channelses)), "length of both strides and in_channelses must be 3 in YoloX"
    super().__init__()
    self.num_classes = num_classes
    self.stems     = nn.ModuleList()
    self.cls_convs = nn.ModuleList()
    self.reg_convs = nn.ModuleList()
    self.cls_preds = nn.ModuleList()
    self.reg_preds = nn.ModuleList()
    self.obj_preds = nn.ModuleList()
    Conv = DWConv if depthwise else BaseConv
    for i in range(len(in_channelses)):
      self.stems.append(BaseConv(in_channels=int(in_channelses[i] * width),
                                 out_channels=int(256 * width),
                                 kernel_size=1,
                                 stride=1,
                                 act=act))
      self.cls_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width),
                                                 out_channels=int(256 * width),
                                                 kernel_size=3,
                                                 stride=1,
                                                 act=act),
                                            Conv(in_channels=int(256 * width),
                                                 out_channels=int(256 * width),
                                                 kernel_size=3,
                                                 stride=1,
                                                 act=act)]))
      self.reg_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width),
                                                 out_channels=int(256 * width),
                                                 kernel_size=3,
                                                 stride=1,
                                                 act=act),
                                            Conv(in_channels=int(256 * width),
                                                 out_channels=int(256 * width),
                                                 kernel_size=3,
                                                 stride=1,
                                                 act=act)]))
      self.cls_preds.append(nn.Conv2d(in_channels=int(256 * width),
                                      out_channels=self.num_classes,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0))
      self.reg_preds.append(nn.Conv2d(in_channels=int(256 * width),
                                      out_channels=4,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0))
      self.obj_preds.append(nn.Conv2d(in_channels=int(256 * width),
                                      out_channels=1,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0))
    self.use_l1 = True
    self.strides = strides
    self.grids = [torch.zeros(1)] * len(in_channelses)
  def initialize_biases(self, prior_prob):
    for conv in self.cls_preds:
      b = conv.bias.view(1, -1)
      b.data.fill_(-torch.log((1 - prior_prob) / prior_prob))
      conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    for conv in self.obj_preds:
      b = conv.bias.view(1, -1)
      b.data.fill_(-torch.log((1 - prior_prob) / prior_prob))
      conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
  def forward(self, xin):
    # xin               list of 3 tensors of [batch_size, channels, height, width]
    # outputs           list of 3 tensors of [batch_size, 4+1+num_classes, height, width]
    # ------>           list of 3 tensors of [batch_size, height * width, 4+1+num_classes]
    # x_shifts          list of 3 tensors of [1, height * width] 
    # y_shifts          list of 3 tensors of [1, height * width] 
    # expanded_strides  list of 3 tensors of [1, height * width] 
    # origin_preds      list of 3 tensors of [batch_size, height * width, 4] 
    outputs = []
    origin_preds = []
    x_shifts = []
    y_shifts = []
    expanded_strides = []
    for k, (cls_conv, 
            reg_conv, 
            stride_this_level, 
            x) in enumerate(zip(self.cls_convs, 
                                self.reg_convs, 
                                self.strides, 
                                xin)):
      x     = self.stems[k](x)
      cls_x = x
      reg_x = x
      cls_feat   = cls_conv(cls_x)
      cls_output = self.cls_preds[k](cls_feat)
      reg_feat   = reg_conv(reg_x)
      reg_output = self.reg_preds[k](reg_feat)
      obj_output = self.obj_preds[k](reg_feat)
      output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
      output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
      x_shifts.append(grid[:, :, 0])
      y_shifts.append(grid[:, :, 1])
      expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
      outputs.append(output)
      if self.use_l1:
        batch_size   = reg_output.shape[0]
        hsize, wsize = reg_output.shape[-2:]
        reg_output   = reg_output.view(batch_size, 1, 4, hsize, wsize)
        reg_output   = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
        origin_preds.append(reg_output.clone())
    return outputs, x_shifts, y_shifts, expanded_strides, origin_preds
  def get_output_and_grid(self, output, k, stride, dtype):
    # output [batch_size, 4+1+num_classes, height, width]
    # -----> [batch_size, height * width, 4+1+num_classes]
    # grid   [1, height * width, 2]
    grid = self.grids[k]
    batch_size = output.shape[0]
    n_ch = 5 + self.num_classes
    height, width = output.shape[-2:]
    if grid.shape[2:4] != output.shape[2:4]:
      yv, xv = torch.meshgrid([torch.arange(height), torch.arange(width)], indexing="ij")
      grid = torch.stack((xv, yv), 2).view(1, 1, height, width, 2).type(dtype)
      self.grids[k] = grid
    output = output.view(batch_size, 1, n_ch, height, width)
    output = output.permute(0, 1, 3, 4, 2).reshape(
        batch_size, height * width, -1
    )
    grid = grid.view(1, -1, 2)
    output[..., :2] = (output[..., :2].sigmoid() + grid) * stride
    output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
    return output, grid

if __name__ == '__main__':
  x1 = torch.ones((2,256,16,16))
  x2 = torch.ones((2,512,8,8))
  x3 = torch.ones((2,1024,4,4))
  x=[x1,x2,x3]
  for item in x:
    print("x ",item.shape)
  head = Head(20)
  res = head(x)
  for item in res:
    print("------")
    for subitem in item:
      print("head",subitem.shape)