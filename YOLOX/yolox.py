import torch
import torch.nn as nn
import torch.nn.functional as F
from heads import Head
from fpns import PAFPN
from losses import IOUloss
from boxes import cxcywh2xyxy, non_max_suppression

class YOLOX(nn.Module):
  def __init__(self, num_classes=20, backbone=None, head=None):
    super().__init__()
    self.num_classes = num_classes
    if backbone is None:
        backbone = PAFPN()
    if head is None:
        head = Head(num_classes)
    self.backbone = backbone
    self.head = head
    self.num_classes = num_classes
  def forward(self, x):
    return self.head(self.backbone(x))
  def extract(self, preds, height, width):
    preds = torch.cat(preds,dim=1)
    preds[...,0:3:2] *= height
    preds[...,1:4:2] *= width
    batched_preds_out = []
    for items in preds:
      batched_preds_out.append(non_max_suppression(items))
    bbox_preds = torch.stack(batched_preds_out)[..., :4]   # [1, m, 4]
    obj_preds  = torch.stack(batched_preds_out)[..., 4:5]  # [1, m, 1]
    cls_preds  = torch.argmax(torch.stack(batched_preds_out)[..., 5:],dim=-1,keepdim=True)   # [batch, m, n_cls] -> [batch, m, 1]
    return bbox_preds.cpu().numpy(), obj_preds.cpu().numpy(), cls_preds.cpu().numpy()

if __name__ == '__main__':
  num_class = 20
  imgs      = torch.ones((2,3,640,640))
  labels    = torch.ones((2,6400,5))
  yolox     = YOLOX(num_class)
  outputs, x_shifts, y_shifts, expanded_strides, origin_preds = yolox(imgs)
  for item in outputs:
    print("outputs ", item.shape)