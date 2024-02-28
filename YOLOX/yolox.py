import torch
import torch.nn as nn
import torch.nn.functional as F
from heads import Head
from fpns import PAFPN
from losses import IOUloss
from boxes import bboxes_iou

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
    self.l1_loss = nn.L1Loss(reduction="none")
    self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
    self.iou_loss = IOUloss(reduction="none")
  def forward(self, x):
    return self.head(self.backbone(x))

if __name__ == '__main__':
  num_class = 10
  imgs      = torch.ones((2,3,640,640))
  labels    = torch.ones((2,6400,5))
  yolox     = YOLOX(num_class)
  outputs, x_shifts, y_shifts, expanded_strides, origin_preds = yolox(imgs)
  for item in outputs:
    print("outputs ", item.shape)