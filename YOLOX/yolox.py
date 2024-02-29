import torch
import torch.nn as nn
import torch.nn.functional as F
from heads import Head
from fpns import PAFPN
from losses import IOUloss
from boxes import cxcywh2xyxy

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
  def extract(self, preds, height,width):
    preds = torch.cat(preds,dim=1)
    bbox_preds = preds[..., :4]   # [batch, height*width, 4]
    obj_preds  = preds[..., 4]  # [batch, height*width, 1]
    cls_preds  = torch.argmax(preds[..., 5:],dim=-1,keepdim=False)   # [batch, height*width, n_cls] -> [batch, height*width, 1]
    bbox_preds[...,0::2] *= height
    bbox_preds[...,1::2] *= width
    bbox_preds = cxcywh2xyxy(bbox_preds)
    return bbox_preds.cpu().numpy(), obj_preds.cpu().numpy(), cls_preds.cpu().numpy()

if __name__ == '__main__':
  num_class = 20
  imgs      = torch.ones((2,3,640,640))
  labels    = torch.ones((2,6400,5))
  yolox     = YOLOX(num_class)
  outputs, x_shifts, y_shifts, expanded_strides, origin_preds = yolox(imgs)
  for item in outputs:
    print("outputs ", item.shape)