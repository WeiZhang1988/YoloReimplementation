import torch
import torch.nn as nn

class IOUloss(nn.Module):
  def __init__(self, reduction="none", loss_type="iou"):
    super(IOUloss, self).__init__()
    self.reduction = reduction
    self.loss_type = loss_type
  def forward(self, pred, target):
    assert pred.shape[0] == target.shape[0]
    pred   = pred.view(-1, 4)
    target = target.view(-1, 4)
    tpp_left     = torch.max((pred[:,:2]-pred[:,2:]/2), (target[:,:2]-target[:,2:]/2))
    bottom_right = torch.min((pred[:,:2]+pred[:,2:]/2), (target[:,:2]+target[:,2:]/2))
    area_p = torch.prod(pred[:,2:],  dim=1)
    area_g = torch.prod(target[:,2:],dim=1)
    en = (tpp_left < bottom_right).type(tpp_left.type()).prod(dim=1)
    area_i = torch.prod(bottom_right - tpp_left, dim=1) * en
    area_u = area_p + area_g - area_i
    iou = (area_i) / (area_u + 1e-16)
    if self.loss_type == "iou":
      loss = 1 - iou ** 2
    elif self.loss_type == "giou":
      c_top_left     = torch.min((pred[:,:2]-pred[:,2:]/2), (target[:,:2]-target[:,2:]/2))
      c_bottom_right = torch.max((pred[:,:2]+pred[:,2:]/2), (target[:,:2]+target[:,2:]/2))
      area_c = torch.prod(c_bottom_right - c_top_left, dim=1)
      giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
      loss = 1 - giou.clamp(min=-1.0, max=1.0)
    else:
      raise AttributeError(f"unsupported iou loss type: {self.loss_type}")
    if self.reduction == "mean":
      loss = loss.mean()
    elif self.reduction == "sum":
      loss = loss.sum()
    return loss