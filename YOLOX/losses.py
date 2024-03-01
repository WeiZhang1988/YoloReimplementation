import torch
import torch.nn as nn
import torch.nn.functional as F
from boxes import bboxes_iou

class IOUloss(nn.Module):
  def __init__(self, reduction="none", loss_type="iou"):
    super().__init__()
    self.reduction = reduction
    self.loss_type = loss_type
  def forward(self, pred, target):
    # pred and target [batch_size, num_obj, 4]
    # --------------> [batch_size*num_obj,  4]
    # top_left------> [batch_size*num_obj,  2]
    # bot_right-----> [batch_size*num_obj,  2]
    # area_p--------> [batch_size*num_obj]
    # area_g--------> [batch_size*num_obj]
    # en------------> [batch_size*num_obj]
    # area_i--------> [batch_size*num_obj]
    # area_u--------> [batch_size*num_obj]
    # iou-----------> [batch_size*num_obj]
    # loss----------> [batch_size*num_obj]
    # c_top_left----> [batch_size*num_obj]
    # c_bot_right---> [batch_size*num_obj]
    # c_en----------> [batch_size*num_obj]
    # area_c--------> [batch_size*num_obj]
    # giou----------> [batch_size*num_obj]
    # loss----------> [batch_size*num_obj]
    # --------------> [1]
    assert pred.shape[0] == target.shape[0]
    pred   = pred.view(-1, 4)
    target = target.view(-1, 4)
    top_left  = torch.max((pred[:,:2]-pred[:,2:]/2), (target[:,:2]-target[:,2:]/2))
    bot_right = torch.min((pred[:,:2]+pred[:,2:]/2), (target[:,:2]+target[:,2:]/2))
    area_p = torch.prod(pred[:,2:],  dim=-1)
    area_g = torch.prod(target[:,2:],dim=-1)
    en = (top_left < bot_right).type(top_left.type()).prod(dim=-1)
    area_i = torch.prod(bot_right - top_left, dim=-1) * en
    area_u = area_p + area_g - area_i
    iou = (area_i) / (area_u + 1e-16)
    if self.loss_type == "iou":
      loss = 1 - iou ** 2
    elif self.loss_type == "giou":
      c_top_left  = torch.min((pred[:,:2]-pred[:,2:]/2), (target[:,:2]-target[:,2:]/2))
      c_bot_right = torch.max((pred[:,:2]+pred[:,2:]/2), (target[:,:2]+target[:,2:]/2))
      c_en = (c_top_left < c_bot_right).type(c_top_left.type()).prod(dim=-1)
      area_c = torch.prod(c_bot_right - c_top_left, dim=-1) * c_en
      giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
      loss = 1 - giou.clamp(min=-1.0, max=1.0)
    else:
      raise AttributeError(f"unsupported iou loss type: {self.loss_type}")
    if self.reduction == "mean":
      loss = loss.mean()
    elif self.reduction == "sum":
      loss = loss.sum()
    return loss

class YoloXLoss(nn.Module):
  def __init__(self, num_classes=20):
    super().__init__()
    self.num_classes = num_classes
    self.l1_loss = nn.L1Loss(reduction="none")
    self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
    self.iou_loss = IOUloss(reduction="none")
  def forward(self, preds, labels=None):
    outputs, x_shifts, y_shifts, expanded_strides, origin_preds = preds
    return self.get_losses(x_shifts,
                           y_shifts,
                           expanded_strides,
                           labels,
                           outputs,
                           origin_preds,
                           torch.float32)

  def get_losses(self, 
                 x_shifts,
                 y_shifts,
                 expanded_strides,
                 labels,
                 outputs,
                 origin_preds,
                 dtype):
    outputs    = torch.cat(outputs, 1)
    bbox_preds = outputs[:, :, :4]   # [batch, height1*width1+height2*width2+height3*width3, 4]
    obj_preds  = outputs[:, :, 4:5]  # [batch, height1*width1+height2*width2+height3*width3, 1]
    cls_preds  = outputs[:, :, 5:]   # [batch, height1*width1+height2*width2+height3*width3, n_cls]
    num_labels = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
    total_num_cells = outputs.shape[1]
    x_shifts = torch.cat(x_shifts, 1)  # [1, height1*width1+height2*width2+height3*width3]
    y_shifts = torch.cat(y_shifts, 1)  # [1, height1*width1+height2*width2+height3*width3]
    expanded_strides = torch.cat(expanded_strides, 1) # [1, height1*width1+height2*width2+height3*width3]
    origin_preds = torch.cat(origin_preds, 1)  # [1, height1*width1+height2*width2+height3*width3]
    cls_targets = []
    reg_targets = []
    l1_targets  = []
    obj_targets = []
    fg_masks    = []
    num_fg      = 0.0
    num_gts     = 0.0
    for batch_idx in range(outputs.shape[0]):
      num_gt = int(num_labels[batch_idx])
      num_gts += num_gt
      if num_gt == 0:
        cls_target = outputs.new_zeros((0, self.num_classes))
        reg_target = outputs.new_zeros((0, 4))
        l1_target  = outputs.new_zeros((0, 4))
        obj_target = outputs.new_zeros((total_num_cells, 1))
        fg_mask    = outputs.new_zeros(total_num_cells).bool()
      else:
        gt_bboxes_per_image    = labels[batch_idx, :num_gt, 1:5]
        gt_classes             = labels[batch_idx, :num_gt, 0]
        bboxes_preds_per_image = bbox_preds[batch_idx]
        try:
          (gt_matched_classes,
           fg_mask,
           pred_ious_this_matching,
           matched_gt_inds,
           num_fg_img) = self.get_assignments(batch_idx,
                                              num_gt,
                                              gt_bboxes_per_image,
                                              gt_classes,
                                              bboxes_preds_per_image,
                                              expanded_strides,
                                              x_shifts,
                                              y_shifts,
                                              cls_preds,
                                              obj_preds)
        except RuntimeError as e:
          if "CUDA out of memory. " not in str(e):
            raise
          torch.cuda.empty_cache()
          (gt_matched_classes,
           fg_mask,
           pred_ious_this_matching,
           matched_gt_inds,
           num_fg_img) = self.get_assignments(batch_idx,
                                              num_gt,
                                              gt_bboxes_per_image,
                                              gt_classes,
                                              bboxes_preds_per_image,
                                              expanded_strides,
                                              x_shifts,
                                              y_shifts,
                                              cls_preds,
                                              obj_preds,
                                              "cpu")
        torch.cuda.empty_cache()
        num_fg += num_fg_img
        cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
        obj_target = fg_mask.unsqueeze(-1)
        reg_target = gt_bboxes_per_image[matched_gt_inds]
        l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)),
                                       gt_bboxes_per_image[matched_gt_inds],
                                       expanded_strides[0][fg_mask],
                                       x_shifts=x_shifts[0][fg_mask],
                                       y_shifts=y_shifts[0][fg_mask])
      cls_targets.append(cls_target)
      reg_targets.append(reg_target)
      obj_targets.append(obj_target.to(dtype))
      fg_masks.append(fg_mask)
      l1_targets.append(l1_target)
    cls_targets = torch.cat(cls_targets, 0)
    reg_targets = torch.cat(reg_targets, 0)
    obj_targets = torch.cat(obj_targets, 0)
    fg_masks    = torch.cat(fg_masks, 0)
    l1_targets  = torch.cat(l1_targets, 0)
    num_fg = max(num_fg, 1)
    loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
    loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
    loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
    reg_weight = 5.0
    loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
    return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )
  def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
    l1_target[:, 0] = gt[:, 0] / stride - x_shifts
    l1_target[:, 1] = gt[:, 1] / stride - y_shifts
    l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
    l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
    return l1_target
  @torch.no_grad()
  def get_assignments(self,
                      batch_idx,
                      num_gt,
                      gt_bboxes_per_image,
                      gt_classes,
                      bboxes_preds_per_image,
                      expanded_strides,
                      x_shifts,
                      y_shifts,
                      cls_preds,
                      obj_preds,
                      mode="gpu"):
    if mode == "cpu":
      print("-----------Using CPU for the Current Batch-------------")
      gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
      bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
      gt_classes = gt_classes.cpu().float()
      expanded_strides = expanded_strides.cpu().float()
      x_shifts = x_shifts.cpu()
      y_shifts = y_shifts.cpu()
    fg_mask, geometry_relation = self.get_geometry_constraint(gt_bboxes_per_image,
                                                              expanded_strides,
                                                              x_shifts,
                                                              y_shifts)
    bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
    cls_preds_ = cls_preds[batch_idx][fg_mask]
    obj_preds_ = obj_preds[batch_idx][fg_mask]
    num_in_boxes_cell = bboxes_preds_per_image.shape[0]
    if mode == "cpu":
      gt_bboxes_per_image = gt_bboxes_per_image.cpu()
      bboxes_preds_per_image = bboxes_preds_per_image.cpu()
    pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
    gt_cls_per_image = (F.one_hot(gt_classes.to(torch.int64), self.num_classes).float())
    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
    if mode == "cpu":
      cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
    with torch.cuda.amp.autocast(enabled=False):
      cls_preds_ = ( cls_preds_.float() * obj_preds_.float()).sqrt()
      pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                                                  gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_cell, 1),
                                                  reduction="none").sum(-1)
    del cls_preds_
    cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + float(1e6) * (~geometry_relation))
    (num_fg,
    gt_matched_classes,
    pred_ious_this_matching,
    matched_gt_inds) = self.simota_matching(cost, 
                                            pair_wise_ious, 
                                            gt_classes, 
                                            num_gt, 
                                            fg_mask)
    del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
    if mode == "cpu":
      gt_matched_classes = gt_matched_classes.cuda()
      fg_mask = fg_mask.cuda()
      pred_ious_this_matching = pred_ious_this_matching.cuda()
      matched_gt_inds = matched_gt_inds.cuda()
    return (gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg)
  def get_geometry_constraint(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts):
    expanded_strides_per_image = expanded_strides[0]
    x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
    y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
    # in fixed center
    center_radius = 1.5
    center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
    gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
    gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
    gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
    gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist
    c_l = x_centers_per_image   - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - x_centers_per_image
    c_t = y_centers_per_image   - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - y_centers_per_image
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    cell_filter = is_in_centers.sum(dim=0) > 0
    geometry_relation = is_in_centers[:, cell_filter]
    return cell_filter, geometry_relation
  def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
    matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
    n_candidate_k = min(10, pair_wise_ious.size(1))
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    for gt_idx in range(num_gt):
      _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
      matching_matrix[gt_idx][pos_idx] = 1
    del topk_ious, dynamic_ks, pos_idx
    anchor_matching_gt = matching_matrix.sum(0)
    # deal with the case that one anchor matches multiple ground-truths
    if anchor_matching_gt.max() > 1:
        multiple_match_mask = anchor_matching_gt > 1
        _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
        matching_matrix[:, multiple_match_mask] *= 0
        matching_matrix[cost_argmin, multiple_match_mask] = 1
    fg_mask_inboxes = anchor_matching_gt > 0
    num_fg = fg_mask_inboxes.sum().item()
    fg_mask[fg_mask.clone()] = fg_mask_inboxes
    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    gt_matched_classes = gt_classes[matched_gt_inds]
    pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
        fg_mask_inboxes
    ]
    return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
  