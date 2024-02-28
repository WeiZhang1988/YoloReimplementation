import numpy as np
import torch
import torchvision

def filter_box(output, min_scale, max_scale):
  # output   [n, 4]
  # w------> [n]
  # h------> [n]
  # keep---> [n]
  # return-> [k, 4] k depends on keep
  w = output[:, 2] - output[:, 0]
  h = output[:, 3] - output[:, 1]
  keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
  return output[keep]

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
  # bboxes_a  [n, 4]
  # bboxes_b  [n, 4]
  # top_left  [n, n, 2]
  # bot_right [n, n, 2]
  # area_a    [n]
  # area_b    [n]
  # en        [n, n]
  # area_i    [n, n]
  # return    [n, n]
  if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
    raise IndexError
  if xyxy:
    top_left  = torch.max(bboxes_a[:,None,:2], bboxes_b[:,:2])
    bot_right = torch.min(bboxes_a[:,None,2:], bboxes_b[:,2:])
    area_a = torch.prod(bboxes_a[:,2:]-bboxes_a[:,:2], dim=-1)
    area_b = torch.prod(bboxes_b[:,2:]-bboxes_b[:,:2], dim=-1)
  else:
    top_left  = torch.max((bboxes_a[:,None,:2]-bboxes_a[:,None,2:]/2),
                          (bboxes_b[:,:2]-bboxes_b[:,2:]/2))
    bot_right = torch.min((bboxes_a[:,None,:2]+bboxes_a[:,None,2:]/2),
                          (bboxes_b[:,:2]+bboxes_b[:,2:]/2))
    area_a = torch.prod(bboxes_a[:,2:], dim=-1)
    area_b = torch.prod(bboxes_b[:,2:], dim=-1)
  en = (top_left < bot_right).type(top_left.type()).prod(dim=-1)
  area_i = torch.prod(bot_right-top_left, dim=-1) * en
  return area_i / (area_a[:,None] + area_b - area_i)

def matrix_iou_tensor(a, b):
  # a      [n, 4]
  # b      [n, 4]
  # lt     [n, n, 2]
  # rb     [n, n, 2]
  # area_i [n, n]
  # area_a [n]
  # area_b [n]
  # return [n, n]
  lt = torch.max(a[:,None,:2], b[:,:2])
  rb = torch.min(a[:,None,2:], b[:,2:])
  area_i = torch.prod(rb-lt, dim=-1) * (lt < rb).all(dim=-1)
  area_a = torch.prod(a[:,2:]-a[:,:2], dim=-1)
  area_b = torch.prod(b[:,2:]-b[:,:2], dim=-1)
  return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)

def matrix_iou_numpy(a, b):
  lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
  rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
  area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
  area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
  area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
  return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)

def adjust_box_anns_tensor(bbox, scale_ratio, padw, padh, w_max, h_max):
  # bbox [n, 4]
  # ---> [n, 4]
  bbox[:, 0::2] = torch.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
  bbox[:, 1::2] = torch.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
  return bbox

def adjust_box_anns_numpy(bbox, scale_ratio, padw, padh, w_max, h_max):
  bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
  bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
  return bbox

def xyxy2xywh(bboxes):
  bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
  bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
  return bboxes

def xyxy2cxcywh(bboxes):
  bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
  bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
  bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
  bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
  return bboxes

def cxcywh2xyxy(bboxes):
  bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
  bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
  bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
  bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
  return bboxes

if __name__ == '__main__':
  x=torch.tensor([[1,2,3,4],[5,6,7,8]])
  print("x ",x.shape)
  min_scale=0.1
  max_scale=5
  filter_box = filter_box(x, min_scale, max_scale)
  print("filter_box ",filter_box.shape)
  bboxes_iou = bboxes_iou(x, x, xyxy=True)
  print("bboxes_iou ",bboxes_iou.shape)
  matrix_iou_tensor = matrix_iou_tensor(x,x)
  print("matrix_iou_tensor ",matrix_iou_tensor.shape)
  adjust_box_anns_tensor = adjust_box_anns_tensor(x, 
                                                  1.0, 
                                                  1e-6, 
                                                  1e-6, 
                                                  100, 
                                                  100)
  print("adjust_box_anns_tensor ",adjust_box_anns_tensor.shape)