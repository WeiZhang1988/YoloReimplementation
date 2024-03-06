import numpy as np
import torch

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
  # bboxes_a  [n, 4];    bboxes_b  [n, 4]
  # top_left  [n, n, 2]; bot_right [n, n, 2]
  # area_a    [n];       area_b    [n]
  # en        [n, n]
  # area_i    [n, n]
  # return    [n, n]
  if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
    raise IndexError
  if xyxy:
    tl  = torch.max(bboxes_a[:,None,:2], bboxes_b[:,:2])
    br = torch.min(bboxes_a[:,None,2:], bboxes_b[:,2:])
    area_a = torch.prod(bboxes_a[:,2:]-bboxes_a[:,:2], dim=-1)
    area_b = torch.prod(bboxes_b[:,2:]-bboxes_b[:,:2], dim=-1)
  else:
    tl  = torch.max((bboxes_a[:,None,:2]-bboxes_a[:,None,2:]/2),
                    (bboxes_b[:,:2]-bboxes_b[:,2:]/2))
    br = torch.min((bboxes_a[:,None,:2]+bboxes_a[:,None,2:]/2),
                   (bboxes_b[:,:2]+bboxes_b[:,2:]/2))
    area_a = torch.prod(bboxes_a[:,2:], dim=-1)
    area_b = torch.prod(bboxes_b[:,2:], dim=-1)
  en = (tl < br).type(tl.type()).prod(dim=-1)
  area_i = torch.prod(br-tl, dim=-1) * en
  return area_i / (area_a[:,None] + area_b - area_i + 1e-12)

def matrix_iou(bboxes_a, bboxes_b, xyxy=True):
  # bboxes_a  [n, 4];    bboxes_b  [n, 4]
  # top_left  [n, n, 2]; bot_right [n, n, 2]
  # area_a    [n];       area_b    [n]
  # en        [n, n]
  # area_i    [n, n]
  # return    [n, n]
  if bboxes_a.shape[-1] != 4 or bboxes_b.shape[-1] != 4:
    raise IndexError
  if xyxy:
    tl = np.maximum(bboxes_a[:,np.newaxis,:2], bboxes_b[:,:2])
    br = np.minimum(bboxes_a[:,np.newaxis,2:], bboxes_b[:,2:])
    area_a = np.prod(bboxes_a[:,2:]-bboxes_a[:,:2], axis=-1)
    area_b = np.prod(bboxes_b[:,2:]-bboxes_b[:,:2], axis=-1)
  else:
    tl = np.maximum((bboxes_a[:,np.newaxis,:2]-bboxes_a[:,np.newaxis,2:]/2),
                    (bboxes_b[:,:2]-bboxes_b[:,2:]/2))
    br = np.minimum((bboxes_a[:,np.newaxis,:2]+bboxes_a[:,np.newaxis,2:]/2),
                    (bboxes_b[:,:2]+bboxes_b[:,2:]/2))
    area_a = np.prod(bboxes_a[:,2:], axis=-1)
    area_b = np.prod(bboxes_b[:,2:], axis=-1)
  en = (tl < br).astype(tl.dtype).prod(axis=-1)
  area_i = np.prod(br-tl, axis=-1) * en
  return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)

def cxcywh2xyxy(bboxes):
  # bboxes  [n, 4]
  # return  [n, 4]
  bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
  bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
  bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
  bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
  return bboxes

def non_max_suppression(preds, iou_threshold=0.5, conf_threshold=0.5, cvtxyxy=True):
  # preds  [n, 25]
  # return [n, 25]
  preds = [pred for pred in preds if pred[4] > conf_threshold]
  preds = sorted(preds, key=lambda x: x[4], reverse=True)
  preds_out = []
  while preds:
    chosen_pred = preds.pop(0)
    preds = [pred for pred in preds if pred[4] != chosen_pred[4] \
                                    or bboxes_iou(chosen_pred[0:4].unsqueeze(0), torch.stack(preds), ~cvtxyxy) \
                                       < iou_threshold]
    if cvtxyxy:
      chosen_pred[0:4] = cxcywh2xyxy(chosen_pred[0:4].unsqueeze(0))
    preds_out.append(chosen_pred)
  return torch.stack(preds_out)

if __name__ == '__main__':
  x=torch.tensor([[1,2,3,4],[5,6,7,8]])
  bboxes_iou1 = bboxes_iou(x, x, xyxy=True)
  bboxes_iou2 = bboxes_iou(x, x, xyxy=False)
  print("bboxes_iou xyxy ",bboxes_iou1)
  print("bboxes_iou xyhw ",bboxes_iou2)
  matrix_iou1 = matrix_iou(x.numpy(), x.numpy(), xyxy=True)
  matrix_iou2 = matrix_iou(x.numpy(), x.numpy(), xyxy=False)
  print("matrix_iou xyxy ",matrix_iou1)
  print("matrix_iou xyhw ",matrix_iou2)
  print("bboxes_iou1 - matrix_iou1 ", bboxes_iou1.numpy()-matrix_iou1)
  print("bboxes_iou2 - matrix_iou2 ", bboxes_iou2.numpy()-matrix_iou2)
  y=torch.tensor([[0.1],[0.6]])
  non_max_suppression = non_max_suppression(x,y)
  print("non_max_suppression ",non_max_suppression)