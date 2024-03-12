import glob
import pickle 
import torch
import config
import numpy as np
from PIL import Image
from boxes import cxcywh2xyxy

def save_checkpoint(state, loss, checkpoint='./model/model.pth.tar', loss_serial='./model/loss.pkl'):
  print("--> saving checkpoint")
  torch.save(state, checkpoint)
  with open(loss_serial, "wb") as file:
    pickle.dump(loss, file)

def load_checkpoint(checkpoint, loss_serial, model, optimizer):
  state = torch.load(checkpoint)
  print("--> loading checkpoint")
  model.load_state_dict(state["state_dict"])
  optimizer.load_state_dict(state["optimizer"])
  with open(loss_serial, "rb") as file:
    loss = pickle.load(file)
  return loss

def load_images(pathes = glob.glob('./dataset/images/*'), num=5, height=config.IMAGE_SIZE, width=config.IMAGE_SIZE):
  pathes.sort()
  imgs = [np.array(Image.open(pathes[i]).resize((height, width))).transpose(2, 0, 1) for i in range(min(num,len(pathes)))]
  return imgs

def load_labels(pathes = glob.glob('./dataset/labels/*'), num=5):
  pathes.sort()
  labels = []
  for i in range(min(num,len(pathes))):
    detects = []
    with open(pathes[i]) as f:
      lines=f.readlines()
      for line in lines:
        detects.append(np.fromstring(line, dtype=float, sep=' '))
      detects = np.stack(detects)
    labels.append(detects)
  return labels

def process_labels(labels,height=config.IMAGE_SIZE,width=config.IMAGE_SIZE):
  bboxes    = []
  scores    = []
  cls       = []
  cls_conf  = []
  for label in labels:
    label[:,1::2] *= height
    label[:,2::2] *= width
    bboxes.append(cxcywh2xyxy(label[:,1:]))
    scores.append(np.ones(label.shape[0]))
    cls.append(label[:,0:1])
    cls_conf.append(np.ones(label.shape[0]))
  return bboxes, scores, cls, cls_conf