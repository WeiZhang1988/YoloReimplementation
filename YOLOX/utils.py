import glob
import torch
import config
import numpy as np
from PIL import Image

def save_checkpoint(state, checkpoint='./model/model.pth.tar'):
  print("--> saving checkpoint")
  torch.save(state, checkpoint)

def load_checkpoint(checkpoint, model, optimizer):
  print("--> loading checkpoint")
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])

def load_images(pathes = glob.glob('./dataset/images/*'), num=5, height=config.IMAGE_SIZE, width=config.IMAGE_SIZE):
  pathes.sort()
  imgs = [np.array(Image.open(pathes[i]).resize((height, width))).transpose(2, 0, 1) for i in range(min(num,len(pathes)))]
  return imgs