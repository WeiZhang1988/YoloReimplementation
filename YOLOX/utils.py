import glob
import pickle 
import torch
import config
import numpy as np
from PIL import Image

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