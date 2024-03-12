import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import config
import warnings
import torch.optim as optim
from tqdm import tqdm
from yolox import YOLOX
from losses import YoloXLoss
from dataset import YoloXDataset
from dataloader import YoloXDataLoader
from utils import load_checkpoint, save_checkpoint, load_images, load_labels, process_labels
from visualize import visualize
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=True
torch.cuda.synchronize()

def train():
  yolox = YOLOX(config.NUM_CLASSES).to(config.DEVICE)
  yoloXloss  = YoloXLoss(config.NUM_CLASSES).to(config.DEVICE)
  scaler = torch.cuda.amp.GradScaler()
  optimizer = optim.Adam(yolox.parameters(),lr=config.LEARNING_RATE,weight_decay=config.WEIGHT_DECAY)
  train_dataset = YoloXDataset(img_dir=config.IMG_DIR, 
                               label_dir=config.LABEL_DIR,
                               image_size=(config.IMAGE_SIZE,config.IMAGE_SIZE))
  train_dataloader = YoloXDataLoader(dataset=train_dataset,
                                     batch_size=config.BATCH_SIZE,
                                     num_workers=config.NUM_WORKERS,
                                     collate_fn=YoloXDataset.collate_fn,
                                     shuffle=True)
  scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                           max_lr=1e-3, epochs=config.NUM_EPOCHS, steps_per_epoch=len(train_dataloader))
  avg_loss_pre_epoch = 1e6 
  if os.path.exists(config.CHECKPOINT_FILE) and os.path.exists(config.LOSS_FILE) and config.LOAD_MODEL:
    avg_loss = load_checkpoint(config.CHECKPOINT_FILE,config.LOSS_FILE,yolox,optimizer)    
    avg_loss_pre_epoch = avg_loss                           
  for epoch in range(config.NUM_EPOCHS):
    print(f"## starting {epoch}th epoch")
    loop = tqdm(train_dataloader, leave=True)
    mean_loss = []
    for batch_idx, (imgs, labels) in enumerate(loop):
      # for img, label in zip(imgs, labels):
      #   print("label-- ",label)
      #   cv2.imshow(f"image_",cv2.cvtColor((img.cpu().numpy().transpose(1,2,0)*255.).astype(np.uint8), cv2.COLOR_BGR2RGB))
      #   cv2.waitKey(0)
      #   cv2.destroyAllWindows()
      imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
      preds = yolox(imgs)
      loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = yoloXloss(preds, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      mean_loss.append(loss.item())
    scheduler.step()
    avg_loss = sum(mean_loss)/len(mean_loss)
    print(f"== {epoch}th training epoch average loss: {avg_loss}")
    if avg_loss < avg_loss_pre_epoch and config.SAVE_MODEL:
      state = {"state_dict": yolox.state_dict(),"optimizer": optimizer.state_dict(),}
      save_checkpoint(state, avg_loss, checkpoint=config.CHECKPOINT_FILE, loss_serial=config.LOSS_FILE)
      time.sleep(3)
    avg_loss_pre_epoch = avg_loss
@torch.no_grad()
def eval():
  yolox = YOLOX(config.NUM_CLASSES).to(config.DEVICE)
  optimizer = optim.Adam(yolox.parameters(),lr=config.LEARNING_RATE,weight_decay=config.WEIGHT_DECAY)
  origin_images = load_images()
  labels = load_labels()
  gt_bboxes, gt_scores, gt_cls, gt_cls_conf = process_labels(labels)
  images = torch.tensor(origin_images,dtype=torch.float32).to(config.DEVICE)
  if os.path.exists(config.CHECKPOINT_FILE):
    avg_loss = load_checkpoint(config.CHECKPOINT_FILE,config.LOSS_FILE,yolox,optimizer)  
    outputs, _, _, _, _ = yolox(images/255.)
    bboxes, scores, cls = yolox.extract(outputs,config.IMAGE_SIZE,config.IMAGE_SIZE)
    for i, img in enumerate(origin_images):
      print(f"pred_cls {cls[i]}, gt_cls {gt_cls[i]}")
      img = img.transpose(1,2,0)
      res = visualize(img, bboxes[i], scores[i], cls[i], 0.5, config.PASCAL_CLASSES)
      cv2.imshow(f"image_{i}",cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

if __name__ == '__main__':
  ONLY_EVAL = False
  if ONLY_EVAL:
    eval()
  else:
    train()
    eval()