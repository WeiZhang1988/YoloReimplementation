import torch
import config
import torch.optim as optim
from tqdm import tqdm
from yolox import YOLOX
from losses import YoloXLoss
from dataset import YoloXDataset
from dataloader import YoloXDataLoader

def main():
  yolox = YOLOX(config.NUM_CLASSES).to(config.DEVICE)
  yoloXloss  = YoloXLoss(config.NUM_CLASSES).to(config.DEVICE)
  optimizer = optim.Adam(yolox.parameters(),lr=config.LEARNING_RATE,weight_decay=config.WEIGHT_DECAY)
  train_dataset = YoloXDataset(img_dir=config.IMG_DIR, 
                               label_dir=config.LABEL_DIR,
                               image_size=(config.IMAGE_SIZE,config.IMAGE_SIZE))
  print("len dataset", len(train_dataset))
  train_dataloader = YoloXDataLoader(dataset=train_dataset,
                                     batch_size=config.BATCH_SIZE,
                                     num_workers=config.NUM_WORKERS,
                                     collate_fn=YoloXDataset.collate_fn)
  for epoch in range(config.NUM_EPOCHS):
    print(f"{epoch}th epoch")
    loop = tqdm(train_dataloader, leave=True)
    mean_loss = []
    for batch_idx, (imgs, labels) in enumerate(loop):
      imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
      preds = yolox(imgs)
      loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = yoloXloss(preds, labels)
      optimizer.zero_grad(set_to_none=True)
      loss.backward(retain_graph = True)
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      mean_loss.append(loss.item())
    avg_loss = sum(mean_loss)/len(mean_loss)
    print(f"{epoch}th training epoch average loss: {avg_loss}")


if __name__ == '__main__':
  main()