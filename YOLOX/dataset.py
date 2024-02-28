import torch
import numpy as np
import os
import glob
from PIL import Image

class YoloXDataset(torch.utils.data.Dataset):
  """
  suppose file names follow the pattern:
  images: frameID.png
  labels: frameID.txt
  """
  def __init__(self, img_dir='./data/images', label_dir='./data/labels', image_size=(640,640)):
    super().__init__()
    self.image_size    = image_size
    self.img_dir       = img_dir
    self.label_dir     = label_dir
    self.images_png = glob.glob(img_dir+'/*')
    self.labels_txt = glob.glob(label_dir+'/*')
    assert os.path.exists(img_dir),   f"image directory {img_dir} does not exist"
    assert os.path.exists(label_dir), f"label directory {label_dir} does not exist"
    assert self.images_png, f'No images found'
    assert self.labels_txt, f'No labels found'
    self.images = self.load_images(self.images_png[:2007])
    self.labels = self.load_labels(self.labels_txt[:2007])
    assert len(self.images) == len(self.labels), "number of images must equal to number of labelstxt"
    self.num_samples = len(self.images)
  def __len__(self):
    return self.num_samples
  def __getitem__(self, index):
    imgs   = self.images[index].copy()
    labels = self.labels[index].copy()
    return (torch.from_numpy(np.array(imgs)).type(torch.float32), torch.from_numpy(np.array(labels)).type(torch.float32))
  def load_images(self, pathes):
    pathes.sort()
    imgs = [np.array(Image.open(img_file).resize(self.image_size).convert('RGB')).transpose((2, 0, 1)) for img_file in pathes]
    return imgs
  def load_labels(self, pathes):
    pathes.sort()
    labels = []
    for path in pathes:
      detects = []
      with open(path) as f:
        lines=f.readlines()
        for line in lines:
          detects.append(np.fromstring(line, dtype=float, sep=' '))
        detects = np.stack(detects)
      labels.append(detects)
    return labels
  @staticmethod
  def collate_fn(batch):
    imgs, labels = zip(*batch)  # transposed
    stacked_imgs = torch.stack(imgs, 0)
    max_num_obj = max([label.shape[0] for label in labels])
    stacked_labels = []
    for label in labels:
      max_label = torch.zeros((max_num_obj,5))
      max_label[0:label.shape[0]]=label
      stacked_labels.append(max_label)
    stacked_labels = torch.stack(stacked_labels, 0)
    return stacked_imgs, stacked_labels
    