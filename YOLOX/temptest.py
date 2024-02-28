import numpy as np
import glob
from PIL import Image


def load_images(pathes = glob.glob('YOLOX/data/images/*')):
  pathes.sort()
  imgs = [np.array(Image.open(img_file)) for img_file in pathes]
  return imgs


def load_labels(pathes = glob.glob('YOLOX/data/labels/*')):
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

if __name__ == '__main__':
  labels = load_images()
  print(labels[0].shape)