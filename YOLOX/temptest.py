import cv2
import config
import numpy as np
import glob
from visualize import visualize
from boxes import cxcywh2xyxy
from PIL import Image


def load_images(pathes = glob.glob('./dataset/images/*'), num=60, height=config.IMAGE_SIZE, width=config.IMAGE_SIZE):
  pathes.sort()
  imgs = [np.array(Image.open(pathes[i]).resize((height, width))) for i in range(min(num,len(pathes)))]
  return imgs


def load_labels(pathes = glob.glob('./dataset/labels/*'), num=60):
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

if __name__ == '__main__':
  images = load_images()
  print("iamges ",type(images[0]))
  labels = load_labels()
  bboxes, scores, cls, cls_conf = process_labels(labels)
  for i, img in enumerate(images):
    res = visualize(images[i],bboxes[i], scores[i], cls[i], 0.5, config.PASCAL_CLASSES)
    cv2.imshow(f"image_{i}",cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows() 