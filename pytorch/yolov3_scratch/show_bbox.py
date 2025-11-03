import os,sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch, torch.nn as nn, torchvision.transforms as T
import torchvision, torchinfo

from yolov3_dataset import YoloV3_DatasetFromCOCO

def visualization(y_pred,anchor,img_size,num_of_class,conf = 0.5,is_label = False):
  size = y_pred.shape[2]
  anchor_size = anchor[anchor["scale"] == size]
  bbox_list = []
  for i in range(3):
    a = anchor_size.iloc[i]
    grid = img_size[0]/size
    y_pred_cut = y_pred[0,i*(4 + 1 + num_of_class) :(i+1)*(4 + 1 + num_of_class) ].cpu()
    if is_label:
      y_pred_conf = y_pred_cut[4,:,:].cpu().numpy()
    else:
      y_pred_conf = torch.sigmoid(y_pred_cut[4,:,:]).cpu().numpy()         
    index = np.where(y_pred_conf > conf)
    
    for y,x in zip(index[0],index[1]):
      cx = x*grid + torch.sigmoid(y_pred_cut[0,y,x]).numpy()*grid
      cy = y*grid + torch.sigmoid(y_pred_cut[1,y,x]).numpy()*grid
      width = a["width"]*torch.exp(y_pred_cut[2,y,x]).numpy()*img_size[0]
      height = a["height"]*torch.exp(y_pred_cut[3,y,x]).numpy()*img_size[1]
      xmin,ymin,xmax,ymax = cx - width/2 , cy - height/2 ,cx + width/2 , cy + height/2
      bbox_list.append([xmin,ymin,xmax,ymax])
  return bbox_list
 
import torchvision.transforms.functional as FF

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

def show(imgs,orig_size):
    print(orig_size)
    if not isinstance(imgs, list): #listでなければlistにする.
        imgs = [imgs] 
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FF.to_pil_image(img)
        img = pil2cv(img)
        img = cv2.resize(img, orig_size)
        cv2.imshow("",img)
        cv2.waitKey(0)

def main():
    
    args = sys.argv
    annotation_file = args[1]
    img_size = (416,416)
    train_dataset = YoloV3_DatasetFromCOCO(annotation_file, img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1)

    #np.set_printoptions(threshold=sys.maxsize)
    #torch.set_printoptions(profile="full")

    for n, (img_path, scale3_label, scale2_label, scale1_label) in enumerate(train_loader):
        img_path = img_path[0]
        img = cv2.imread(img_path)[:,:,::-1]
        orig_size = (img.shape[1],img.shape[0])
        #cv2.imshow(img_path, img)
        #cv2.waitKey(0)
        img = cv2.resize(img, img_size)
        img = torch.tensor(img.transpose(2,0,1))
        preds = [scale3_label, scale2_label, scale1_label]
        for color,pred in zip(["red","green","blue"],preds):
            bboxes = visualization(pred,train_dataset.anchor_dict,
                                   img_size,train_dataset.num_of_class,
                                   conf = 0.9,is_label = True)
            img = torchvision.utils.draw_bounding_boxes(img, torch.tensor(bboxes), colors=color, width=1)
        show(img,orig_size)

if __name__ == '__main__':
    main()

