import sys
import cv2
from PIL import Image

import torch, torchvision.transforms as T
import torchvision

from yolov3 import YoloV3
from show_bbox import visualization, show
import anchorbox

def main():
  args = sys.argv
  model_weight = args[1]
  target_img_file = args[2]
  img_size = (416,416)

  transform = T.Compose([T.ToTensor()])
    
  model = YoloV3()

  #print(model_weight)
  #data = torch.load(model_weight, weights_only=False)
  #print(type(data))
  model.load_state_dict(torch.load(model_weight))
  
  model = model.cuda()
  #model.eval()
  

  img = cv2.imread(target_img_file)
  img = cv2.resize(img, img_size)
  img = Image.fromarray(img)
  img = transform(img).unsqueeze(0).cuda()

  with torch.no_grad():
    preds  = list(model(img))

  img = cv2.imread(target_img_file)[:,:,::-1]
  orig_size = (img.shape[1],img.shape[0])
  img = cv2.resize(img , img_size)
  img = torch.tensor(img.transpose(2,0,1))

  for color,pred in zip(["red","green","blue"],preds):
    bbox_list = visualization(pred, anchorbox.obtain_anchorbox(), img_size,conf = 0.9)
    img = torchvision.utils.draw_bounding_boxes(img, torch.tensor(bbox_list), colors=color, width=1)

  show(img, orig_size)
  
if __name__ == '__main__':
  main()
