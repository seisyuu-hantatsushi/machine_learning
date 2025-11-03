import sys
import cv2
from PIL import Image
import torch, torchvision.transforms as T
from tqdm import tqdm

from yolov3 import YoloV3
from yolov3_dataset import YoloV3_DatasetFromCOCO

criterion_bce = torch.nn.BCEWithLogitsLoss()

def bbox_metric(y_pred, y_true, class_n = 80):
  for i in range(3):
    y_pred_cut = y_pred[:,i*(4 + 1 + class_n) :(i+1)*(4 + 1 + class_n) ]
    y_true_cut = y_true[:,i*(4 + 1 + class_n) :(i+1)*(4 + 1 + class_n) ]
    loss_coord = torch.sum(torch.square(y_pred_cut[:,0:4] - y_true_cut[:,0:4])*y_true_cut[:,4])
    loss_obj = torch.sum((-1 * torch.log(torch.sigmoid(y_pred_cut[:,4] )+ 1e-10) + criterion_bce(y_pred_cut[:,5:],y_true_cut[:,5:]))*y_true_cut[:,4]) 
    loss_noobj =  torch.sum((-1 * torch.log(1 - torch.sigmoid(y_pred_cut[:,4])+ 1e-10))*(1 - y_true_cut[:,4]))
    return loss_coord , loss_obj , loss_noobj

def main():
  args = sys.argv
  annotation_file = args[1]
  img_size = (416,416)
  train_dataset = YoloV3_DatasetFromCOCO(annotation_file, img_size, bShortSet = True)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1)

  transform = T.Compose([T.ToTensor()])
    
  model = YoloV3().to('cuda')
  #model = YoloV3()
  optimizer = torch.optim.Adam(model.parameters())

  lambda_coord = 1
  lambda_obj   = 10
  lambda_noobj = 1

  confidence = 0.5
  best_loss  = 99999

  for epoch in range(300):
    total_train_loss = 0
    total_train_loss_coord = 0
    total_train_loss_obj = 0
    total_train_loss_noobj = 0
    
    with tqdm(train_loader) as pbar:
      pbar.set_description("[train] Epoch %d" % epoch)
      for n, (img_file, scale3_label , scale2_label ,scale1_label) in enumerate(pbar):
        optimizer.zero_grad()
        img = cv2.imread(img_file[0])
        img = cv2.resize(img , img_size)
        img = Image.fromarray(img)
        img = transform(img)
        img = img.cuda()
        img = img.unsqueeze(0)
        scale1_label = scale1_label.cuda()
        scale2_label = scale2_label.cuda()
        scale3_label = scale3_label.cuda()
        labels = [scale3_label , scale2_label ,scale1_label]
        preds  = list(model(img))
        loss_coord = 0
        loss_obj = 0
        loss_noobj = 0
        for label , pred in zip(labels , preds):
          _loss_coord , _loss_obj , _loss_noobj = bbox_metric(pred , label)
          loss_coord += _loss_coord
          loss_obj += _loss_obj
          loss_noobj += _loss_noobj
                
        loss = lambda_coord*loss_coord + lambda_obj*loss_obj + lambda_noobj*loss_noobj
        total_train_loss += loss.item()
        total_train_loss_coord += loss_coord.item()
        total_train_loss_obj += loss_obj.item()
        total_train_loss_noobj += loss_noobj.item()
        loss.backward()
        optimizer.step()
        pbar.set_description("[train] Epoch %d loss %f loss_coord %f loss_obj %f loss_noobj %f" %
                             (epoch ,total_train_loss/(n+1),total_train_loss_coord/(n+1) ,
                              total_train_loss_obj/(n+1),total_train_loss_noobj/(n+1)))

      if best_loss > total_train_loss/(n+1):
        model_path = 'model.pth'
        torch.save(model.state_dict(), model_path)
        best_loss = total_train_loss/(n+1)

  return

if __name__ == '__main__':
    main()
