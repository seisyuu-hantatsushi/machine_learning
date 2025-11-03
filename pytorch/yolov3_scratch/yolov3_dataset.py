import os,math,urllib
import torch,torchvision
from pycocotools.coco import COCO

import anchorbox

class YOLOv3_DatasetFromCOCO(torch.utils.data.Dataset):

    def __init__(self, annotation_file, img_size):
        super().__init__()
        self.annotation_file = annotation_file
        self.coco_ctx = COCO(annotation_file)
        self.anchor_dict = anchorbox.obtain_anchorbox(annotation_file)
        self.ids = self.coco_ctx.getImgIds()
        self.img_size = img_size;
        self.num_of_class = len(anchorbox.map_category)
        self.map_size = [(int(self.img_size[1]/32), int(self.img_size[0]/32)),
                         (int(self.img_size[1]/16), int(self.img_size[0]/16)),
                         (int(self.img_size[1]/8),  int(self.img_size[0]/8))]
        self.anchor_iou = torch.cat([torch.zeros(9,2) , torch.tensor(self.anchor_dict[["width","height"]].values)] ,dim = 1)

        #print(self.num_of_class)
        #print(self.anchor_dict)
        
    def get_annotation_box(self, img_id):
        img_info   = self.coco_ctx.loadImgs(img_id)
        img_h      = img_info[0]['height']
        img_w      = img_info[0]['width']
        anno_ids   = self.coco_ctx.getAnnIds(img_id)
        anno_infos = self.coco_ctx.loadAnns(anno_ids)
        
        # COCOのBOX指定は, 左上中心座標と幅と高さ. 実画像上の座標  (lt_x, lt_y, w ,h)
        # YOLOのBOX指定は, BOX中心座標と幅と高さ.  正規化したもの  ( c_x,  c_y, w, h)
        # c_x = lt_x + w / 2, c_x = lt_x + w / 2, 
        #print(anno_infos)
        
        map_category = anchorbox.map_category
        yolo_bboxes = []
        for anno_info in anno_infos:
            lt_x = anno_info['bbox'][0]/img_w
            lt_y = anno_info['bbox'][1]/img_h
            w    = anno_info['bbox'][2]/img_w
            h    = anno_info['bbox'][3]/img_h
            c_x  = lt_x + w/2
            c_y  = lt_y + h/2
            #yolo_bbox = { 'category_id': map_category[str(anno_info['category_id'])]-1, 'x' : c_x, 'y' : c_y, 'width' : w, 'height': h }
            yolo_bbox = [ float(map_category[str(anno_info['category_id'])]-1), c_x, c_y, w, h ]
            yolo_bboxes.append(yolo_bbox)

        return yolo_bboxes

    def width_height_to_tw_th(self, wh):
        twths = []
        # bw = Pw*e^(tw), bh = Ph*e^(th)
        # bw = w, bh = h, Pw = anochor_w, Ph = anochor_h
        # tw = ln(bw/Pw), th = ln(bh/Ph)
        for i in range(len(self.anchor_dict)):
            anchor = self.anchor_dict.iloc[i]
            aw = anchor['width']
            ah = anchor['height']
            twths.append([math.log(wh[0]/aw), math.log(wh[1]/ah)])
        return twths
    
    def center_x_center_y_to_tx_ty(self, cxcy):
        txtys = []

        for size in self.map_size:
            # bx = sigma(tx) + Cx -> cx = sigma(tx) + grid_x
            # by = sigma(ty) + Cy -> cy = sigma(ty) + grid_y
            # tx = ln((bx - Cx)/(1-(bx - Cx)))
            # ty = ln((by - Cy)/(1-(by - Cy)))
            
            # YOLOは元サイズを32,16,8分割する.
            # cxcyには物体の中心の正規化座標が入っている.
            # self.map_sizeは入力サイズの32,16,8分の1のサイズが入っていて,
            # それと正規化座標をかけると,実画像上で物体の中心位置を含むグリッドの左上座標がわかる.
            grid_x = int(cxcy[0]*size[1])
            grid_y = int(cxcy[1]*size[0])

            tx = math.log((cxcy[0]*size[1] - grid_x + 1e-10) / (1 - cxcy[0]*size[1] + grid_x + 1e-10))
            ty = math.log((cxcy[1]*size[0] - grid_y + 1e-10) / (1 - cxcy[1]*size[0] + grid_y + 1e-10))
            txtys.append([grid_x , tx , grid_y ,ty])
            
        return txtys
    
    def annotation_bbox_to_tensor(self, bbox_list):

        tensor_list = []

        for size in self.map_size:
            for _ in range(3):
                tensor_list.append(torch.zeros((4 + 1 + self.num_of_class,size[0],size[1])))

        for bbox in bbox_list:
            cls = int(bbox[0])
            txtys = self.center_x_center_y_to_tx_ty(bbox[1:3])
            twths = self.width_height_to_tw_th(bbox[3:])

            object_iou = torch.cat([torch.zeros((1,2)), torch.tensor(bbox[3:]).unsqueeze(0)],dim=1)
            iou = torchvision.ops.boxes.box_iou(object_iou, self.anchor_iou)[0]
            obj_idx = torch.argmax(iou).item()
            for i , twth in enumerate(twths):
                tensor = tensor_list[i]
                txty = txtys[int(i/3)]
       
                if i == obj_idx:          
                    tensor[0,txty[2],txty[0]] = txty[1]
                    tensor[1,txty[2],txty[0]] = txty[3]
                    tensor[2,txty[2],txty[0]] = twth[0]
                    tensor[3,txty[2],txty[0]] = twth[1]
                    tensor[4,txty[2],txty[0]] = 1
                    tensor[5 + cls,txty[2],txty[0]] = 1
    
        scale3_label = torch.cat(tensor_list[0:3] , dim = 0)
        scale2_label = torch.cat(tensor_list[3:6] , dim = 0)
        scale1_label = torch.cat(tensor_list[6:]  , dim = 0)
            
        return scale3_label, scale2_label, scale1_label

    def __getitem__(self, idx):
        print(idx)
        img_id = self.ids[idx]
        #img_id = 9
        img_info   = self.coco_ctx.loadImgs(img_id)        
        bbox_list = self.get_annotation_box(img_id)
        scale3_label, scale2_label, scale1_label = self.annotation_bbox_to_tensor(bbox_list)        
        cocoDataFolder = os.path.dirname(os.path.dirname(self.annotation_file))
        coco_url = urllib.parse.urlparse(img_info[0]['coco_url'])
        img_file = cocoDataFolder+coco_url.path
        return img_file, scale3_label, scale2_label, scale1_label

    def __len__(self):
        return len(self.ids)

