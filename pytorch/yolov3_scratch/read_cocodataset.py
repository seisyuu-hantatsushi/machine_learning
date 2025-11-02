import os,sys,urllib 
import cv2

from pycocotools.coco import COCO

# COCOAPI https://github.com/cocodataset/cocoapi 
# COCOデータ・セットは91カテゴリが用意されているが,実際には80カテゴリしか使われていない.
# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

map_category = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}

category_names =[ "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",  "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                  "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                  "scissors", "teddy bear", "hair drier", "toothbrush" ]

colormap = [
(229, 68, 68),
(229, 91, 68),
(229, 114, 68),
(229, 137, 68),
(229, 160, 68),
(229, 183, 68),
(229, 206, 68),
(222, 229, 68),
(199, 229, 68),
(176, 229, 68),
(153, 229, 68),
(130, 229, 68),
(107, 229, 68),
(84, 229, 68),
(68, 229, 79),
(68, 229, 102),
(68, 229, 125),
(68, 229, 148),
(68, 229, 171),
(68, 229, 194),
(68, 229, 217),
(68, 215, 229),
(68, 192, 229),
(68, 169, 229),
(68, 146, 229),
(68, 123, 229),
(68, 100, 229),
(68, 77, 229),
(91, 68, 229),
(114, 68, 229),
(137, 68, 229),
(160, 68, 229),
(183, 68, 229),
(206, 68, 229),
(229, 68, 222),
(229, 68, 199),
(229, 68, 176),
(229, 68, 153),
(229, 68, 130),
(229, 68, 107),
(229, 68, 84),
(229, 68, 68),
(229, 91, 68),
(229, 114, 68),
(229, 137, 68),
(229, 160, 68),
(229, 183, 68),
(229, 206, 68),
(222, 229, 68),
(199, 229, 68),
(176, 229, 68),
(153, 229, 68),
(130, 229, 68),
(107, 229, 68),
(84, 229, 68),
(68, 229, 79),
(68, 229, 102),
(68, 229, 125),
(68, 229, 148),
(68, 229, 171),
(68, 229, 194),
(68, 229, 217),
(68, 215, 229),
(68, 192, 229),
(68, 169, 229),
(68, 146, 229),
(68, 123, 229),
(68, 100, 229),
(68, 77, 229),
(91, 68, 229),
(114, 68, 229),
(137, 68, 229),
(160, 68, 229),
(183, 68, 229),
(206, 68, 229),
(229, 68, 222),
(229, 68, 199),
(229, 68, 176),
(229, 68, 153),
(229, 68, 130),
(229, 68, 107),
(229, 68, 84),
(229, 68, 68)
]


def main():
    filename = sys.argv[1]
    img_id = int(sys.argv[2])
    
    caption_anno = COCO(filename)
    img_info = caption_anno.loadImgs(img_id)
    #cap_ids = caption_anno.getAnnIds(img_id)
    #cap_infos = caption_anno.loadAnns(cap_ids)
    #print(img_info)
    #print(cap_infos)

    ins_anno   = COCO(filename)
    inst_ids   = ins_anno.getAnnIds(img_id)
    inst_infos = ins_anno.loadAnns(inst_ids)

    print(img_info)
    
    img_h = img_info[0]['height']
    img_w = img_info[0]['width']
    
    # COCOのBOX指定は, 左上座標と幅と高さ. 実画像上の座標  (lt_x, lt_y, w ,h)
    # YOLOのBOX指定は, BOX中心座標と幅と高さ.  正規化したもの  ( c_x,  c_y, w, h)
    # c_x = lt_x + w / 2, c_x = lt_x + w / 2, 
    for inst_info in inst_infos:
        lt_x = inst_info['bbox'][0]/img_w
        lt_y = inst_info['bbox'][1]/img_h
        w    = inst_info['bbox'][2]/img_w
        h    = inst_info['bbox'][3]/img_h
        c_x  = lt_x + w/2
        c_y  = lt_y + h/2
        print(map_category[str(inst_info['category_id'])]-1," ",
              c_x, " ", c_y, " ", w, " ", h, " ")

    cocoDataFolder = os.path.dirname(os.path.dirname(filename))
    coco_url = urllib.parse.urlparse(img_info[0]['coco_url'])
    #print(coco_url)
    #print(cocoDataFolder)

    actual_image_filepath = cocoDataFolder+coco_url.path
    print(actual_image_filepath)
    img = cv2.imread(actual_image_filepath)
    for inst_info in inst_infos:
        pt1 = (int(inst_info['bbox'][0]), int(inst_info['bbox'][1]))
        pt2 = (int(inst_info['bbox'][0]+inst_info['bbox'][2]),
               int(inst_info['bbox'][1]+inst_info['bbox'][3]))
        cv2.rectangle(img, pt1, pt2, colormap[map_category[str(inst_info['category_id'])]-1], 2)
        
    cv2.imshow(coco_url.path, img)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main()
