import sys
from pycocotools.coco import COCO

import pandas as pd
from sklearn.cluster import KMeans

map_category = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}

first_128_listed_id = [  9,   25,   30,  34,  36,  42,  49,  61,
                         64,  71,   72,  73,  74,  77,  78,  81,
                         86,  89,   92,  94, 109, 110, 113, 127,
                         133, 136, 138, 142, 143, 144, 149, 151,
                         154, 164, 165, 192, 194, 196, 201, 208,
                         241, 247, 257, 260, 263, 283, 294, 307,
                         308, 309, 312, 315, 321, 322, 326, 328,
                         332, 338, 349, 357, 359, 360, 368, 370,
                         382, 384, 387, 389, 394, 395, 397, 400,
                         404, 415, 419, 428, 431, 436, 438, 443,
                         446, 450, 459, 471, 472, 474, 486, 488,
                         490, 491, 502, 510, 514, 520, 529, 531,
                         532, 536, 540, 542, 544, 560, 562, 564,      
                         569, 572, 575, 581, 584, 589, 590, 595,
                         597, 599, 605, 612, 620, 623, 625, 626,
                         629, 634, 636, 641, 643, 650, 656, 659  ]

def calc_anchorbox(annotation_file, bfirst_128 = False):
    inst_anno = COCO(annotation_file)
    yolo_bboxes = []
    img_size = 416

    listed_id = first_128_listed_id;
    if(not bfirst_128):
        listed_id = inst_anno.getImgIds()
        #print(listed_id)
    
    for img_id in listed_id:
        img_info = inst_anno.loadImgs(img_id)
        img_h = img_info[0]['height']
        img_w = img_info[0]['width']
        inst_ids   = inst_anno.getAnnIds(img_id)
        inst_infos = inst_anno.loadAnns(inst_ids)
        # COCOのBOX指定は, 左上中心座標と幅と高さ. 実画像上の座標  (lt_x, lt_y, w ,h)
        # YOLOのBOX指定は, BOX中心座標と幅と高さ.  正規化したもの  ( c_x,  c_y, w, h)
        # c_x = lt_x + w / 2, c_x = lt_x + w / 2, 
        for inst_info in inst_infos:
            lt_x = inst_info['bbox'][0]/img_w
            lt_y = inst_info['bbox'][1]/img_h
            w    = inst_info['bbox'][2]/img_w
            h    = inst_info['bbox'][3]/img_h
            c_x  = lt_x + w/2
            c_y  = lt_y + h/2
            yolo_bbox = { 'category_id': map_category[str(inst_info['category_id'])]-1, 'x' : c_x, 'y' : c_y, 'width' : w, 'height': h }
            yolo_bboxes.append(yolo_bbox)

    '''
    for yolo_bbox in yolo_bboxes:
    print(yolo_bbox)
    '''
    # K-mean法によりbboxの大きさを9つにカテゴライズする
    bbox_dict = { 'width' : [], 'height' : [] }
    bbox_list = []
    for yolo_bbox in yolo_bboxes:
        bbox_dict['width'].append(yolo_bbox['width'])
        bbox_dict['height'].append(yolo_bbox['height'])
        bbox_list.append([yolo_bbox['width'], yolo_bbox['height']])
        
    df = pd.DataFrame(bbox_dict)
    km = KMeans(n_clusters = 9,
                init='random',
                n_init = 10,
                max_iter = 300,
                tol=1e-04,
                random_state = 0)
    y_km = km.fit_predict(bbox_list) #各Boxの大きさがどのクラスタに判別されたかが変える
    df['cluster'] = y_km

    # クラスタ別に大きさの平均値をとる
    anchor_dict = {"width":[],"height":[],"area":[]}
    for i in range(9):
        anchor_dict["width"].append(df[df["cluster"] == i].mean()["width"])
        anchor_dict["height"].append(df[df["cluster"] == i].mean()["height"])
        anchor_dict["area"].append(df[df["cluster"] == i].mean()["width"]*df[df["cluster"] == i].mean()["height"])
            
    anchors = pd.DataFrame(anchor_dict).sort_values('area', ascending=False) # 面積を基準に並べ替え
    anchors["scale"] = [int(img_size/32) ,int(img_size/32) ,int(img_size/32) ,
                        int(img_size/16) ,int(img_size/16) ,int(img_size/16) ,
                        int(img_size/8), int(img_size/8), int(img_size/8)]

    for a in anchors.itertuples():
        print("{:>3} {:>3} {:>2}".format(round(a[1]*img_size),round(a[2]*img_size),a[4]))
                
    return anchors

def obtain_anchorbox(annotation_file=None, bfirst_128 = False, bcalc = False):

    if bcalc:
        return calc_anchorbox(annotation_file, bfirst_128)
    else:
        img_size = 416
        calced_anchorboxs = [
            (0.904678, 0.858803, 0.776940),
            (0.499253, 0.811462, 0.405125),
            (0.836040, 0.380303, 0.317949),
            (0.454634, 0.418600, 0.190310),
            (0.232230, 0.650158, 0.150986),
            (0.162652, 0.363501, 0.059124),
            (0.298059, 0.173338, 0.051665),
            (0.101796, 0.167664, 0.017068),
            (0.044272, 0.055500, 0.002457)   
        ]
        anchor_dict = {"width":[],"height":[],"area":[]}
        for a in calced_anchorboxs:
            anchor_dict['width'].append(a[0])
            anchor_dict['height'].append(a[1])
            anchor_dict['area'].append(a[2])

        anchor_dict = pd.DataFrame(anchor_dict)
        anchor_dict["scale"] = [int(img_size/32) ,int(img_size/32) ,int(img_size/32) ,
                                int(img_size/16) ,int(img_size/16) ,int(img_size/16) ,
                                int(img_size/8), int(img_size/8), int(img_size/8)]
        return anchor_dict

def main():

    args = sys.argv
    
    anchor_dict = obtain_anchorbox(args[1],
                                   bfirst_128 = (not (len(args) < 3)),
                                   bcalc = False) 

    print(anchor_dict)
    
if __name__ == '__main__':
    main()
