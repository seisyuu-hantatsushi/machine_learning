import sys
import cv2
from PIL import Image

import torch, torch.nn as nn, torch.onnx, torchvision.transforms as T
import torchinfo

import anchorbox

class residual_block(nn.Module):

    def __init__(self, ch, repeat_count=1):
        super().__init__()
        module_list = nn.ModuleList()
        for r in range(repeat_count):
            resblock = nn.ModuleList()
            resblock.append(convolutional_layer(1,    ch, ch//2, stride = 1))
            resblock.append(convolutional_layer(3, ch//2,    ch, stride = 1))
            module_list.append(resblock)
        self.module_list = module_list

    def forward(self, x):
        for m in self.module_list:
            holder = x
            for r in m:
                x = r(x)
            x = x + holder
        return x

def convolutional_layer(kernel_size, input_ch, output_ch, stride=1):
    new_layer = nn.Sequential()
    pad = (kernel_size - 1) // 2
    new_layer.add_module('convolutional',
                         nn.Conv2d(input_ch,
                                   output_ch,
                                   kernel_size,
                                   stride = stride,
                                   padding = pad,
                                   bias = False))
 
    new_layer.add_module('batch_norm',
                         nn.BatchNorm2d(output_ch))
    new_layer.add_module('leaky_relu',
                         nn.LeakyReLU(0.1))
    return new_layer

def build_darknet53(module_list):

                       
    # module_list[0]
    module_list.append(convolutional_layer(kernel_size = 3,
                                           input_ch = 3,
                                           output_ch = 32)) # 1 x 32 x H x W 
    #
    # module_list[1]
    module_list.append(convolutional_layer(kernel_size = 3,
                                           input_ch = 32,
                                           output_ch = 64,
                                           stride = 2)) # 1 x 64 x (H/2) x (W/2) 
    # module_list[2]
    module_list.append(residual_block(64, repeat_count = 1)) # 1 x 64 x (H/2) x (W/2)

    # module_list[3]
    module_list.append(convolutional_layer(kernel_size = 3,
                                           input_ch = 64,
                                           output_ch = 128,
                                           stride = 2))  # 1 x 128 x (H/4) x (W/4)
    # module_list[4]
    module_list.append(residual_block(128, repeat_count = 2)) # 1 x 128 x (H/4) x (W/4)

    # module_list[5]
    module_list.append(convolutional_layer(kernel_size = 3,
                                           input_ch = 128,
                                           output_ch = 256,
                                           stride = 2))  # 1 x 256 x (H/8) x (W/8)
    # module_list[6]
    module_list.append(residual_block(256, repeat_count = 8)) # 1 x 256 x (H/8) x (W/8)

    # module_list[7]
    module_list.append(convolutional_layer(kernel_size = 3,
                                           input_ch = 256,
                                           output_ch = 512,
                                           stride = 2))  # 1 x 512 x (H/16) x (W/16)
    # module_list[8]
    module_list.append(residual_block(512, repeat_count = 8)) # 1 x 512 x (H/16) x (W/16)


    # module_list[9]
    module_list.append(convolutional_layer(kernel_size = 3,
                                           input_ch = 512,
                                           output_ch = 1024,
                                           stride = 2))  # 1 x 1024 x (H/32) x (W/32) 

    # module_list[10]
    module_list.append(residual_block(1024, repeat_count = 4)) # 1 x 1024 x (H/32) x (W/32)

    # scale_1はmodule_list[6]の結果
    # scale_2はmodule_list[8]の結果
    # scale_3はmodule_list[10]の結果
    
    return module_list

class detect_object_net(nn.Module):
        def __init__(self, ch, num_of_anchors, num_of_classes):
            super().__init__()

            module_list = nn.ModuleList()

            for i in range(3):
                module_list.append(convolutional_layer(kernel_size = 1,
                                                       input_ch  = ch,
                                                       output_ch = ch//2,
                                                       stride = 1)) 
                module_list.append(convolutional_layer(kernel_size = 3,
                                                       input_ch  = ch//2,
                                                       output_ch = ch,
                                                       stride = 1)) 
            # (物体の位置とサイズの予測（中心のx座標、中心のy座標、幅、高さ）、物体があるかどうかの信頼度、目標のクラス別の信頼度)
            # なのでBoxの情報として(​4+1+num_of_class)が付随して、アンカーボックの数が,num_of_anchorsとして, num_of_anchors*(4+1+num_of_class)
            # COCOでは3つのBoxが指定されているので,3*(4+1+num_of_class)が結果を収めるのに必要な大きさ
            pred_conv = nn.Conv2d(ch,
                                  num_of_anchors*(num_of_classes + 4 + 1),
                                  1,
                                  stride = 1,
                                  padding = 0,
                                  bias = False)
            upsample = nn.Sequential()
            #print("ch=",ch)
            upsample.add_module('down_conv', convolutional_layer(kernel_size = 1,
                                                                 input_ch = ch*2,
                                                                 output_ch = ch//2,
                                                                 stride = 1))
            upsample.add_module('upsample', nn.Upsample(scale_factor=(2,2), mode="nearest"))
            self.pred_conv   = pred_conv
            self.upsample    = upsample
            self.down_ch     = convolutional_layer(kernel_size = 1,
                                                   input_ch = ch + ch//2,
                                                   output_ch = ch,
                                                   stride = 1)
            self.module_list = module_list
            
        def forward(self, x, upsample=None):

            if upsample != None:
                x = torch.cat((x, self.upsample(upsample)), 1)
                x = self.down_ch(x)

            for m in self.module_list:
                x = m(x)
                
            return self.pred_conv(x),x
            
class YoloV3(nn.Module):

    def __init__(self):
        super(YoloV3, self).__init__()
        module_list = nn.ModuleList()
        module_list = build_darknet53(module_list)
        self.scale3_detect = detect_object_net(1024, 3, 80)
        self.scale2_detect = detect_object_net( 512, 3, 80)
        self.scale1_detect = detect_object_net( 256, 3, 80)
        self.module_list = module_list
        
    def forward(self, x):

        scales = []

        for i, m in enumerate(self.module_list):
            x = m(x)
            if i in [6, 8, 10]:
                scales.append(x)  #結果を格納する
        
        scale3_object_pred, upsample_for_scale2 = self.scale3_detect(scales[2])
        scale2_object_pred, upsample_for_scale1 = self.scale2_detect(scales[1], upsample_for_scale2)
        scale1_object_pred, _                   = self.scale1_detect(scales[0], upsample_for_scale1)

        return [scale3_object_pred, scale2_object_pred, scale1_object_pred]
    
def main():

    input_size = (416, 416)
    args = sys.argv

    #image_file = args[1]

    model = YoloV3()
    model.eval()
    #print(model)

    img_tensor = torch.randn(1, 3, input_size[1], input_size[0]) #NCHW

    #torchinfo.summary(model, input_size=(1, 3, input_size[1], input_size[0]))
    
    #onnx_model = torch.onnx.export(model, img_tensor, "yolov3.onnx", dynamo=True, verbose=True)

    for i in range(9):
        anchor = anchor_dict.iloc[i]
        print(anchor)
        
if __name__ == '__main__':
    main()
