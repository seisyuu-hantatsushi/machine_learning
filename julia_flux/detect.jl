import FileIO
import Images,ImageView
import Flux, AMDGPU
import ONNXNaiveNASflux

#include("yolox.jl")

# In Flux, the dimension order is HWCN.
# FluxはHWCNの並び
#=
      列 (W)
        1    2    3    4
行 1  a11  a12  a13  a14
(H)2  a21  a22  a23  a24
   3  a31  a32  a33  a34
   4  a41  a42  a43  a44
を

      1    2    3    4
  1  ○a11   ×   ○a13   ×
  2    ×    ×    ×    ×
  3  ○a31   ×   ○a33   ×
  4    ×    ×    ×    ×

=>
  [ a11   a13
    a31   a33 ]
とDownSamplingする.
=#

struct PatchDownsample
end

Flux.@functor PatchDownsample
function (m::PatchDownsample)(x)
    # x: (H, W, C, N)
    # 2x2 のパッチをチャンネル方向に詰める
    patch_top_left  = x[1:2:end, 1:2:end, :, :]  # (H/2, W/2, C, N)
    patch_top_right = x[1:2:end, 2:2:end, :, :]
    patch_bot_left  = x[2:2:end, 1:2:end, :, :]
    patch_bot_right = x[2:2:end, 2:2:end, :, :]

    x2 = cat(
        patch_top_left,
        patch_bot_left,
        patch_top_right,
        patch_bot_right;
        dims = 3,          # チャンネル方向 (C) で結合 = 4C チャンネル
    )
    return x2
end

function detect(img_file::String)

    println("AMDGPU.functional()=",AMDGPU.functional())

    model = Flux.Chain(PatchDownsample)
    
    input_shape = (640, 640, 3, :Batch)

    ONNXNaiveNASflux.save("yolox_proto.onnx", model, input_shape)
    
    #img = FileIO.load(img_file)
    #ImageView.imshow(img)
end

detect("/home/kaz/work/COCOData/train2017/000000000009.jpg")
    
