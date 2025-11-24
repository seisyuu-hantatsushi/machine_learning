import FileIO
import Images,ImageView
import Flux, AMDGPU
import ONNXNaiveNASflux

#include("yolox.jl")
include("flux_slice.jl")

# FluxはHWCNの並び

function detect(img_file::String)

    println("AMDGPU.functional()=",AMDGPU.functional())

    model = Flux.Chain(Slice([1,1],[1000,1000]; axes=[1,1], steps=[2,2]))

    y = model(images)
    
    input_shape = (640, 640, 3, :Batch)
    ONNXNaiveNASflux.save("yolox_proto.onnx", model, input_shape)
    
    #img = FileIO.load(img_file)
    #ImageView.imshow(img)
end

detect("/home/kaz/work/COCOData/train2017/000000000009.jpg")
    
