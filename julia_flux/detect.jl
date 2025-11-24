import Images,ImageView,ImageTransformations
import Flux, AMDGPU
import ONNXNaiveNASflux
import FileIO,Gtk4
using ColorVectorSpace 

include("yolox.jl")

# FluxはHWCNの並び

function detect(img_file::String)

    println("AMDGPU.functional()=",AMDGPU.functional())

    img = FileIO.load(img_file)
    println(typeof(img))
    #=
    guidict = ImageView.imshow(img)
    if (!isinteractive())
        
        # Create a condition object
        c = Condition()
        
        # Get the window
        win = guidict["gui"]["window"]
        
        # Start the GLib main loop
        @async Gtk4.GLib.glib_main()
        
        # Notify the condition object when the window closes
        Gtk4.GLib.signal_connect(win, :close_request) do widget
            notify(c)
        end
        
        # Wait for the notification before proceeding ...
        wait(c)
    end
    =#
    resized_img = ImageTransformations.imresize(img, (640, 640))
    println(size(resized_img))

    # to RGB
    img_rgb = Images.RGB.(resized_img)

    x_chw = Images.channelview(img_rgb)              # size: (C, H, W)
    println(size(x_chw))
    x_hwc = Images.permutedims(x_chw, (2, 3, 1)) # size: (H, W, C)
    println(size(x_hwc))
    x_hwcn = reshape(x_hwc, size(x_hwc,1), size(x_hwc,2), size(x_hwc,3), 1) # size: (H, W, C, N)
    println(size(x_hwcn))

    model = yolox()
                       
    y = model(x_hwcn)
    println(size(y))
    
    input_shape = (640, 640, 3, 1)
    ONNXNaiveNASflux.save("yolox_proto.onnx", model, input_shape)

end

detect("/home/kaz/work/COCOData/train2017/000000000009.jpg")
    
