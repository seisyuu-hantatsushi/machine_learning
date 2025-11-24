
using Flux

include("flux_slice.jl")

Focus = Chain(Parallel((xs...)->cat(xs...; dims = 3), 
                       Slice([1,1],[1000,1000]; steps=[2,2]),
                       Slice([1,2],[1000,1000]; steps=[2,2]),
                       Slice([2,1],[1000,1000]; steps=[2,2]),
                       Slice([2,2],[1000,1000]; steps=[2,2])),
              Conv((3,3),12=>80; stride=1, pad=1),
              x -> x .* sigmoid(x))

function self_gated_block(kernel::NTuple{N,Integer}, io::Pair{<:Integer,<:Integer};
                          stride=1, pad=1) where N
    return Chain(Conv(kernel, io; stride=stride, pad=pad),
                 x -> x .* sigmoid(x))
end

function yolox()
    model = Chain(Focus,
                  self_gated_block((3,3), 80=>160; stride=2, pad=1),
                  Parallel((xs...)->cat(xs...; dims = 3),
                           self_gated_block((1,1), 160=>80; stride=1, pad=0),
                           self_gated_block((1,1), 160=>80; stride=1, pad=0)))
    return model
end
