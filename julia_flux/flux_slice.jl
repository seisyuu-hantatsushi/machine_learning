import Flux, AMDGPU

import ONNXNaiveNASflux,ONNX


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
#=
struct PatchDownsample
end

Flux.Functors.@functor PatchDownsample

function PatchDownsample(w::AbstractArray{T,N}) where {T,N}
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

function PatchDownsample(probes::AbstractProbe...)
    p = probes[1] # select any probe
    optype = "MyOpType"
    # Naming strategy (e.g. how to avoid duplicate names) is provided by the probe
    # Not strictly needed, but the onnx model is basically corrupt if duplicates exist
    nodename = recursename(optype, nextname(p))

    # Add ONNX node info
    add!(p, ONNX.NodeProto(
        # Names of input is provided by probes. This is why new probes need to be provided as output
        input = collect(name.(probes)),
        # Name of output from this node
        output = [nodename],
        op_type = optype))

    # Probes can procreate like this
    return newfrom(p, nodename, s -> s)
end
=#

struct Slice
    starts::AbstractVector{<:Integer}
    ends::AbstractVector{<:Integer}
    axes::Union{Nothing,AbstractArray{<:Integer}}
    steps::Union{Nothing,AbstractArray{<:Integer}}
end

Flux.Functors.@functor Slice

#=
import ONNXNaiveNASflux: AbstractProbe, recursename, nextname, newfrom, add!, name
function (m::Slice)(probes::ONNXNaiveNASflux.AbstractProbe...)

    p = probes[1]
    optype = "Slice"
    nodename = ONNXNaiveNASflux.recursename(optype, ONNXNaiveNASflux.nextname(p))

    attributes = ONNX.AttributeProto[
        ONNX.AttributeProto(
            name = "starts",
            ints = Int64.(m.starts),
        ),
        ONNX.AttributeProto(
            name = "ends",
            ints = Int64.(m.ends),
        ),
    ]

    if m.axes !== nothing
        ONNXNaiveNASflux.push!(attributes, ONNX.AttributeProto(name = "axes", ints = Int64.(m.axes)))
    end

    if m.steps !== nothing
        ONNXNaiveNASflux.push!(attributes, ONNX.AttributeProto(name = "steps", ints = Int64.(m.steps)))
    end

    ONNXNaiveNASflux.add!(
        p,
        ONNX.NodeProto(
            input = collect(ONNXNaiveNASflux.name.(probes)),
            output = [nodename],
            op_type = optype,
            attribute = attributes,
        )
    )

    return ONNXNaiveNASflux.newfrom(p, nodename, identity)

end
=#

function Slice(starts::AbstractArray{<:Integer},
               ends::AbstractArray{<:Integer};
               axes::Union{Nothing,AbstractArray{<:Integer}}=nothing,
               steps::Union{Nothing,AbstractArray{<:Integer}}=nothing)
    
    n = length(starts)
    length(ends) == n || throw(ArgumentError("`starts` and `ends` must match"))

    axes = axes === nothing ? collect(1:n) : collect(axes)
    length(axes) == n || throw(ArgumentError("`axes` must match `starts`"))

    steps = steps === nothing ? fill(1, n) : collect(steps)
    length(steps) == n || throw(ArgumentError("`steps` must match `starts`"))

    println("slice construction: ", starts, ",", ends, ",", axes, ",", steps)
    
    return Slice(starts, ends, axes, steps)

end

function (m::Slice)(x::AbstractArray{T,N}) where {T,N}

    r = ndims(x)
    axes = Vector{Int}()
    dims = Vector{Int}()
    #starts = Vector{Int}(1, r)
    starts = fill(Int(1),r)
    ends = Vector{Int}()
    steps = fill(Int(1),r)

    for i in 1:r
        dim = size(x, i)
        append!(axes,   i)
        append!(ends,   dim)
        append!(dims,   dim)
    end

    for (a, s, e, st) in zip(m.axes, m.starts, m.ends, m.steps)
        if a in axes
            starts[a] = s <= 0 ? s + dims[a] : s
            ends[a]   = e <= 0 ? e + (dims[a]-1) : e
            steps[a]  = st
            if st > 0
                starts[a] = clamp(starts[a], 1, dims[a])
                ends[a]   = clamp(ends[a],   1, dims[a])
            else
                starts[a] = clamp(starts[a], 0, dims[a]-1)
                ends[a]   = clamp(ends[a],   0, dims[a]-1)
            end
        end
    end

    function idxs(axes, starts, ends, steps)
        idxs = Vector{Any}(undef, length(axes))
        fill!(idxs, Colon())
        #println(ndims(axes))
        #println(zip(starts, ends, steps));
        for (a, s, e,st) in zip(axes, starts, ends, steps)
            idxs[a] = s:st:e 
        end
        return idxs
    end
    
    idxs = idxs(axes, starts, ends, steps)

    return  x[idxs...]

end

function (m::Slice)(probes::ONNXNaiveNASflux.AbstractProbe...)

    p = probes[1]
    optype = "Slice"
    nodename = ONNXNaiveNASflux.recursename(optype, ONNXNaiveNASflux.nextname(p))

    starts_node_name = nodename * "_starts"
    ONNXNaiveNASflux.add!(
        p,
        ONNXNaiveNASflux.BaseOnnx.TensorProto(
            name      = starts_node_name,
            data_type = Int(ONNXNaiveNASflux.BaseOnnx.TensorProto_DataType.INT64),
            dims      = Int64[length(m.starts)],
            int64_data = m.starts .- 1))


    ends_node_name = nodename * "_ends"
    ONNXNaiveNASflux.add!(
        p,
        ONNXNaiveNASflux.BaseOnnx.TensorProto(
            name      = ends_node_name,
            data_type = Int(ONNXNaiveNASflux.BaseOnnx.TensorProto_DataType.INT64),
            dims      = Int64[length(m.ends)],
            int64_data = m.ends .- 1))

    inputs = [ONNXNaiveNASflux.name(p), starts_node_name, ends_node_name]

    if m.axes !== nothing
        ndims = length(p.shape)
        axes_node_name = nodename * "_axes"
        axes = ndims .- m.axes 
        ONNXNaiveNASflux.add!(
            p,
            ONNXNaiveNASflux.BaseOnnx.TensorProto(
                name      = axes_node_name,
                data_type = Int(ONNXNaiveNASflux.BaseOnnx.TensorProto_DataType.INT64),
                dims      = Int64[length(axes)],
                int64_data = axes))
        push!(inputs, axes_node_name)
    end

    if m.steps !== nothing
        steps_node_name = nodename * "_steps"
        ONNXNaiveNASflux.add!(
            p,
            ONNXNaiveNASflux.BaseOnnx.TensorProto(
                name      = steps_node_name,
                data_type = Int(ONNXNaiveNASflux.BaseOnnx.TensorProto_DataType.INT64),
                dims      = Int64[length(m.steps)],
                int64_data = m.steps))
        push!(inputs, steps_node_name)
    end

    function fshape(s)
        fs = collect(s)
        for (a,st) in zip(m.axes, m.steps)
            fs[a] = Int(s[a]/st)
        end
        return Tuple(fs)        
    end
    
    ONNXNaiveNASflux.add!(
        p,
        ONNXNaiveNASflux.BaseOnnx.NodeProto(
            #input = collect(ONNXNaiveNASflux.name.(probes)),
            name  = nodename,
            input = inputs,
            output = [nodename],
            op_type = optype,
        )
    )

    return ONNXNaiveNASflux.newfrom(p,
                                    nodename,
                                    fshape)

end

if abspath(PROGRAM_FILE) == @__FILE__

    x = [ 1 2 3 4;
          5 6 7 8 ]

    axes = [1, 2]
    starts = [2, 1]
    ends = [3, 4]
    steps = [1, 2]
    result =  [ 5 7 ]

    #idxs = (2:1:2,:)
    #y = x[idxs...]
    #println(y[:,1:2:4])

    #println(x[2:1:2,1:2:4])
    
    model = Slice(starts, ends; axes=axes, steps=steps)
    y = model(x)
    @assert y == result
    
    #println(x[1:1:1,2:1:4])

    starts = [1, 2]
    ends = [0, 1000]
    result = [ 2 3 4 ]
    model = Slice(starts, ends)
    y = model(x)
    @assert y == result

    x = [ 11 12 13 14;
          21 22 23 24;
          31 32 33 34;
          41 42 43 44; ]

    result = [11 13; 31 33]    
    starts = [1, 1]
    ends   = [1000, 1000]
    steps  = [2, 2]
    model = Slice(starts, ends; steps=steps)
    y = model(x)
    @assert y == result

    result = [12 14; 32 34]
    starts = [1, 2]
    ends   = [1000, 1000]
    steps  = [2, 2]
    model = Slice(starts, ends; steps=steps)
    y = model(x)
    @assert y == result


    result = [21 23; 41 43]
    starts = [2, 1]
    ends   = [1000, 1000]
    steps  = [2, 2]
    model = Slice(starts, ends; steps=steps)
    y = model(x)
    @assert y == result

    result = [22 24; 42 44]
    starts = [2, 2]
    ends   = [1000, 1000]
    steps  = [2, 2]
    model = Slice(starts, ends; steps=steps)
    y = model(x)
    @assert y == result



end
