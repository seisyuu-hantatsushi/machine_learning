import Flux, AMDGPU
import ONNXNaiveNASflux

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
            steps[a]  = m.steps[a]
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
        println(ndims(axes))
        println(zip(starts, ends, steps));
        for (a, s, e,st) in zip(axes, starts, ends, steps)
            idxs[a] = s:st:e 
        end
        return idxs
    end
    
    idxs = idxs(axes, starts, ends, steps)

    return  x[idxs...]

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
    println("1:",y)

    
    #println(x[1:1:1,2:1:4])

    starts = [1, 2]
    ends = [0, 1000]
    result = [ 2 4 7 ]
    model = Slice(starts, ends)
    y = model(x)
    println("2:", y)

    x = [ 11 12 13 14;
          21 22 23 24;
          31 32 33 34;
          41 42 43 44; ]

    starts = [1, 1]
    ends   = [1000, 1000]
    steps  = [2, 2]
    model = Slice(starts, ends; steps=steps)
    y = model(x)
    println("3:", y)

    starts = [2, 2]
    ends   = [1000, 1000]
    steps  = [2, 2]
    model = Slice(starts, ends; steps=steps)
    y = model(x)
    println("4:", y)
end
