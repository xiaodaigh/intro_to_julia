using CuArrays, CUDAnative, CUDAdrv
import Flux: cpu

CuArrays.allowscalar(false)

x = cu(rand(8))

# see
# https://www.youtube.com/watch?v=w544Rn4KC8I
# for an explanation of bitonic sort

function bitonic!(x)
    index = 2*((blockIdx().x - 1) * blockDim().x + threadIdx().x)-1
    stride = 2*(blockDim().x * gridDim().x)

    for i in index:stride:length(x)-1
        if x[i] > x[i+1]
            x[i], x[i + 1] = x[i+1], x[i]
        end
    end
    sync_threads()

    return nothing
end


function bench_bitonic!(x)
    numblocks = ceil(Int, length(x)/256)
    @cuda threads=256 blocks=numblocks bitonic!(x)
end

xx = cpu(x); issorted(xx)
@time bench_bitonic!(x)
xx = cpu(x); issorted(xx)


#
x = rand(Float32, 100_000_000)

@time sum(x)

using CuArrays
cux = cu(x)

@time sum(cux[1:1])
@time sum(cux)


# Broadcasting: apply this element-wise
f(x) = x/(1+x^2)
@time f.(x .+ 2sqrt.(x))
@time f.(cux .+ 2sqrt.(cux))
