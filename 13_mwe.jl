using CuArrays, CUDAnative, CUDAdrv
import Flux: cpu

CuArrays.allowscalar(false)

x = cu(rand(8))

function kernel_for_loop!(x)
    index = 2*((blockIdx().x - 1) * blockDim().x + threadIdx().x)-1
    stride = 2*(blockDim().x * gridDim().x)

    offset = 0
    while offset <= 1
        sync_threads()
        offset += 1
    end
    return nothing
end


function run_kernel_for_loop!(x)
    numblocks = ceil(Int, length(x)/256)
    @cuda threads=256 blocks=numblocks kernel_for_loop!(x)
end

@time run_kernel_for_loop!(x)

@device_code_warntype run_kernel_for_loop!(x)

function kernel_no_loop!(x)
    index = 2*((blockIdx().x - 1) * blockDim().x + threadIdx().x)-1
    stride = 2*(blockDim().x * gridDim().x)
    sync_threads()
    sync_threads()

    return nothing
end
function run_kernel_no_loop!(x)
    numblocks = ceil(Int, length(x)/256)
    @cuda threads=256 blocks=numblocks kernel_no_loop!(x)
end

run_kernel_no_loop!(x)
