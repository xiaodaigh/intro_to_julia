using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using CuArrays
# Classify MNIST digits with a convolutional network

imgs = MNIST.images()

labels = onehotbatch(MNIST.labels(), 0:9)

# Partition into batches of size 32
train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i])
         for i in partition(1:60_000, 32)]

train = gpu.(train)

# Prepare test set (first 1,000 images)
#tX = cat(float.(MNIST.images(:test)[1:1000])..., dims = 4) |> gpu
tX = reshape(reduce(hcat, vec.(float.(MNIST.images(:test)))),28,28,1,10_000)
tY = onehotbatch(MNIST.labels(:test), 0:9)

trainX = reshape(reduce(hcat, vec.(float.(MNIST.images()))),28,28,1,60_000) |> gpu
trainY = onehotbatch(MNIST.labels(), 0:9) |> gpu

 struct IdentitySkip
    inner
 end

(m::IdentitySkip)(x) = m.inner(x) .+ x

import Flux: @treelike
@treelike IdentitySkip

m = Chain(
    Conv((5, 5), 1=>2, relu),
    x -> reshape(x, :, size(x, 4)),
    Dense(1152, 128, relu),
    Dense(128, 10),
    softmax) |> gpu

m(train[1][1])

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold((cpu(m))(x)) .== onecold(y))

evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
opt = ADAM()

@time Flux.@epochs 10 Flux.train!(loss, params(m), train, opt, cb = evalcb)
@time evalcb()

diag_filter = Array{Float32, 2}(undef, 5, 5)

diag_filter .= 0.0

[diag_filter[i,i] = 1.0 for i in 1:5]

using ColorTypes
a = Gray.(diag_filter)

a

imgs[1]

imgs1 = imgs[1]
conv(x, filterm) = sum(x.*filterm)/5

a

imgs1_filterdd = [conv(float.(imgs1[i:i+4,j:j+4]),diag_filter)
    for i=1:size(imgs1,1)-size(diag_filter,1)+1, j=1:size(imgs1,2)-size(diag_filter,2)+1]

Gray.(imgs1_filterdd)
