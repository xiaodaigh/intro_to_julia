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

# accuracy of 99% achievable
m = Chain(
    Conv((2, 2), 1=>64, relu),
    x -> maxpool(x, (2,2)),
    Conv((2, 2), 64=>64, relu),
    x -> maxpool(x, (2,2)),
    Conv((2, 2), 64=>64, relu),
    x -> reshape(x, :, size(x, 4)),
    #IdentitySkip(Dense(800, 800, relu)),
    Dense(1600, 256, relu),
    #IdentitySkip(Dense(128, 128, relu)),
    Dense(256, 10),
    softmax) |> gpu

m(train[1][1])

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold((cpu(m))(x)) .== onecold(y))

evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
opt = ADAM()

@time Flux.@epochs 6 Flux.train!(loss, params(m), train, opt, cb = evalcb)
@time evalcb()
