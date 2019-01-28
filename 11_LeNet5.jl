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

trainX = reshape(reduce(hcat, vec.(float.(MNIST.images()))),28,28,1,60_000)
trainY = onehotbatch(MNIST.labels(), 0:9)

struct IdentitySkip
   inner
end

(m::IdentitySkip)(x) = m.inner(x) .+ x

import Flux: @treelike
@treelike IdentitySkip

import Flux: testmode!
# classic LeNet https://engmrk.com/lenet-5-a-classic-cnn-architecture/
m = Chain(
    Conv((5, 5), 1=>16, relu),
    x -> maxpool(x, (2,2)),
    Dropout(0.5),
    Conv((5, 5), 16=>32, relu),
    x -> maxpool(x, (2,2)),
    x -> reshape(x, : , size(x, 4)),
    Dropout(0.5),
    Dense(512, 256, relu),
    Dropout(0.5),
    Dense(256, 10),
    softmax
    ) |> gpu

m(train[1][1])

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = begin
    copym = m
    testmode!(copym)
    a = mean(onecold((cpu(copym))(x)) .== onecold(y))
    testmode!(copym, false)
    a
end

evalcb = throttle(() -> @show((training = accuracy(trainX, trainY), test = accuracy(tX, tY))), 10)
opt = ADAM()

@time Flux.@epochs 80 Flux.train!(loss, params(m), train, opt, cb = evalcb)
# can achieve about 99.3%
@time evalcb()
