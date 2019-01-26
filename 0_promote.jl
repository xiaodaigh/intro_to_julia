import Base: promote_rule, convert, *, +

abstract type Distance end

struct KM <: Distance
    value::Float64
end

struct Mile <: Distance
   value::Float64
end

const km = KM(1)
const mile = Mile(1)

+(a::KM, b::KM) =
    KM(a.value  + b.value)

+(a::Mile, b::Mile) =
    Mile(a.value  + b.value)

function *(a::Number, d::D) where D <: Distance
   D(d.value * a)
end

10km + 20km


1mile + 2mile


convert(::Type{KM},  t::Mile) =
    KM(1.60934*t.value)

convert(::Type{Mile}, t::KM)  =
    Mile(0.621371*t.values)

promote_rule(::Type{Mile}, ::Type{KM}) =
    KM

+(a::Distance, b::Distance) =
    +(promote(a,b)...)

1km + 1mile




1km + 2mile


x = rand(10_000_000)

xkm = KM.(x)


y = rand(10_000_000)
ymile = Mile.(y)

@time xkm .+ ymile
@time xkm .+ ymile

@time x .+ y
@time x .+ y

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using CuArrays
CuArrays.allowscalar(true)
# Classify MNIST digits with a convolutional network

imgs = MNIST.images()

labels = onehotbatch(MNIST.labels(), 0:9)

X = cat(float.(imgs)..., dims = 4) |> gpu
tY = labels |> gpu


# Partition into batches of size 32
train = [(cat(float.(imgs[i])..., dims = 4), labels[:,i])
         for i in partition(1:60_000, 32)]

train = gpu.(train)

# Prepare test set (first 1,000 images)
tX = cat(float.(MNIST.images(:test))..., dims = 4) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

m = Chain(
    Conv((3, 3), 1=>32, relu),
    Conv((3, 3), 32=>32, relu),
    x -> maxpool(x, (2,2)),
    Conv((3, 3), 32=>16, relu),
    x -> maxpool(x, (2,2)),
    Conv((3, 3), 16=>10, relu),
    x -> reshape(x, :, size(x, 4)),
    Dense(90, 10), softmax) |> gpu

m(train[1][1])

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

evalcb = throttle(() -> @show(accuracy(tX, tY)), 10)
opt = ADAM()

Flux.train!(loss, params(m), train, opt, cb = evalcb)

accuracy(tX,tY)




using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using CuArrays

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels()
# One-hot-encode the labels
Y = onehotbatch(labels, 0:9) |> gpu

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

dataset = repeated((X, Y), 200)
evalcb = () -> @show(loss(X, Y))
opt = ADAM()

Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))

accuracy(X, Y)

# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

accuracy(tX, tY)
