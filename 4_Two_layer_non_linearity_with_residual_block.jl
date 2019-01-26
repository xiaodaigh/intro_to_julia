using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using CuArrays

# Classify MNIST digits with a GLM
imgs = MNIST.images()
raw_labels = MNIST.labels()
labels = onehotbatch(raw_labels, 0:9)

@time glmtrain = [(reduce(hcat, vec.(float.(imgs[i]))), labels[:,i])
         for i in partition(1:60_000, 32)] .|>
            gpu

struct IdentitySkip
   inner
end

(m::IdentitySkip)(x) = m.inner(x) .+ x

import Flux: @treelike

# tells Flux to look for parameters in the right place
@treelike IdentitySkip

# neural network with layers with non-linearity
glm = Chain(
   IdentitySkip(Dense(28^2, 28^2, relu)),
   Dense(28^2, 10),
   softmax) |> gpu

glm_cpu = cpu(glm)

@time glm(glmtrain[1][1])

glmtrainX = reduce(hcat, vec.(float.(MNIST.images(:test))))
glmtrainY = onehotbatch(MNIST.labels(:test), 0:9)

loss(x, y) = crossentropy(glm(x), y)

accuracy(x, y) = mean(onecold((cpu(glm))(x)) .== onecold(y))

evalcb = () -> @show(accuracy(glmtrainX, glmtrainY))
opt = ADAM()

@time Flux.@epochs 3 Flux.train!(loss, params(glm), glmtrain, opt, cb = throttle(evalcb, 10))
@time evalcb()
