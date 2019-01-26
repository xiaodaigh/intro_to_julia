
x = rand(Float32, 1000)

create_new_vec(x) = begin
   n = length(x)
   new_vec = typeof(x)(undef, Int((n*n-n)/2) + n)
   upto = 0
   @inbounds for i=1:n
      for j=i:n
         upto += 1
         new_vec[upto] = x[i]*x[j]
      end
   end
   new_vec
end

@time new_vec = create_new_vec(x)

using CuArrays
xcu = cu(x)

@time create_new_vec(xcu)

ok

using ASTInterpreter2

function foo(n)
    x = n+1
    ((BigInt[1 1; 1 0])^x)[2,1]
end

@enter foo(20)

# m = Chain(
#     Dropout(0.5),
#     Conv((5, 5), 1=>20, relu),
#     Dropout(0.5),
#     Conv((5, 5), 20=>1, relu),
#     x->reshape(x, :, size(x, 4)),
#     Dropout(0.5),
#     Dense(400, 10),
#     softmax) |> gpu
#

# m = Chain(
#     Conv((3, 3), 1=>32, relu),
#     x -> maxpool(x, (2,2))
#     )
#
#  m = Chain(
#      Dropout(0.1),
#      Conv((3, 3), 1=>32, relu),
#      Dropout(0.1),
#      x -> maxpool(x, (2,2)),
#      Dropout(0.1),
#      Conv((3, 3), 32=>16, relu),
#      Dropout(0.1),
#      x -> maxpool(x, (2,2)),
#      Dropout(0.1),
#      Conv((3, 3), 16=>10, relu),
#      Dropout(0.1),
#      x -> reshape(x, :, size(x, 4)),
#      Dropout(0.1),
#      Dense(90, 10), softmax) |> gpu
#
#  # this achieves almost 99%
#  # m = Chain(
#  #    Conv((2, 2), 1=>32, relu),
#  #    x -> maxpool(x, (2,2)),
#  #    Conv((2, 2), 32=>16, relu),
#  #    x -> maxpool(x, (2,2)),
#  #    Conv((2, 2), 16=>10, relu),
#
# struct IdentitySkip
#    inner
# end
#
# (m::IdentitySkip)(x) = m.inner(x) .+ x

struct DenseSkip{F,S,T}
  W::S
  b::T
  σ::F
end

DenseSkip(W, b) = DenseSkip(W, b, identity)

import Flux: glorot_uniform, @treelike

function DenseSkip(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros)
  return DenseSkip(param(initW(out, in)), param(initb(out)), σ)
end

function (a::DenseSkip)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  #σ.(W*x .+ b) .+ x
  σ.(W*x .+ b)
end

#@treelike DenseSkip

m = Chain(
    #Dense(100, 100, relu)
    #,
    DenseSkip(100, 100)
    ) |> gpu

x = rand(Float32, 100) |> gpu

m(x)

# lenet 5
# https://engmrk.com/lenet-5-a-classic-cnn-architecture/
# m = Chain(
#     #Dropout(0.5),
#     Conv((5, 5), 1=>6, relu),
#     x -> maxpool(x, (2,2)),
#     Conv((3, 3), 32=>32, relu),
#     x -> maxpool(x, (2,2)),
#     Conv((3, 3), 32=>16, relu),
#     x -> maxpool(x, (2,2)),
#     Conv((3, 3), 16=>10, relu),
#     x -> reshape(x, :, size(x, 4)),
#     Dense(90, 10), softmax) |> gpu


m(train[1][1])

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))


opt = ADAM()

evalcb = throttle(() -> @show(accuracy(tX, tY)), 5)
@time Flux.@epochs 40 Flux.train!(loss, params(m), train, opt, cb = evalcb)
accuracy(tX, tY)


using Distributions
using Flux.Tracker: gradient, param, Params
using Flux.Optimise: Descent, ADAM, update!

D = 2
num_samples = 100

function log_density(params)
   mu, log_sigma = params
   d1 = Normal(0, 1.35)
   d2 = Normal(0, exp(log_sigma))
   d1_density = logpdf(d1, log_sigma)
   d2_density = logpdf(d2, mu)
   return d1_density + d2_density
end


function J(log_std)
   H = 0.5 * D * (1.0 + log(2 * pi)) + sum(log_std)
   return H
end

function objective(mu, log_std; D=2)
   samples = rand(Normal(), num_samples, D) .* sqrt.(log_std) .+ mu
   log_px = mapslices(log_density, samples; dims=2)
   elbo = J(log_std) + mean(log_px)
   return -elbo
end

mu = param(reshape([-1, -1], 1, :))
sigma = param(reshape([5, 5], 1, :))

grads = gradient(() -> objective(mu, sigma), Params([mu, sigma]))

opt = Descent(0.001)

update!(mu, -0.001.*grads[mu])
update!(sigma, -0.001.*grads[sigma])


for p in (mu, sigma)
   update!(opt, p, grads[p])
end
