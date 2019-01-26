using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using CuArrays # arrays that work on GPU
using ColorTypes
# load the images
const imgs = MNIST.images()

# images
const raw_labels = MNIST.labels()
const labels = onehotbatch(raw_labels, 0:9)

import Base.Iterators: product

# function to create average images

avg_img_diff(digit) = begin
   imgs0 = imgs[raw_labels .== digit]
   avg_imgs0 = reduce(+, float.(imgs0))./length(imgs0)

   imgs_not0 = imgs[raw_labels .!= digit]
   avg_imgs_not0 = reduce(+, float.(imgs_not0))./length(imgs_not0)

   diffs = reshape((avg_imgs0 .- avg_imgs_not0).^2, 28, 28)

   ordered_diffs = sort(vec([(i,j, diffs[i,j], avg_imgs0[i,j]) for (i,j) in product(1:28, 1:28)]), by = x->x[3], rev=true)

   #i2 = copy(imgs[2])
   i2 = Gray.(avg_imgs0)
   i2 .= float.(i2)./2

   for (i,j,k,l) in ordered_diffs[1:10]
      i2[i,j] = 1
   end
   i2

   (avg_imgs0, avg_imgs_not0, ordered_diffs, i2)
end

avg_img_diffs = (x->x[1])(avg_img_diff.(0:9))

avg_img_diffs1 = reduce(x->x)

avg_img_diffs[8][4]

const imgs_test = MNIST.images(:test)

# images
const raw_labels_test = MNIST.labels(:test)
const labels_test = onehotbatch(raw_labels_test, 0:9)

imgs_test[1]
test_img = float.(imgs_test[1])

n_pixels = 1

digit = 8

compute_similarity_to_avg(digit, test_img, n_pixels) = begin
   diffs = avg_img_diffs[digit+1][3][1:n_pixels]
   #avg_img1 = avg_img_diffs[digit+1][1]

   # err = 0.0
   # for (i,j,_) in diffs
   #    err += (test_img[i,j] - avg_img1[i,j])^2
   # end
   #
   # err
end

@code_warntype compute_similarity_to_avg(0, imgs_test_float[1], 1)

compute_similarity_to_all_avg(test_img, n_pixels) = compute_similarity_to_avg.(0:9, Ref(test_img), n_pixels)

compute_similarity_to_avg(0, test_img, 1)

compute_similarity_to_all_avg(test_img, 9)

const imgs_test_float = float.(imgs_test)
#@time predicted_label = [findmin(compute_similarity_to_all_avg(it, n_pixels))[2]-1 for (n_pixels, it) in product(1:100, imgs_test_float)]

#[sum(predicted_label[i,:] .== raw_labels_test)/length(raw_labels_test) for i in 1:100]
pred(imgs_test_float) = begin
   findmin(compute_similarity_to_all_avg(imgs_test_float, 728))[2]-1
end

@time pred(imgs_test_float[1])

@code_warntype compute_similarity_to_all_avg(imgs_test_float[1])

@code_warntype  pred(imgs_test_float[1])

@time predicted_label = [findmin(compute_similarity_to_all_avg(it, 728))[2]-1 for it in imgs_test_float]
sum(predicted_label .== raw_labels_test)/length(raw_labels_test)

using StatsBase, Lazy

zip(predicted_label, raw_labels_test) |>
   collect |>
   countmap |>
   x -> filter(x) do x
      x[1][1] != x[1][2]
   end |>
   collect |>
   x -> sort(x, by = y->y[2], rev = true)
