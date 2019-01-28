using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using CuArrays # arrays that work on GPU
using ColorTypes
# load the images
imgs = MNIST.images()

# images
raw_labels = MNIST.labels()
labels = onehotbatch(raw_labels, 0:9)

import Base.Iterators: product

# function to create average images

create_avg_img_diff(digit) = begin
   imgs0 = imgs[raw_labels .== digit]
   avg_imgs0 = reduce(+, float.(imgs0))./length(imgs0)
end

avg_images = reshape(reduce(hcat, vec.(create_avg_img_diff.(0:9))), 28, 28, 10)

imgs_test = MNIST.images(:test)
raw_labels_test = MNIST.labels(:test)
labels_test = onehotbatch(raw_labels_test, 0:9)

compute_similarity_to_avg(avg_images, test_img) = begin
   errs = [sum((test_img .- avg_images[:,:,i]).^2) for i=1:10]
   findmin(errs)[2]-1
end

imgs_test_float = float.(imgs_test)

@time compute_similarity_to_avg(avg_images, imgs_test_float[1])
#@code_warntype compute_similarity_to_avg(avg_images, imgs_test_float[1])

@time predictions = compute_similarity_to_avg.(Ref(avg_images), imgs_test_float)

sum(predictions .== raw_labels_test)/length(raw_labels_test)

# which digits do I get wrong?
import Lazy: @>
import StatsBase: countmap

@> begin
   zip(predictions, raw_labels_test)
   collect
   x->filter(elt -> elt[1] != elt[2], x)
   collect
   countmap
   collect
   x->sort(x, by = y->y[2], rev = true)
end
