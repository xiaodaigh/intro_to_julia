x = "to be or not to be"

xs = split(x, " ")

res = Dict{String, Int}()

unique_counter = 0

for xs1 in xs
    global unique_counter
    if haskey(res, xs1)
        if res[xs1] == 0
            unique_counter += 1
            res[xs1] = unique_counter
        end
    else
        res[xs1] = 0
    end
end

res

for (i,xs1) in enumerate(xs)
    print(i)
    if res[xs1] > 0
        xs[i] = string(res[xs1])
    end
end

join(xs, " ")


# put all these into a function
compress_str(x) = begin
    xs = split(x, " ")

    res = Dict{String, Int}()

    unique_counter = 0

    for xs1 in xs
        if haskey(res, xs1)
            if res[xs1] == 0
                unique_counter += 1
                res[xs1] = unique_counter
            end
        else
            res[xs1] = 0
        end
    end

    res

    for (i,xs1) in enumerate(xs)
        print(i)
        if res[xs1] > 0
            xs[i] = string(res[xs1])
        end
    end

    join(xs, " ")
end

compress_str("to be or not to be")

compress_str("that is the question")

compress_str("Julia is to Python what Python was to Perl")
