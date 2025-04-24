function BiasContraction(bs, stacked_array)
    b1, b2, b3, b4, b5, b6, b7, f = bs
    biases = Array([ b1^2, 2*b1*f, f^2, 1., b1, b2, b3, b4, b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4, 2*b1*b5, 2*b1*b6, 2*b1*b7, 2*f*b5, 2*f*b6, 2*f*b7 ])
    return stacked_array * biases
end
