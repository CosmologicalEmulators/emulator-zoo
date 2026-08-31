biases -> begin
    b1, b2, bs, b3, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, pshot, f0 = biases
    [
        b1^2, 2b1 * f0, f0^2,
        one(b1), b1, b1^2, b2, b1 * b2, b2^2,
        bs, b1 * bs, b2 * bs, bs^2, b3, b1 * b3,
        alpha0, alpha2, alpha4,
        ctilde * b1^2, 2ctilde * b1 * f0, ctilde * f0^2,
        pshot * alphashot0, pshot * alphashot2,
    ]
end
