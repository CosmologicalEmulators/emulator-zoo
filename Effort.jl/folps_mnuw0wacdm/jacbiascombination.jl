biases -> begin
    b1, b2, bs, b3, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, pshot, f0 = biases
    jacobian = zeros(eltype(biases), 23, 12)
    jacobian[1, 1] = 2b1
    jacobian[2, 1], jacobian[2, 12] = 2 * f0, 2b1
    jacobian[3, 12] = 2 * f0
    jacobian[5, 1] = 1
    jacobian[6, 1] = 2b1
    jacobian[7, 2] = 1
    jacobian[8, 1], jacobian[8, 2] = b2, b1
    jacobian[9, 2] = 2b2
    jacobian[10, 3] = 1
    jacobian[11, 1], jacobian[11, 3] = bs, b1
    jacobian[12, 2], jacobian[12, 3] = bs, b2
    jacobian[13, 3] = 2bs
    jacobian[14, 4] = 1
    jacobian[15, 1], jacobian[15, 4] = b3, b1
    jacobian[16, 5], jacobian[17, 6], jacobian[18, 7] = 1, 1, 1
    jacobian[19, 1], jacobian[19, 8] = 2ctilde * b1, b1^2
    jacobian[20, 1], jacobian[20, 8], jacobian[20, 12] = 2ctilde * f0, 2b1 * f0, 2ctilde * b1
    jacobian[21, 8], jacobian[21, 12] = f0^2, 2ctilde * f0
    jacobian[22, 9], jacobian[22, 11] = pshot, alphashot0
    jacobian[23, 10], jacobian[23, 11] = pshot, alphashot2
    jacobian
end
