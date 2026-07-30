(input, output, D, emu) -> output .* (exp(input[2]) * 1e-10 .* D^2)
