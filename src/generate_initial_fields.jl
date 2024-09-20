using FFTW
using Plots
using GLMakie

# Energy profile
kp = 20.0 # Controls smoothness
A = 16π / 3kp^5
profile(k) = sqrt(A * k^4 * exp(-2π * (k / kp)^2))

# Wavenumbers
K = 64
N = 2K
kx = 0:K-1
ky = (0:K-1)'
knorm = @. sqrt(kx^2 + ky^2)

# Amplitude
a = @. N^2 * profile(knorm)
a = [a; reverse(a; dims = 1);; reverse(a; dims = 2); reverse(a; dims = (1, 2))]

# Phase shift
ξx = rand(K, K)
ξy = rand(K, K)
ξx = [ξx; -reverse(ξx; dims = 1);; reverse(ξx; dims = 2); -reverse(ξx; dims = (1, 2))]
ξy = [ξy; reverse(ξy; dims = 1);; -reverse(ξy; dims = 2); -reverse(ξy; dims = (1, 2))]
ξ = @. ξx + ξy

# Random amplitude
a = @. exp(2π * im * ξ) * a

# Random velocity direction for (u, v)
θ = rand(2K, 2K)
ex = @. cospi(2 * θ)
ey = @. sinpi(2 * θ)
ex = @. ex / sqrt(ex^2 + ey^2)
ey = @. ey / sqrt(ex^2 + ey^2)

# Velocity vector uhat
ux = @. a * ex
uy = @. a * ey

# Velocity vector u
ux = real.(ifft(ux))
uy = real.(ifft(uy))

# # Then project...
# u = project(u)

heatmap(ux)
heatmap(uy)