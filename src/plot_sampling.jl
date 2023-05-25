# Plot distributions of sampling points

using CairoMakie
using Distributions, Random

lower = 0.0  # Lower bound for sampling
upper = 1.0  # Upper bound for sampling
n_pts = 10  # Number of points to choose


## Uniform sampling
points_uniform = rand(Uniform(lower, upper), n_pts * n_pts)
f_uni = Figure()
ax_uni = Axis(f_uni[1, 1], xlabel="x", ylabel="x", title="Uniform sampling")
scatter!(ax_uni, points_uniform[1:2:end-1], points_uniform[2:2:end])
f_uni
save("plots/sampling_uniform.png", f_uni)


## Regular grid sampling
# Sample regular points along each dimension
xrg_grid = range(lower, stop=upper, length=n_pts)
yrg_grid = range(lower, stop=upper, length=n_pts)
# Repeat for forming vectors x_i

xrg_grid = repeat(xrg_grid, inner=n_pts, outer=1)
yrg_grid = repeat(yrg_grid, inner=1, outer=n_pts)


f_grid = Figure()
ax_grid = Axis(f_grid[1, 1], xlabel="x", ylabel="x", title="Grid sampling")
scatter!(ax_grid, xrg_grid, yrg_grid)
f_grid
save("plots/sampling_grid.png", f_grid)

## Latin hypercube sampling
# Collect random permutations 
P_j = zeros(Int64, 2, n_pts * n_pts)
for i ∈ axes(P_j, 1)
    P_j[i, :] = shuffle(1:(n_pts*n_pts))
end

# Generate uniformly distributed points
#U_ij = rand(Uniform(lower, upper), 2, n_pts*n_pts)
U_ij = rand(Uniform(0.0, 1.0), 2, n_pts*n_pts)

X_ij = zeros(2, n_pts * n_pts)
for d = 1:2
    for n = 1:n_pts*n_pts
        X_ij[d, n] = (P_j[d, n] - U_ij[d, n]) / (n_pts * n_pts)
    end
end


f_lhs = Figure()
ax_lhs = Axis(f_lhs[1, 1], xlabel="x", ylabel="x", title="Latin hypercube")
ax_lhs.xticks = 0.0:0.1:1.0
ax_lhs.yticks = 0.0:0.1:1.0
scatter!(ax_lhs, X_ij[1,:], X_ij[2, :])
f_lhs

save("plots/sampling_lhs.png", f_lhs)

