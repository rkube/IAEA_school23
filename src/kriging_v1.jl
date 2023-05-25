# Try Kriging on a simple dataset by hand.

using Surrogates
using CairoMakie


# https://www.sfu.ca/~ssurjano/forretal08.html
# Forrester et al. (2008) Function
f(x) = (6 * x - 2)^2 * sin(12 * x - 4)

n_samples = 4
lower_bound = 0.0
upper_bound = 1.0

xs = lower_bound:0.001:upper_bound

x = sample(n_samples, lower_bound, upper_bound, SobolSample())
sort!(x)
x
# Cache evaluation of f on x.
y = f.(x)


## Calculate co-variance matrixx
σ(x1, x2) = exp(-(abs(x1 - x2))^2)
Σ = zeros(4, 4)
for i ∈ 1:4
    for j ∈ 1:4
        Σ[i,j] = σ(x[i], x[j])
    end
end
Σ

# Calculate function at new point at x=0.5:
σ_new = [σ(0.5, xold) for xold ∈ x]
# Calculcate weights λ by solving the linear system Σ λ = σ_new for 
# weights λ
λ = Σ \ σ_new
# Calculate function value using weights.
λ' * f.(x)

fig_s = Figure()
ax_s = Axis(fig_s[1, 1])
scatter!(ax_s, x, y, label="Sampled points", xlims=(lower_bound, upper_bound), ylims=(-7, 17))
lines!(ax_s, xs,  f.(xs), color=:red, label="f(x) = (6x-2)² * sin(12x -4)")
axislegend(ax_s, position=:lt)

save("plots/surrogate_kriging0.png", fig_s)
scatter!(ax_s, [0.5], [λ' * f.(x)], label="f(0.5)", color=:black)
axislegend(ax_s, position=:lt)

fig_s


save("plots/surrogate_kriging1.png", fig_s)



### Let's try this again, but add 0.5 To the vector of known values
x2 = sort(cat(x, [0.5], dims=1))

## Calculate co-variance matrixx
Σ2 = zeros(5, 5)
for i ∈ 1:size(Σ2, 1)
    for j ∈ 1:size(Σ2, 2)
        Σ2[i,j] = σ(x2[i], x2[j])
    end
end
Σ2

# Calculate function at new point at x=0.5:
σ_new2 = [σ(0.55, xold) for xold ∈ x2]
# Calculcate weights λ
λ2 = Σ2 \ σ_new2
# Calculate function value using weights.
λ2' * f.(x2)
y2 = f.(x2)

fig_s = Figure()
ax_s = Axis(fig_s[1, 1])
scatter!(ax_s, x2, y2, label="Sampled points", xlims=(lower_bound, upper_bound), ylims=(-7, 17))
lines!(ax_s, xs,  f.(xs), color=:red, label="f(x) = (6x-2)² * sin(12x -4)")

axislegend(ax_s, position=:lt)

save("plots/surrogate_kriging2.png", fig_s)
scatter!(ax_s, [0.55], [λ2' * f.(x2)], label="f(0.55)", color=:black)
axislegend(ax_s, position=:lt)

fig_s

save("plots/surrogate_kriging2.png", fig_s)


## Now instantiate a surrogate model on the 5 sampling points
kriging_surrogate = Kriging(x2, y2, lower_bound, upper_bound);

fig_sur = Figure()
ax_sur = Axis(fig_sur[1, 1])

scatter!(ax_sur, x2, y2, label="Sampled points", color=Makie.wong_colors()[1])
lines!(ax_sur, xs, f.(xs), label="f(x) = (6x-2)² * sin(12x -4)", color=Makie.wong_colors()[2])
lines!(ax_sur, xs, kriging_surrogate.(xs), label="Surrogate function", color=Makie.wong_colors()[3])
y_sur = kriging_surrogate.(xs)
y_sur_std = map(x -> std_error_at_point(kriging_surrogate,x), xs)
band!(ax_sur, xs, y_sur - y_sur_std, y_sur + y_sur_std, color=(Makie.wong_colors()[3], 0.3))

axislegend(ax_sur,position=:lt)

fig_sur

save("plots/surrogate_kriging3.png", fig_sur)

## Optimize the function using the surrogate model
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, kriging_surrogate, SobolSample())
# Note that now the vector of positions, x2, is 27 elements long.
y_sur2 =  kriging_surrogate.(xs)
y_sur2_std = map(x -> std_error_at_point(kriging_surrogate, x), xs)

fig_opt = Figure()
ax_opt = Axis(fig_opt[1, 1])

scatter!(ax_opt, x2, y2, label="Sampled points")
lines!(ax_opt, xs, f.(xs), label="True function")
lines!(ax_opt, xs, y_sur2, label="Surrogate function")
band!(ax_opt, xs, y_sur2 - y_sur2_std, y_sur2 + y_sur2_std, color=(Makie.wong_colors()[3],0.3))
axislegend(ax_opt, position=:lt)
fig_opt

save("plots/surrogate_kriging4.png", fig_opt)



