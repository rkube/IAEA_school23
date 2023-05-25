# Following the surrogates.jl tutorial

# In vscode you can press shift-enter to evaluate the current line in the repl
# (tab-window below)
using Surrogates
using CairoMakie

# First example is to build a surrogate for f(x) = log(x) * x^2 + x^3

lower = 2.5
upper = 25.0
x_all = lower:0.1:upper
f(x) = log(x) * x^2 + x^3
# Let's store f evaluated on x 
y_all = @. f(x_all)

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="f(x) = log(x) * x² + x³")
lines!(ax, x_all, y_all, label="True function")
fig    # Evaluate fig with shift-enter gives the output. This is automatically plotted in
       # plot window

# Sample 30 points on the interval [5.0:25.0]
n_samples = 10
x_surrogate = sort(sample(n_samples, lower, upper, UniformSample()))
y_surrogate = f.(x_surrogate)

scatter!(ax, x_surrogate, y_surrogate, color=:red, label="Sampled points")
axislegend(ax, position=:lt)
fig

# Let's build our first surrogate model
radial_surrogate = RadialBasis(x_surrogate, y_surrogate, lower_bound, upper_bound; rad=linearRadial())

# Zoom in on a surrogate point 
x_zoom = (x_surrogate[3] - 0.1):1e-3:(x_surrogate[5] + 0.1)
y_zoom = radial_surrogate.(x_zoom)

fig_zoom = Figure()
ax_zoom = Axis(fig_zoom[1,1], xlabel="x", ylabel="y")
lines!(ax_zoom, x_zoom, y_zoom)
scatter!(ax_zoom, x_surrogate[3:5], y_surrogate[3:5])
lines!(ax_zoom, x_zoom, f.(x_zoom))
fig_zoom



# Evaluate the surrogate performance
fig_eval = Figure()
ax_vals = Axis(fig_eval[1, 1], title="Linear radial basis function",
               ylabel="f(x) = log(x) x² + x³")
ax_diff = Axis(fig_eval[2, 1], ylabel="Model - Surrogate", xlabel="x")

lines!(ax_vals, x_all, y_all, label="Model")
lines!(ax_vals, x_surrogate, y_surrogate, label="Surrogate")
scatter!(ax_vals, x_surrogate, y_surrogate, color=:red, label="Sampled points")
axislegend(ax_vals, position=:lt)

lines!(ax_diff, x_all, radial_surrogate.(x_all) .- y_all)

fig_eval
save("plots/basis_linear.png", fig_eval)


