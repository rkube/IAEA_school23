## Use neural network as surrogate

using CairoMakie
using Surrogates
using NNlib
using Flux
using SurrogatesFlux


# Define the Schaffer function that we will create a surrogate for
function schaffer(x)
    x1=x[1]
    x2=x[2]
    fact1 = x1 ^2;
    fact2 = x2 ^2;
    y = fact1 + fact2;
end

# Sample at 60 points
n_samples = 100
lower_bound = [0.0, 0.0]
upper_bound = [8.0, 8.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = schaffer.(xys);


# Plot true function and sampling locations
x_fine = 0.0:0.01:8.0
y_fine = 0.0:0.01:8.0
zs_fine = zeros(Float64, length(x_fine), length(y_fine))
for ix_y ∈ 1:length(y_fine)
    for ix_x ∈ 1:length(x_fine)
        zs_fine[ix_x, ix_y] = schaffer([x_fine[ix_x], y_fine[ix_y]])
    end
end

fig_sample = Figure()
ax_sample = Axis(fig_sample[1, 1], title="f(x) = x² + y²", xlabel="x", ylabel="y")
contourf!(ax_sample, x_fine, y_fine, zs_fine, colormap=:lisbon, levels=32)
scatter!(ax_sample, [x[1] for x ∈ xys], [x[2] for x ∈ xys], color=:red, label="Samples")
axislegend(ax_sample, position=:lt)

fig_sample
save("plots/neural_samples.png", fig_sample)


## Instantiate surrogate and Optimize

model_σ = Chain(
  Dense(2, 32, NNlib.σ),
  Dense(32, 16, NNlib.σ),
  Dense(16, 1)
)

model_tanh = Chain(
  Dense(2, 32, NNlib.tanh),
  Dense(32, 32, NNlib.tanh),
  Dense(32, 32, NNlib.tanh),
  Dense(32, 1)
)

model_relu = Chain(
  Dense(2, 32, NNlib.relu),
  Dense(32, 32, NNlib.relu),
  Dense(32, 32, NNlib.relu),
  Dense(32, 1)
)

# Fit surrogates
neural_σ = NeuralSurrogate(xys, zs, lower_bound, upper_bound, model=model_σ, opt=Adam(0.001), n_echos=30)
neural_tanh = NeuralSurrogate(xys, zs, lower_bound, upper_bound, model=model_tanh, opt=Adam(0.001), n_echos=30)
neural_relu = NeuralSurrogate(xys, zs, lower_bound, upper_bound, model=model_relu, opt=Adam(0.001), n_echos=30)

# Calculate MSE
MSE_σ = sum(abs2, [neural_σ(xy) for xy ∈ xys] - zs) / n_samples
MSE_tanh = sum(abs2, [neural_tanh(xy) for xy ∈ xys] - zs) / n_samples
MSE_relu = sum(abs2, [neural_relu(xy) for xy ∈ xys] - zs) / n_samples


# Plot performance of surrogate:
fig_perf = Figure()
ax_perf = Axis(fig_perf[1, 1], xlabel="f⁺(x,y) - Neural network", ylabel="f(x,y) - True value")
scatter!(ax_perf, [neural_σ(xy) for xy ∈ xys], zs, color=Makie.wong_colors()[1], label="σ, MSE=$(round(MSE_σ, digits=3))")
scatter!(ax_perf, [neural_tanh(xy) for xy ∈ xys], zs, color=Makie.wong_colors()[2], label="tanh, MSE=$(round(MSE_tanh, digits=3))")
scatter!(ax_perf, [neural_relu(xy) for xy ∈ xys], zs, color=Makie.wong_colors()[3], label="relu, MSE=$(round(MSE_relu, digits=3))")
axislegend(ax_perf)
fig_perf

save("plots/neural_fit1.png", fig_perf)


#res = surrogate_optimize(schaffer, SRBF(), lower_bound, upper_bound, neural, SobolSample(), maxiters=20, num_new_samples=10)


