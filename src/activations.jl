# activations.jl


using NNlib
using Zygote

xrg = -5.0:0.1:5.0

f_tanh = Figure()
ax_tanh = Axis(f_tanh[1, 1], xlabel="x", ylabel="f(x)")
lines!(ax_tanh, xrg, NNlib.tanh.(xrg), label="f(x) = tanh(x)")
lines!(ax_tanh, xrg, [Zygote.gradient(x -> NNlib.tanh(x), x)[1] for x ∈ xrg], label="f'(x)")
axislegend(ax_tanh,position=:lt)
f_tanh
save("plots/activation_tanh.png", f_tanh)

f_σ = Figure()
ax_σ = Axis(f_σ[1, 1], xlabel="x", ylabel="f(x)")
lines!(ax_σ, xrg, NNlib.σ.(xrg), label="f(x) = σ(x)")
lines!(ax_σ, xrg, [Zygote.gradient(x -> NNlib.σ(x), x)[1] for x ∈ xrg], label="f'(x)")
axislegend(ax_σ, position=:lt)
f_σ
save("plots/activation_σ.png", f_σ)

f_relu = Figure()
ax_relu = Axis(f_relu[1, 1], xlabel="x", ylabel="f(x)")
lines!(ax_relu, xrg, NNlib.relu.(xrg), label="f(x) = relu(x)")
lines!(ax_relu, xrg, [Zygote.gradient(x -> NNlib.relu(x), x)[1] for x ∈ xrg], label="f'(x)")

axislegend(ax_relu,position=:lt)
f_relu
save("plots/activation_relu.png", f_relu)



≈



