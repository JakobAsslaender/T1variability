# # Helper functions
# ## Load packages
using MRIgeneralizedBloch
using StaticArrays
using Statistics
using QuadGK
using DifferentialEquations
using SpecialFunctions
using LinearAlgebra
using DataFrames
using StatsBase
using LsqFit
using ApproxFun
using Plots
plotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native, size=(600, 600))
nothing #hide #md

# ## Define MT models
# These structs are used to identify the MT model with Julia's multiple dispatch mechanism
struct gBloch end
struct Graham end
struct Sled end
nothing #hide #md

# ## Spoiler propagator
# This matrix implements "perfect" RF spoiling that destroys all transversal magnetization.
u_sp = @SMatrix [
    0 0 0 0 0 0
    0 0 0 0 0 0
    0 0 1 0 0 0
    0 0 0 0 0 0
    0 0 0 0 1 0
    0 0 0 0 0 1]
nothing #hide #md


# ## Steady state
# The input `U` of this function is a matrix that describes the spin evolution during an RF pulse sequence. The function uses linear algebra to calculate the steady state magnetization `m` resulting from repeated execution of the pulse sequence described by `U`.
function steady_state(U)
    U0 = @SMatrix [
        1 0 0 0 0 0;
        0 1 0 0 0 0;
        0 0 1 0 0 0;
        0 0 0 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 0]
    Q = U - U0
    m = Q \ @SVector [0,0,0,0,0,1]
    return m
end
nothing #hide #md

# ## RF pulse propagators
# ### Generalized Bloch model
# First, we pre-calculate the linearization of the generalized Bloch model. Refer to the [documentation of the generalized Bloch package](https://jakobasslaender.github.io/MRIgeneralizedBloch.jl/stable/build_literate/Linear_Approximation/) for details.
const G = interpolate_greens_function(greens_superlorentzian, 0, 1000)
const R2sl_1 = precompute_R2sl(TRF_min=1e-6, TRF_max=3e-6, ω1_max=π / 500e-6, T2s_min=12e-6, T2s_max=17e-6, B1_max=1.1)[1]
const R2sl_2 = precompute_R2sl(TRF_min=10e-6, TRF_max=20e-6, ω1_max=π / 10e-6, T2s_min=12e-6, T2s_max=17e-6, B1_max=1.1)[1]
const R2sl_3 = precompute_R2sl(TRF_max=1e-3, ω1_max=π / 500e-6, T2s_min=12e-6, T2s_max=17e-6, B1_max=1.1)[1]
nothing #hide #md

# We implemented different methods for the function `RF_pulse_propagator` that inferred with Julia's multiple dispatch logic based on the type of the input parameters. The functions in this section take the variable `model` of type `gBloch` and implement the generalized Bloch model. The first method further takes the variable `ω1` of the abstract type `Number`, i.e., it implements pulse propagators for a constant ω₁.
function RF_pulse_propagator(ω1::Number, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::gBloch; spoiler=true)
    if TRF >= 1e-6 && TRF <= 3e-6
        R2s = R2sl_1(TRF, abs(ω1 * TRF), B1, T2s)
        U = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, R2s))
        U = spoiler ? U * u_sp : U
    elseif TRF >= 10e-6 && TRF <= 20e-6
        R2s = R2sl_2(TRF, abs(ω1 * TRF), B1, T2s)
        U = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, R2s))
        U = spoiler ? U * u_sp : U
    elseif TRF >= 100e-6 && TRF <= 1e-3
        R2s = R2sl_3(TRF, abs(ω1 * TRF), B1, T2s)
        U = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, R2s))
        U = spoiler ? U * u_sp : U
    else
        U = RF_pulse_propagator(_ -> ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model; spoiler=spoiler)
    end
    return U
end
nothing #hide #md

# The following method takes the variable `ω1` of type `Function`, i.e., it implements pulse propagators with a variable ω₁ that is implemented as the function `ω1(t)`:
function RF_pulse_propagator(ω1::Function, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::gBloch; spoiler=true)
    U = zeros(6, 6)
    Ui = @view U[[1:3; 5:6], [1:3; 5:6]]

    i_in = spoiler ? (3:5) : (1:5) # if a spoiler precedes the pulse, only the z magnetization is non-zero
    Threads.@threads for i ∈ i_in
        m0 = zeros(5)
        m0[i] = 1
        mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
        Ui[1:5, i] = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)), reltol=1e-6, MethodOfSteps(RK4()))[end]
    end
    return U
end
nothing #hide #md

# ### Graham's model
function RF_pulse_propagator(ω1::Number, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::Graham; spoiler=true)
    m0 = zeros(5)
    U = zeros(6, 6)
    Ui = @view U[[1:3; 5:6], [1:3; 5:6]]

    i_in = spoiler ? (3:5) : (1:5) # if a spoiler precedes the pulse, only the z magnetization is non-zero
    for i ∈ i_in
        m0 .= 0
        m0[i] = 1
        Ui[1:5, i] = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0, (0, TRF), (ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s)), Vern9(), reltol=1e-6)[end]
    end
    return U
end
nothing #hide #md

#-
function RF_pulse_propagator(ω1::Function, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::Graham; spoiler=true)
    Rrf = graham_saturation_rate_spectral_fast(ω1, B1, ω0, TRF, T2s)

    m0 = zeros(5)
    U = zeros(6, 6)
    Ui = @view U[[1:3; 5:6], [1:3; 5:6]]

    i_in = spoiler ? (3:5) : (1:5) # if a spoiler precedes the pulse, only the z magnetization is non-zero
    for i ∈ i_in
        m0 .= 0
        m0[i] = 1
        Ui[1:5, i] = solve(ODEProblem(apply_hamiltonian_linear!, m0, (0, TRF), (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf)), Vern9(), reltol=1e-6)[end]
    end
    return U
end
nothing #hide #md

# ### Sled's model
function RF_pulse_propagator(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::Sled; spoiler=true)
    m0 = zeros(5)
    U = zeros(6, 6)
    Ui = @view U[[1:3; 5:6], [1:3; 5:6]]

    i_in = spoiler ? (3:5) : (1:5) # if a spoiler precedes the pulse, only the z magnetization is non-zero
    for i ∈ i_in
        m0 .= 0
        m0[i] = 1
        mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
        Ui[1:5, i] = solve(ODEProblem(apply_hamiltonian_sled!, m0, (0, TRF), (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)), Vern9(), reltol=1e-6)[end]
    end
    return U
end
nothing #hide #md

# ## Helper functions for Graham's spectral model
# The following code saves and reuses the saturation rate `Rrf` of an RF pulse to speed up the simulation.
struct Rrf_Atom
    ω1_sqrtTRF::Float64
    ω1_TRFo2::Float64
    ω1_TRFmsqrtTRF::Float64
    B1::Float64
    ω0::Float64
    TRF::Float64
    T2s::Float64
    Rrf::Float64
end
Rrf_Dict = Rrf_Atom[]
function graham_saturation_rate_spectral_fast(ω1, B1, ω0, TRF, T2s)
    T2s == 0 && return 1 # for mono-exponential model

    d_ω0 = typeof(ω0) <: Function ? ω0(sqrt(TRF)) : ω0 # just used for comparison

    for x ∈ Rrf_Dict
        if ω1(sqrt(TRF)) == x.ω1_sqrtTRF && ω1(TRF/2) == x.ω1_TRFo2 && ω1(TRF-sqrt(TRF)) == x.ω1_TRFmsqrtTRF && B1 == x.B1 && d_ω0 == x.ω0 && TRF == x.TRF && T2s == x.T2s
            return x.Rrf
        end
    end

    Rrf = graham_saturation_rate_spectral(ω0_int -> lineshape_superlorentzian(ω0_int, T2s), t -> ω1(t)*B1, TRF, ω0)
    push!(Rrf_Dict, Rrf_Atom(ω1(sqrt(TRF)), ω1(TRF/2), ω1(TRF-sqrt(TRF)), B1, d_ω0, TRF, T2s, Rrf))
    return Rrf
end
nothing #hide #md

# ## RF pulse definitions
function sinc_pulse(α, TRF; nLobes=3)
    nLobes % 2 != 1 ? error() : nothing
    x = (nLobes - 1) / 2 + 1
    ω1(t) = sinc((2t / TRF - 1) * x) * α * x * π / (sinint(x * π) * TRF)
end
nothing #hide #md

#-
function hanning_pulse(α, TRF)
    ω1(t) = α * cos(π * t / TRF - π / 2)^2 * 2 / TRF
end
nothing #hide #md

#-
function gauss_pulse(α, TRF; shape_alpha=2.5)
    y(t) = exp(-((2 * t / TRF - 1) * shape_alpha)^2 / 2)
    ω1(t) = α * (y(t) - y(0)) / (sqrt(π/2) * TRF * erf(shape_alpha/sqrt(2)) / shape_alpha - y(0)*TRF)
end
nothing #hide #md

# Controlled saturation MT pulse as described by [Teixeira et al. (2019)](http://doi.org/10.1002/mrm.27442):
function CSMT_pulse(α, TRF, TR, ω1rms; ω0=12000π)
    ω1_0 = gauss_pulse(α, TRF; shape_alpha=2.5) # shape & shape_alpha confirmed by Dr. Teixeira
    ω1ms_0 = quadgk(t -> ω1_0(t)^2, 0, TRF)[1] / TR

    β = ω1rms^2 / ω1ms_0 - 1
    if β < 0
        β = 0
        @error "negative Δω1ms with α = $(α*180/pi)deg; on-resonant pulse has ω1rms = $(sqrt(ω1ms_0)). Setting it to zero."
    end

    wt(t) = 1 - sqrt(2 * β) * cos(ω0 * t)
    ω1(t) = ω1_0(t) * wt(t)
    return ω1
end
nothing #hide #md

#-
#src Standard Siemen parameters:
#src TRF = 10.24e-3 s
#src β = 674.1 shape parameter in 1/s
#src μ = 5 shape parameter (dimensionless)
#src ω₁ᵐᵃˣ = 4965.910769033364 rad/s – scaled such that the integral over the real part of the pulse (real(ω1 * exp(1im * φ))) equates to 2π
function sech_inversion_pulse(; TRF=10.24e-3, ω₁ᵐᵃˣ=4965.910769033364, μ=5, β=674.1)
    ω1(t) = ω₁ᵐᵃˣ * sech(β * (t - TRF / 2)) # rad/s
    ω0(t) = -μ * β * tanh(β * (t - TRF/2)) # rad/s
    φ_u(t) = μ * log(cosh(β * t) - sinh(β * t) * tanh(β * TRF / 2)) # rad
    φ(t) = φ_u(t) - φ_u(TRF/2)
    return (ω1, ω0, φ, TRF)
end
nothing #hide #md

# `Sechn` pulse shape according to [this paper](http://doi.org/10.1006/jmre.2001.2340), confirmed by Dr. O'Brien:
function sechn_inversion_pulse(; TRF=10.24e-3, ω₁ᵐᵃˣ=10e-6 * 267.522e6, β=240, μ=35, n = 8)
    f1(τ) = sech((β * τ)^n)
    ω1(t) = ω₁ᵐᵃˣ * f1(t - TRF / 2) # rad/s

    ω0_i(t) = μ * β^2 * quadgk((τ) -> f1(τ)^2, 0, t - TRF/2)[1]
    ω0 = Fun(ω0_i, 0..TRF)
    φ_ = cumsum(ω0)
    φ(t) = φ_(t) - φ_(TRF/2)
    return (ω1, ω0, φ, TRF)
end
nothing #hide #md