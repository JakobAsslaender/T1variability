using DifferentialEquations
using SpecialFunctions
using LinearAlgebra
using DataFrames
using GLM
using LsqFit
using ApproxFun

struct gBloch end
struct Graham end
struct Sled end

##
function WM_param_qMT()
    m0s = 0.2
    R1f = 1 / 1.92   # 1/s
    R1s = 1 / 0.337  # 1/s
    Rx = 16.5       # 1/s
    R2f = 1 / 0.0776 # 1/s
    T2s = 12.4e-6    # s
    return m0s, R1f, R2f, Rx, R1s, T2s
end

function WM_param_Stanisz()
    m0s = 0.139
    R1f = 1 / 1.084  # 1/s
    R2f = 1 / 0.069  # 1/s
    Rx = 23         # 1/s
    R1s = 1          # 1/s
    T2s = 10e-6    # s
    return m0s, R1f, R2f, Rx, R1s, T2s
end


function WM_param_monoExp()
    m0s = 0
    R1f = 1 / 1.084    # 1/s
    R2f = 0          # 1/s
    Rx = 0          # 1/s
    R1s = R1f        # 1/s
    T2s = 12.4e-6    # s
    return m0s, R1f, R2f, Rx, R1s, T2s
end


##
u_sp = @SMatrix [
    0 0 0 0 0 0
    0 0 0 0 0 0
    0 0 1 0 0 0
    0 0 0 0 0 0
    0 0 0 0 1 0
    0 0 0 0 0 1]

function steady_state(U)
    Q = U - MRIgeneralizedBloch.A0(nothing)
    m = Q \ MRIgeneralizedBloch.C(nothing)
    return m
end

##
const G = interpolate_greens_function(greens_superlorentzian, 0, 1000)
R2sl = precompute_R2sl(TRF_max=5e-3, ω1_max=π / 500e-6)[1]
R2sl_short = precompute_R2sl(TRF_min=10e-6, TRF_max=20e-6, ω1_max=π / 10e-6)[1]

function RF_pulse_propagator(ω1::Number, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::gBloch; spoiler=true)
    if TRF >= 10e-6 && TRF <= 20e-6
        R2s = R2sl_short(TRF, abs(ω1 * TRF), B1, T2s)
        U = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, R2s))
        U = spoiler ? U * u_sp : U
    elseif TRF <= 5e-3
        R2s = R2sl(TRF, abs(ω1 * TRF), B1, T2s)
        U = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, R2s))
        U = spoiler ? U * u_sp : U
    else
        U = RF_pulse_propagator(_ -> ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model; spoiler=spoiler)
    end
    return U
end

function RF_pulse_propagator(ω1::Function, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::gBloch; spoiler=true)
    m0 = zeros(5)
    U = zeros(6, 6)
    Ui = @view U[[1:3; 5:6], [1:3; 5:6]]

    i_in = spoiler ? (3:5) : (1:5) # if a spoiler precedes the pulse, only z magnetization is non-zero
    for i ∈ i_in
        m0 .= 0
        m0[i] = 1
        mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
        Ui[1:5, i] = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)), reltol=1e-6, MethodOfSteps(RK4()))[end]
    end
    return U
end

function RF_pulse_propagator(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::Sled; spoiler=true)
    m0 = zeros(5)
    U = zeros(6, 6)
    Ui = @view U[[1:3; 5:6], [1:3; 5:6]]

    i_in = spoiler ? (3:5) : (1:5) # if a spoiler precedes the pulse, only z magnetization is non-zero
    for i ∈ i_in
        m0 .= 0
        m0[i] = 1
        mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
        Ui[1:5, i] = solve(ODEProblem(apply_hamiltonian_sled!, m0, (0, TRF), (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)), reltol=1e-6)[end]
    end
    return U
end

function RF_pulse_propagator(ω1::Real, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::Graham; spoiler=true)
    m0 = zeros(5)
    U = zeros(6, 6)
    Ui = @view U[[1:3; 5:6], [1:3; 5:6]]

    i_in = spoiler ? (3:5) : (1:5) # if a spoiler precedes the pulse, only z magnetization is non-zero
    for i ∈ i_in
        m0 .= 0
        m0[i] = 1
        mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
        Ui[1:5, i] = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0, (0, TRF), (ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s)), reltol=1e-6)[end]
    end
    return U
end

function RF_pulse_propagator(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, model::Graham; spoiler=true)
    Rrf = graham_saturation_rate_spectral_fast(ω1, B1, ω0, TRF, T2s)

    m0 = zeros(5)
    U = zeros(6, 6)
    Ui = @view U[[1:3; 5:6], [1:3; 5:6]]

    i_in = spoiler ? (3:5) : (1:5) # if a spoiler precedes the pulse, only z magnetization is non-zero
    for i ∈ i_in
        m0 .= 0
        m0[i] = 1
        mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
        Ui[1:5, i] = solve(ODEProblem(apply_hamiltonian_linear!, m0, (0, TRF), (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf)), reltol=1e-6)[end]
    end
    return U
end

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

##
function sinc_pulse(α, TRF; nLobes=3)
    nLobes % 2 != 1 ? error() : nothing
    x = (nLobes - 1) / 2 + 1
    ω1(t) = sinc((2t / TRF - 1) * x) * α * x * π / (sinint(x * π) * TRF)
end

# function hanning_pulse(α, TRF; ω0=0)
#     if TRF * ω0 > 0 && TRF * ω0 / 2π % 1 == 0
#         @error "TRF*ω0/2π cannot be an integer"
#     end
#     ω1(t) = α * cos(ω0 * (t - TRF / 2)) * cos(π * t / TRF - π / 2)^2 * (2 / TRF - TRF * ω0^2 / (2π^2)) / sinc(TRF * ω0 / 2π)
# end

# function hanning_ω1ms(α, TRF, TR; ω0=0)
#     (3α^2 * (4π^2 - TRF^2 * ω0^2) * (4π^4 * TRF - 5π^2 * TRF^3 * ω0^2 + TRF^5 * ω0^4 + 4π^4 * TRF * sinc(TRF * ω0 / π))) / (64π^4 * sinc(TRF * ω0 / 2π)^2 * TRF^2 * (π^2 - TRF^2 * ω0^2)) / TR
# end

function hanning_pulse(α, TRF)
    ω1(t) = α * cos(π * t / TRF - π / 2)^2 * 2 / TRF
end

function gauss_pulse(α, TRF; shape_alpha=2.5)
    y(t) = exp(-((2 * t / TRF - 1) * shape_alpha)^2 / 2)
    ω1(t) = α * (y(t) - y(0)) / (sqrt(π/2) * TRF * erf(shape_alpha/sqrt(2)) / shape_alpha - y(0)*TRF)
end

function CSMT_pulse(α, TRF, TR, ω1rms; ω0=12000π)
    # ω1_0 = hanning_pulse(α, TRF)
    ω1_0 = gauss_pulse(α, TRF; shape_alpha=2.5) # shape & shape_alpha confirmed by Dr. Teixeira
    ω1ms_0 = quadgk(t -> ω1_0(t)^2, 0, TRF)[1] / TR
    # ω1ms_0 = hanning_ω1ms(α, TRF, TR; ω0=0)

    β = ω1rms^2 / ω1ms_0 - 1
    if β < 0
        β = 0
        @error "negative Δω1ms with α = $(α*180/pi)deg; on-resonant pulse has ω1rms = $(sqrt(ω1ms_0)). Setting it to zero."
    end

    wt(t) = 1 - sqrt(2 * β) * cos(ω0 * t)
    ω1(t) = ω1_0(t) * wt(t)
    return ω1
end


##
# TRF = 10.24e-3 # s
# γ = 267.522e6 # gyromagnetic ratio in rad/s/T
# ω₁ᵐᵃˣ = 13e-6 * γ # rad/s
# μ = 5 # shape parameter in rad
# β = 674.1 # shape parameter in 1/s
function sech_inversion_pulse(; TRF=10.24e-3, ω₁ᵐᵃˣ=13e-6 * 267.522e6, μ=5, β=674.1)
    ω1(t) = ω₁ᵐᵃˣ * sech(β * (t - TRF / 2)) # rad/s
    ω0(t) = -μ * β * tanh(β * (t - TRF/2)) # rad/s
    φ(t) = -μ * log(cosh(β * t) - sinh(β * t) * tanh(β * TRF / 2)) # rad
    return (ω1, ω0, φ, TRF)
end

# Shape according to doi:10.1006/jmre.2001.2340
function sechn_inversion_pulse(; TRF=10e-3, ω₁ᵐᵃˣ=13e-6 * 267.522e6 / 2, μ=14, β=250, n = 8)
    f1(τ) = sech((β * τ)^n)
    ω1(t) = ω₁ᵐᵃˣ * f1(t - TRF / 2) # rad/s

    ω0_i(t) = -μ * β^2 * quadgk((τ) -> f1(τ)^2, 0, t - TRF/2)[1]
    ω0 = Fun(ω0_i, 0..TRF)
    # φ_i(t) = quadgk(ω0, -TRF/2, t)[1] - quadgk(ω0, -TRF/2, 0)[1]
    φ = cumsum(ω0)
    return (ω1, ω0, φ, TRF)
end