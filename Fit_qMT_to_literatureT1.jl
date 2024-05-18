using Pkg
Pkg.activate(Base.source_path() * "/..")
using MRIgeneralizedBloch
using StaticArrays
using Statistics
using QuadGK
using Plots
plotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native, size=(600, 600))

include("helper_functions.jl")

##
T1a_meas = Float64[]
T1a_func = []
incl_fit = Bool[]
seq_name = String[]
seq_type = Symbol[]

## #########################################################
# IR: Stanisz et al. (https://doi.org/10.1002/mrm.20605)
############################################################
function calculate_T1a_IRStanisz(m0s, R1f, R2f, Rx, R1s, T2s)
    TRF_inv = 10e-6 # 10us - 20us according to private conversations
    TRF_exc = TRF_inv # guessed, but has a negligible effect
    TI = exp.(range(log(1e-3), log(32), 35))
    TD = similar(TI)
    TD .= 20

    u_inv = RF_pulse_propagator(π / TRF_inv, B1, ω0, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
    u_exc = RF_pulse_propagator(π / 2 / TRF_exc, B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)

    u_rti = [exp(hamiltonian_linear(0, B1, ω0, iTI - (TRF_inv + TRF_exc) / 2, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]
    u_rtd = [exp(hamiltonian_linear(0, B1, ω0, iTD - (TRF_inv + TRF_exc) / 2, m0s, R1f, R2f, Rx, R1s, 1)) for iTD ∈ TD]

    s = similar(TI)
    for i in eachindex(TI)
        U = u_exc * u_rti[i] * u_inv * u_rtd[i]
        s[i] = steady_state(U)[1]
    end

    model3(t, p) = p[1] .- p[2] .* exp.(-p[3] * t)
    fit = curve_fit(model3, TI, s, [1, 2, 0.8])
    return 1 / fit.param[end]
end

push!(T1a_meas, 1.084) #± 0.045s in WM
push!(T1a_func, calculate_T1a_IRStanisz)
push!(seq_name, "IR Stanisz et al.")
push!(seq_type, :IR)

## #########################################################
# IR: Stikhov et al. (https://doi.org/10.1002/mrm.25135)
############################################################
function calculate_T1a_IRStikhov(m0s, R1f, R2f, Rx, R1s, T2s)
    nLobes = 3 # confirmed by authors
    TRF_exc = 3.072e-3 # s; confirmed by authors
    TRF_ref = 3e-3 # s; confirmed by authors
    TI = [30e-3, 530e-3, 1.03, 1.53]
    TR = 1.55
    TE = 11e-3

    # excitation block
    u_exc = RF_pulse_propagator(sinc_pulse(-π / 2, TRF_exc; nLobes=nLobes), B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model; spoiler=true)
    u_ref = RF_pulse_propagator(gauss_pulse(π, TRF_ref), B1, ω0, TRF_ref, m0s, R1f, R2f, Rx, R1s, T2s, MT_model; spoiler=false)
    u_te2 = exp(hamiltonian_linear(0, B1, ω0, (TE - TRF_exc - TRF_ref) / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc = u_ref * u_te2 * u_exc

    # adiabatic inversion pulse confirmed by the authors
    ω1, _, φ, TRF_inv = sech_inversion_pulse() # 360 deg, defined by the intgral over the RF's real part.
    u_inv = RF_pulse_propagator(ω1, B1, φ, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)


    # relaxation blocks
    u_rti = [exp(hamiltonian_linear(0, B1, ω0, iTI - (TRF_inv + TRF_exc) / 2, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]
    u_rtd = [exp(hamiltonian_linear(0, B1, ω0, TR - iTI - (TRF_inv + TRF_ref + TE) / 2, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]

    s = similar(TI)
    for i in eachindex(TI)
        U = u_exc * u_sp * u_rti[i] * u_sp * u_inv * u_sp * u_rtd[i] * u_sp
        s[i] = steady_state(U)[1]
    end

    model3(t, p) = p[1] .- p[2] .* exp.(-p[3] * t)
    fit = curve_fit(model3, TI, s, [1, 2, 0.8])
    return 1 / fit.param[end]
end

push!(T1a_meas, 850e-3) # peak of histogram
push!(T1a_func, calculate_T1a_IRStikhov)
push!(seq_name, "IR Stikhov et al.")
push!(seq_type, :IR)


## #########################################################
# IR: Preibisch et al. (https://doi.org/10.1002/mrm.21776)
############################################################
function calculate_T1a_IRPreibisch(m0s, R1f, R2f, Rx, R1s, T2s)
    TI = [100, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000, 2600, 3200, 3800, 4400, 5000] .* 1e-3
    TD = 20
    TE = 27e-3

    # The adiabatic inversion pulse was identical to the one described in http://doi.org/10.1002/mrm.20552 (per private communications with Dr. Deichmann)
    TRF_inv = 8.192e-3
    β = 4.5
    μ = 5
    ω₁ᵐᵃˣ=13.5*π/TRF_inv
    # ω1, _, _, _ = sech_inversion_pulse(TRF=TRF_inv, ω₁ᵐᵃˣ=13.5*π/TRF, μ=μ, β=β)
    ω1_inv(t) = ω₁ᵐᵃˣ * sech(β * (2t / TRF_inv - 1)) # rad/s
    φ_inv(t)  = μ * log(sech(β * (2t / TRF_inv - 1))) # rad
    u_inv = RF_pulse_propagator(ω1_inv, B1, φ_inv, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)


    # The shape of the excitation pulse was kindly provided by Dr. Deichmann
    TRF_exc = 2.5e-3
    ω1_exc_(t) = sinc(2 * abs(2t/TRF_exc-1)^0.88) * cos(π/2 * (2t/TRF_exc-1)) # rad/s
    ω1_scale = π/2 / quadgk(ω1_exc_, 0, TRF_exc)[1] # rad/s
    ω1_exc(t) = ω1_exc_(t) * ω1_scale
    u_exc = RF_pulse_propagator(ω1_exc, B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
    u_te = exp(hamiltonian_linear(0, B1, ω0, TE - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc = u_te * u_exc

    # relaxation blocks
    u_ti = [exp(hamiltonian_linear(0, B1, ω0, iTI - (TRF_inv + TRF_exc) / 2, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]
    u_td = exp(hamiltonian_linear(0, B1, ω0, TD - TRF_inv / 2 - TE, m0s, R1f, R2f, Rx, R1s, 1))

    s = similar(TI)
    for i in eachindex(TI)
        U = u_exc * u_sp * u_ti[i] * u_sp * u_inv * u_sp * u_td * u_sp
        s[i] = steady_state(U)[1]
    end

    # Fitted model incl. κ per private communications with Dr. Deichmann
    κ = 1.964
    model2(t, p) = p[1] .* (1 .- κ .* exp.(-p[2] * t))
    fit = curve_fit(model2, TI, s, [1, 0.8])
    return 1 / fit.param[end]
end

push!(T1a_meas, 881e-3) # s; median of WM ROIs; mean is 882ms
push!(T1a_func, calculate_T1a_IRPreibisch)
push!(seq_name, "IR Preibisch et al.")
push!(seq_type, :IR)

## #########################################################
# IR: Shin et al. (https://doi.org/10.1002/mrm.21836)
############################################################
function calculate_T1a_IRShin(m0s, R1f, R2f, Rx, R1s, T2s)
    TI = exp.(range(log(34e-3), log(15), 10)) # Authors did not recall the TIs, but said they had at least 3–4 short times
    TR = 30

    # EPI readout
    TRF_exc = 2.56e-3 # from Shin's memory
    u_exc = RF_pulse_propagator(sinc_pulse(16 / 180 * π, TRF_exc; nLobes=3), B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)

    # adiabatic inversion pulse
    ω1, _, φ, TRF_inv = sech_inversion_pulse() # Shin confirmed "standard Siemens" adiabatic inversion pulse
    u_inv = RF_pulse_propagator(ω1, B1, φ, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)


    # relaxation blocks
    u_rti = [exp(hamiltonian_linear(0, B1, ω0, iTI - (TRF_inv + TRF_exc) / 2, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]
    u_rtd = [exp(hamiltonian_linear(0, B1, ω0, TR - iTI - (TRF_inv + TRF_exc) / 2, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]

    s = similar(TI)
    for i in eachindex(TI)
        U = u_exc * u_sp * u_rti[i] * u_sp * u_inv * u_sp * u_rtd[i] * u_sp
        s[i] = steady_state(U)[1]
    end

    model3(t, p) = p[1] .- p[2] .* exp.(-p[3] * t)
    fit = curve_fit(model3, TI, s, [1, 2, 0.8])
    return 1 / fit.param[end]
end

push!(T1a_meas, 0.943) # ± 0.057s in WM
push!(T1a_func, calculate_T1a_IRShin)
push!(seq_name, "IR Shin et al.")
push!(seq_type, :IR)

## #########################################################
# LL: Shin et al. (https://doi.org/10.1002/mrm.21836)
############################################################
function calculate_T1a_LLShin(m0s, R1f, R2f, Rx, R1s, T2s)
    Nslices = 28 # (inner loop)
    iSlice = Nslices - 18 # guessed from cf. Fig. 6 and 7, the author suggested that the slices were acquired in ascending order

    # adiabatic inversion pulse
    ω1, _, φ, TRF_inv = sech_inversion_pulse() # Shin confirmed "standard Siemens" adiabatic inversion pulse
    u_inv = RF_pulse_propagator(ω1, B1, φ, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)


    TI1 = 12e-5 # the author said < 12ms
    TD = 10 + TI1 # time duration of data acquisition per IR period in s
    TR = 0.4 # s
    TR_slice = TR / Nslices
    # TI = (TI1:TR:TD) #.+ (iSlice - 1) * TR_slice
    TI = (0:TR:TD-TR)

    α_exc = 16 * π / 180
    TRF_exc = 2.56e-3 # s – from the authors' memory
    nLobes = 3
    Δω0 = (nLobes + 1) * 2π / TRF_exc # rad/s
    ω0slice = ((1:Nslices) .- iSlice) * Δω0

    u_exc = Vector{Matrix{Float64}}(undef, length(ω0slice))
    Threads.@threads for is ∈ eachindex(ω0slice)
        if is == iSlice
            u_exc[is] = RF_pulse_propagator(sinc_pulse(α_exc, TRF_exc; nLobes=nLobes), B1, ω0slice[is] + ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
        else # use Graham's model for off-resonant pulses for speed
            u_exc[is] = exp(hamiltonian_linear(0, B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, 1))
            u_exc[is][5, 5] *= exp(-π * quadgk(t -> sinc_pulse(α_exc, TRF_exc; nLobes=nLobes)(t)^2, 0, TRF_exc)[1] * MRIgeneralizedBloch.lineshape_superlorentzian(ω0slice[is] + ω0, T2s))
        end
    end

    U = exp(hamiltonian_linear(0, B1, ω0, TI1 - TRF_inv / 2 - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1))
    for _ ∈ TI, is ∈ eachindex(ω0slice)
        U = u_exc[is] * u_sp * U
        U = exp(hamiltonian_linear(0, B1, -ω0slice[is] + ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, 1)) * U # rewind phase
        U = exp(hamiltonian_linear(0, B1, ω0, TR_slice - 2TRF_exc, m0s, R1f, R2f, Rx, R1s, 1)) * U
    end
    U = u_sp * u_inv * u_sp * U
    m = steady_state(U)

    s = similar(TI)
    m = exp(hamiltonian_linear(0, B1, ω0, TI1 - TRF_inv / 2 - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1)) * m
    for iTI ∈ eachindex(s), is ∈ eachindex(ω0slice)
        m = u_exc[is] * u_sp * m
        m = exp(hamiltonian_linear(0, B1, -ω0slice[is] + ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, 1)) * m # rewind phase
        m = exp(hamiltonian_linear(0, B1, ω0, TR_slice - 2TRF_exc, m0s, R1f, R2f, Rx, R1s, 1)) * m
        if is == iSlice
            s[iTI] = m[1]
        end
    end

    model3(t, p) = p[1] .- p[2] .* exp.(-p[3] * t)
    fit = curve_fit(model3, TI, s, [1, 2, 0.8])
    R1a_est = fit.param[end] + log(cos(α_exc)) / TR
    return 1 / R1a_est
end

push!(T1a_meas, 0.964) # ± 116s in WM
push!(T1a_func, calculate_T1a_LLShin)
push!(seq_name, "LL Shin et al.")
push!(seq_type, :LL)


## #########################################################
# IR: Lu et al. (https://doi.org/10.1002/jmri.20356); reported T1 = 748 ± 64 ms in the splenium of the CC and 699 ± 38 in WM
############################################################
function calculate_T1a_IRLu(m0s, R1f, R2f, Rx, R1s, T2s)
    TRF_exc = 1e-3 # 0.5-2 ms according to P Zijl
    TI = [180, 630, 1170, 1830, 2610, 3450, 4320, 5220, 6120, 7010] .* 1e-3
    TD = 8
    TE = 42e-3

    # excitation block; GRASE RO w/ TSE factor 4
    u_exc = RF_pulse_propagator(sinc_pulse(π / 2, TRF_exc; nLobes=3), B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model; spoiler=true)
    u_ref = RF_pulse_propagator(sinc_pulse(π, TRF_exc; nLobes=3), B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model; spoiler=false)
    u_te1 = exp(hamiltonian_linear(0, B1, ω0, TE / 4 - TRF_exc, m0s, R1f, R2f, Rx, R1s, 1))
    u_te234 = exp(hamiltonian_linear(0, B1, ω0, TE / 4 - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc = u_te234 * u_ref * u_te234^2 * u_ref * u_te1 * u_exc # 2 refocusing pulses before the RO

    # adiabatic inversion pulse
    # ω1, _, φ, TRF_inv = sech_inversion_pulse(ω₁ᵐᵃˣ=13e-6 * 267.522e6 * 750 / 891.1768018407862) # nom. α = 750deg according to P. Zijl
    ω1, _, φ, TRF_inv = sech_inversion_pulse(ω₁ᵐᵃˣ=4965.910769033364 * 750 / 360) # nom. α = 750deg according to P. Zijl
    u_inv = RF_pulse_propagator(ω1, B1, φ, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)


    # relaxation blocks
    u_ti = [exp(hamiltonian_linear(0, B1, ω0, iTI - (TRF_inv + TRF_exc) / 2, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]

    u_et = u_te234 * u_ref * u_te234^2 * u_ref * u_te234 # 2 refocusing pulses after the RO
    u_td = [exp(hamiltonian_linear(0, B1, ω0, TD - 2TE - TRF_inv / 2, m0s, R1f, R2f, Rx, R1s, 1)) * u_et for _ ∈ TI]

    s = similar(TI)
    for i in eachindex(TI)
        U = u_exc * u_sp * u_ti[i] * u_sp * u_inv * u_sp * u_td[i] * u_sp
        s[i] = abs(steady_state(U)[1])
    end

    model3(t, p) = abs.(p[1] .* (1 .- p[2] .* exp.(-p[3] * t)))
    fit = curve_fit(model3, TI, s, [1, 2, 0.8])
    return 1 / fit.param[end]
end

push!(T1a_meas, 0.735) # median of WM ROIs
push!(T1a_func, calculate_T1a_IRLu)
push!(seq_name, "IR Lu et al.")
push!(seq_type, :IR)


## #########################################################
# LL: Stikhov et al. (https://doi.org/10.1002/mrm.25135)
############################################################
function calculate_T1a_LLStikhov(m0s, R1f, R2f, Rx, R1s, T2s)
    TR = 1.55
    TI = [30e-3, 530e-3, 1.03, 1.53]

    TRF_inv = 720e-6 # s; for 180deg pulse, 90deg pulse are half as long
    u_90  = MRIgeneralizedBloch.xs_destructor(nothing) * RF_pulse_propagator(π / TRF_inv, B1, ω0, TRF_inv / 2, m0s, R1f, R2f, Rx, R1s, T2s, MT_model, spoiler=true)
    u_inv = MRIgeneralizedBloch.xs_destructor(nothing) * RF_pulse_propagator(π / TRF_inv, B1, ω0, TRF_inv,     m0s, R1f, R2f, Rx, R1s, T2s, MT_model, spoiler=false)
    u_m90 = MRIgeneralizedBloch.xs_destructor(nothing) * RF_pulse_propagator(π / TRF_inv, B1, ω0, TRF_inv / 2, m0s, R1f, R2f, Rx, R1s, T2s, MT_model, spoiler=false)
    u_rotp = MRIgeneralizedBloch.z_rotation_propagator(π/2, nothing)
    u_rotm = MRIgeneralizedBloch.z_rotation_propagator(-π/2, nothing)
    u_inv = u_rotp * u_m90 * u_rotm * u_inv * u_rotp * u_90  # 90-180-90 pattern confirmed by authors
    TRF_inv *= 2

    α_exc = 5 * π / 180
    nLobes = 7 # confirmed by authors
    TRF_exc = 2.56e-3 # s; confirmed by authors
    ω1 = sinc_pulse(α_exc, TRF_exc; nLobes=nLobes)
    u_exc = RF_pulse_propagator(ω1, B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)

    dTI = TI .- [0; TI[1:end-1]]
    dTI[1] -= (TRF_inv + TRF_exc) / 2
    dTI[2:end] .-= TRF_exc

    u_ir = [exp(hamiltonian_linear(0, B1, ω0, dTI[i], m0s, R1f, R2f, Rx, R1s, 1)) for i in eachindex(dTI)]
    u_fp = exp(hamiltonian_linear(0, B1, ω0, TR - TI[end] - (TRF_inv + TRF_exc) / 2, m0s, R1f, R2f, Rx, R1s, 1))

    U = I
    for i in eachindex(TI)
        U = u_exc * u_ir[i] * U
    end
    U = u_inv * u_fp * U
    m = steady_state(U)

    s = similar(TI)
    for i in eachindex(TI)
        m = u_exc * u_ir[i] * m
        s[i] = m[1]
    end

    # model as provided by Stikhov et al. in a private communication
    function model_num(t, p)
        any(t .!= TI) ? error() : nothing

        cα_exc = cos(α_exc)
        TI1 = TI[1]
        TI2 = TI[2] - TI[1]
        Nll = length(TI)

        tr = TR - TI1 - (Nll - 1) .* TI2 # time between last exc and inv pulse


        E1 = exp.(-TI1 ./ p[2])
        E2 = exp.(-TI2 ./ p[2])
        Er = exp.(-tr ./ p[2])

        F = (1 - E2) ./ (1 - cα_exc .* E2)
        Qnom = -F .* cα_exc .* Er .* E1 .* (1 .- (cα_exc .* E2) .^ (Nll - 1)) .- E1 .* (1 .- Er) .- E1 .+ 1
        Qdenom = 1 .+ cα_exc .* Er .* E1 .* (cα_exc .* E2) .^ (Nll - 1)
        Q = Qnom / Qdenom

        Mz = zeros(Nll)
        Msig = zeros(Nll)

        for ii = 1:Nll
            Mz[ii] = F .+ (cα_exc .* E2) .^ (ii - 1) .* (Q - F)
            Msig[ii] = p[1] .* sin(α_exc) .* Mz[ii]
        end
        return Msig
    end

    fit = curve_fit(model_num, TI, s, [1.0, 1.0])
    return fit.param[end]
end

push!(T1a_meas, 0.750) # peak of histogram; cf. https://doi.org/10.1016/j.mri.2016.08.021
push!(T1a_func, calculate_T1a_LLStikhov)
push!(seq_name, "LL Stikhov et al.")
push!(seq_type, :LL)


## #########################################################
# vFA: Stikhov et al. (https://doi.org/10.1002/mrm.25135)
############################################################
function calculate_T1a_vFAStikhov(m0s, R1f, R2f, Rx, R1s, T2s)
    α = [3, 10, 20, 30] * π / 180

    TR = 15e-3
    TRF = 2e-3 # s; confirmed by authors
    nLobes = 9 # confirmed by authors

    s = similar(α)
    for i in eachindex(α)
        u_exc = RF_pulse_propagator(sinc_pulse(α[i], TRF; nLobes=nLobes), B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
        u_fp = exp(hamiltonian_linear(0, B1, ω0, TR - TRF, m0s, R1f, R2f, Rx, R1s, 1))

        U = u_exc * u_sp * u_fp
        s[i] = steady_state(U)[1]
    end
    f = lm(@formula(Y ~ X), DataFrame(X=s ./ tan.(α), Y=s ./ sin.(α)))
    T1a_est = -TR / log(f.model.pp.beta0[2])
    return T1a_est
end

push!(T1a_meas, 1.07) # peak of histogram; cf. https://doi.org/10.1016/j.mri.2016.08.021
push!(T1a_func, calculate_T1a_vFAStikhov)
push!(seq_name, "vFA Stikhov et al.")
push!(seq_type, :vFA)


## #########################################################
# vFA: Cheng et al. (https://doi.org/10.1002/mrm.20791)
############################################################
function calculate_T1a_vFACheng(m0s, R1f, R2f, Rx, R1s, T2s)
    α = [2, 9, 19] * π / 180
    TR = 6.1e-3
    TRF = 1e-3 # guessed
    nLobes = 3 # guessed

    s = similar(α)
    for i in eachindex(α)
        u_exc = RF_pulse_propagator(sinc_pulse(α[i], TRF; nLobes=nLobes), B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
        u_fp = exp(hamiltonian_linear(0, B1, ω0, TR - TRF, m0s, R1f, R2f, Rx, R1s, 1))

        U = u_exc * u_sp * u_fp
        s[i] = steady_state(U)[1]
    end
    f = lm(@formula(Y ~ X), DataFrame(X=s ./ tan.(α), Y=s ./ sin.(α)))
    T1a_est = -TR / log(f.model.pp.beta0[2])
    return T1a_est
end

push!(T1a_meas, 1.0855) # mean of two volunteers
push!(T1a_func, calculate_T1a_vFACheng)
push!(seq_name, "vFA Cheng et al.")
push!(seq_type, :vFA)


## #########################################################
# vFA: Chavez & Stanisz (https://doi.org/10.1002/nbm.2769)
############################################################
function calculate_T1a_vFA_Chavez(m0s, R1f, R2f, Rx, R1s, T2s)
    α = [1, 40, 130, 150] * π / 180
    TR = 40e-3 # s

    s = similar(α)
    for i in eachindex(α)
        TRF = α[i] / (π/0.5e-3) # guessed, incl. constant ω1 / variable TRF
        u_exc = RF_pulse_propagator(α[i]/TRF, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, MT_model) # rect. pulse shape guessed because "slab-select gradient [...] [is] turned off"
        u_fp = exp(hamiltonian_linear(0, B1, ω0, TR - TRF, m0s, R1f, R2f, Rx, R1s, 1))

        U = u_exc * u_sp * u_fp
        s[i] = steady_state(U)[1]
    end

    # NLLS fit as described in the paper
    function vFA_signal(α, p)
        S0, B1, T1 = p
        E1 = exp(-TR / T1)
        return S0 .* sin.(B1 .* α) .* (1 - E1) ./ (1 .- cos.(B1 .* α) .* E1)
    end

    fit_vFA = curve_fit(vFA_signal, α, s, ones(3))
    return fit_vFA.param[end]
end
# calculate_T1a_vFA_Chavez(m0s, R1f, R2f, Rx, R1s, T2s)
# calculate_T1a_vFA_Chavez(fit_uncon.param[1], fit_uncon.param[2], R2f, fit_uncon.param[3], fit_uncon.param[4], T2s)
# calculate_T1a_vFA_Chavez(0, 1, 1e-9, 1e-9, 1e-9, T2s)

push!(T1a_meas, 1.044) # s; median of corpus callosum ROIs
push!(T1a_func, calculate_T1a_vFA_Chavez)
push!(seq_name, "vFA Chavez & Stanisz")
push!(seq_type, :vFA)

## #########################################################
# vFA: Preibisch et al. (https://doi.org/10.1002/mrm.21776)
############################################################
function calculate_T1a_vFAPreibisch(m0s, R1f, R2f, Rx, R1s, T2s)
    α = [4, 18] * π / 180
    TR = 7.6e-3
    TRF = 0.2e-3 # s; confirmed by Dr. Deichmann

    s = similar(α)
    for i in eachindex(α)
        u_exc = RF_pulse_propagator(α[i] / TRF, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF) / 2, m0s, R1f, R2f, Rx, R1s, 1))

        U = u_fp * u_exc * u_fp
        s[i] = steady_state(U)[1]
    end
    f = lm(@formula(Y ~ X), DataFrame(X=s .* α, Y=s ./ α))
    T1a_est = -2 * TR * f.model.pp.beta0[2]
    return T1a_est
end

push!(T1a_meas, 0.940) # s; median of ROIs; mean = 0.951s
push!(T1a_func, calculate_T1a_vFAPreibisch)
push!(seq_name, "vFA Preibisch et al.")
push!(seq_type, :vFA)


## #########################################################
# vFA - Hybrid FLASH-EPI: Preibisch et al. (http://doi.org/10.1002/mrm.21969)
############################################################
function calculate_T1a_vFAPreibisch_HYB(m0s, R1f, R2f, Rx, R1s, T2s, α, TR)
    TRF = 0.2e-3 # s; confirmed by Dr. Deichmann

    s = similar(α)
    for i in eachindex(α)
        u_exc = RF_pulse_propagator(α[i] / TRF, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF) / 2, m0s, R1f, R2f, Rx, R1s, 1))
        U = u_fp * u_exc * u_fp
        s[i] = steady_state(U)[1]
    end

    SL = (s[2]/sin(α[2]) - s[1]/sin(α[1])) / (s[2]/tan(α[2]) - s[1]/tan(α[1]))
    T1a_est = -TR / log(SL)
    return T1a_est
end

push!(T1a_meas, 0.955) # s
push!(T1a_func, (m0s, R1f, R2f, Rx, R1s, T2s) -> calculate_T1a_vFAPreibisch_HYB(m0s, R1f, R2f, Rx, R1s, T2s, [4, 22] * π / 180, 12.5e-3))
push!(seq_name, "vFA HYB12.5 Preibisch et al.")
push!(seq_type, :vFA)

push!(T1a_meas, 0.949) # s
push!(T1a_func, (m0s, R1f, R2f, Rx, R1s, T2s) -> calculate_T1a_vFAPreibisch_HYB(m0s, R1f, R2f, Rx, R1s, T2s, [4, 24] * π / 180, 15.2e-3))
push!(seq_name, "vFA HYB15.2 Preibisch et al.")
push!(seq_type, :vFA)

push!(T1a_meas, 0.959) # s
push!(T1a_func, (m0s, R1f, R2f, Rx, R1s, T2s) -> calculate_T1a_vFAPreibisch_HYB(m0s, R1f, R2f, Rx, R1s, T2s, [4, 25] * π / 180, 15.9e-3))
push!(seq_name, "vFA HYB15.9 Preibisch et al.")
push!(seq_type, :vFA)


## #########################################################
# vFA: Teixeira et al. (http://doi.org/10.1002/mrm.27442)
############################################################
function calculate_T1a_vFATeixeira(m0s, R1f, R2f, Rx, R1s, T2s, ω1rms)
    # α = (6:2:16) * π / 180  # used for Fig. 6
    # TR = 7e-3 # used for Fig. 6
    α = [6, 12, 18] * π / 180 # values provided Dr. Teixeira
    TR = 15e-3 # provided by Dr. Teixeira for Fig. 7

    TRF = 3e-3 # s; confirmed by Dr. Teixeira
    ω0_CSMT = 6e3 * 2π # 6kHz confirmed by Dr. Teixeira

    s = similar(α)
    Threads.@threads for i in eachindex(α)
        u_exc = RF_pulse_propagator(CSMT_pulse(α[i], TRF, TR, ω1rms, ω0=ω0_CSMT), B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF) / 2, m0s, R1f, R2f, Rx, R1s, 1))

        U = u_fp * u_exc * u_fp
        s[i] = steady_state(U)[1]
    end
    f = lm(@formula(Y ~ X), DataFrame(X=s ./ tan.(α), Y=s ./ sin.(α))) # DESPOT1 confirmed by Dr. Teixeira
    T1a_est = -TR / log(f.model.pp.beta0[2])
    return T1a_est
end

push!(T1a_meas, 0.825) # s; # read from Fig. 7
push!(T1a_func, (m0s, R1f, R2f, Rx, R1s, T2s) -> calculate_T1a_vFATeixeira(m0s, R1f, R2f, Rx, R1s, T2s, 0.4e-6 * 267.522e6)) # rad/s
push!(seq_name, "vFA CSMT w/ B1rms = 0.4uT Teixeira et al.")
push!(seq_type, :vFA)

push!(T1a_meas, 0.775) # s; # read from Fig. 7
push!(T1a_func, (m0s, R1f, R2f, Rx, R1s, T2s) -> calculate_T1a_vFATeixeira(m0s, R1f, R2f, Rx, R1s, T2s, 0.8e-6 * 267.522e6)) # rad/s
push!(seq_name, "vFA CSMT w/ B1rms = 0.8uT Teixeira et al.")
push!(seq_type, :vFA)

push!(T1a_meas, 0.73) # s; # read from Fig. 7
push!(T1a_func, (m0s, R1f, R2f, Rx, R1s, T2s) -> calculate_T1a_vFATeixeira(m0s, R1f, R2f, Rx, R1s, T2s, 1.2e-6 * 267.522e6)) # rad/s
push!(seq_name, "vFA CSMT w/ B1rms = 1.2uT Teixeira et al.")
push!(seq_type, :vFA)

push!(T1a_meas, 0.68) # s; # read from Fig. 7
push!(T1a_func, (m0s, R1f, R2f, Rx, R1s, T2s) -> calculate_T1a_vFATeixeira(m0s, R1f, R2f, Rx, R1s, T2s, 1.6e-6 * 267.522e6)) # rad/s
push!(seq_name, "vFA CSMT w/ B1rms = 1.6uT Teixeira et al.")
push!(seq_type, :vFA)

push!(T1a_meas, 0.64) # s; # read from Fig. 7
push!(T1a_func, (m0s, R1f, R2f, Rx, R1s, T2s) -> calculate_T1a_vFATeixeira(m0s, R1f, R2f, Rx, R1s, T2s, 2e-6 * 267.522e6)) # rad/s
push!(seq_name, "vFA CSMT w/ B1rms = 2uT Teixeira et al.")
push!(seq_type, :vFA)


## #########################################################
# MP2RAGE: Marques et al. (https://doi.org/10.1016/j.neuroimage.2009.10.002)
############################################################
function calculate_T1a_MP2RAGE(m0s, R1f, R2f, Rx, R1s, T2s)
    TRl = 6.75
    TR_FLASH = 7.9e-3
    TI = [0.8, 3.2]
    Nz = 160 ÷ 3

    # adiabatic inversion pulse
    ω1, _, φ, TRF_inv = sechn_inversion_pulse(n=8, ω₁ᵐᵃˣ=25e-6 * 267.522e6) # HS8 pulse confirmed by Dr. Marques; amplitude chosen close to the max. of a typical 3T system
    u_inv = u_sp * RF_pulse_propagator(ω1, B1, φ, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model) * u_sp


    ta = TI[1] - Nz / 2 * TR_FLASH - TRF_inv / 2
    tb = TI[2] - TI[1] - Nz * TR_FLASH
    tc = TRl - TI[2] - Nz / 2 * TR_FLASH - TRF_inv / 2

    u_ta = exp(hamiltonian_linear(0, B1, ω0, ta, m0s, R1f, R2f, Rx, R1s, 1))
    u_tb = exp(hamiltonian_linear(0, B1, ω0, tb, m0s, R1f, R2f, Rx, R1s, 1))
    u_tc = exp(hamiltonian_linear(0, B1, ω0, tc, m0s, R1f, R2f, Rx, R1s, 1))

    # excitation blocks
    α = [4, 5] .* π / 180

    # binomial water excitation pulses; 1-2-1 pulse scheme confirmed for the Siemens product sequence; not specifically for the prototype.
    TRF_bin = 0.2e-3 # guessed, but has little influence the estimated T1
    τ = 1 / (2 * 430) - TRF_bin # fat-water shift = 440Hz
    TRF_exc = 2τ + 3TRF_bin

    u_1 = RF_pulse_propagator(α[1] / 4 / TRF_bin, B1, ω0, TRF_bin, m0s, R1f, R2f, Rx, R1s, T2s, MT_model, spoiler=false)
    u_2 = RF_pulse_propagator(2α[1] / 4 / TRF_bin, B1, ω0, TRF_bin, m0s, R1f, R2f, Rx, R1s, T2s, MT_model, spoiler=false)
    u_t = exp(hamiltonian_linear(0, B1, ω0, τ, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc1 = u_1 * u_t * u_2 * u_t * u_1

    u_1 = RF_pulse_propagator(α[2] / 4 / TRF_bin, B1, ω0, TRF_bin, m0s, R1f, R2f, Rx, R1s, T2s, MT_model, spoiler=false)
    u_2 = RF_pulse_propagator(2α[2] / 4 / TRF_bin, B1, ω0, TRF_bin, m0s, R1f, R2f, Rx, R1s, T2s, MT_model, spoiler=false)
    u_t = exp(hamiltonian_linear(0, B1, ω0, τ, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc2 = u_1 * u_t * u_2 * u_t * u_1

    u_te = exp(hamiltonian_linear(0, B1, ω0, (TR_FLASH - TRF_exc) / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc1 = u_te * u_exc1 * u_sp * u_te
    u_exc2 = u_te * u_exc2 * u_sp * u_te

    # Propagation matrix in temporal order:
    # U = u_tc * u_exc2^Nz * u_tb * u_exc1^Nz * u_ta * u_inv
    U1 = u_exc1^(Nz / 2) * u_ta * u_inv * u_tc * u_exc2^Nz * u_tb * u_exc1^(Nz / 2)
    U2 = u_exc2^(Nz / 2) * u_tb * u_exc1^Nz * u_ta * u_inv * u_tc * u_exc2^(Nz / 2)

    s1 = steady_state(U1)[1]
    s2 = steady_state(U2)[1]
    sm = s1' * s2 / (abs(s1)^2 + abs(s2)^2)

    function MP2RAGE_signal(T1)
        eff_inv = 0.96 # from paper

        E1 = exp(-TR_FLASH / T1)
        EA = exp(-ta / T1)
        EB = exp(-tb / T1)
        EC = exp(-tc / T1)

        mzss = (((((1 - EA) * (cos(α[1]) * E1)^Nz + (1 - E1) * (1 - (cos(α[1]) * E1)^Nz) / (1 - cos(α[1]) * E1)) * EB + (1 - EB)) * (cos(α[2]) * E1)^Nz + (1 - E1) * (1 - (cos(α[2]) * E1)^Nz) / (1 - cos(α[2]) * E1)) * EC + (1 - EC)) / (1 + eff_inv * (cos(α[1]) * cos(α[2]))^Nz * exp(-TRl / T1))

        s1 = sin(α[1]) * ((-eff_inv * mzss * EA + (1 - EA)) * (cos(α[1]) * E1)^(Nz / 2 - 1) + (1 - E1) * (1 - (cos(α[1]) * E1)^(Nz / 2 - 1)) / (1 - cos(α[1]) * E1))
        s2 = sin(α[2]) * ((mzss - (1 - EC)) / (EC * (cos(α[2]) * E1)^(Nz / 2)) - (1 - E1) * ((cos(α[2]) * E1)^(-Nz / 2) - 1) / (1 - cos(α[2]) * E1))

        sm = s1' * s2 / (abs(s1)^2 + abs(s2^2))
        return sm
    end

    fit = curve_fit((_, T1) -> MP2RAGE_signal.(T1), [1], [sm], [0.5])
    return fit.param[1]
end

push!(T1a_meas, 0.81) # +- 0.03
push!(T1a_func, calculate_T1a_MP2RAGE)
push!(seq_name, "MP2RAGE Marques et al.")
push!(seq_type, :MP2RAGE)


## #########################################################
# MPRAGE: Wright et al. (http://doi.org/10.1007/s10334-008-0104-8)
############################################################
function calculate_T1a_MPRAGE_Wright(m0s, R1f, R2f, Rx, R1s, T2s)
    TRl = 5
    TR_FLASH = 11e-3
    TE = 6.7e-3
    TI = [160, 190, 285, 441, 680, 1050, 1619, 2100] .* 1e-3 # s
    Nz = 256

    # adiabatic inversion pulse
    TRF_inv = 13.5e-3 # taken from the paper
    β = 600 # picked for 10kHz bandwidth
    μ = 5  # 50 would match 10kHz bandwidth, chosen 5 for computation speed (makes little difference)
    ω₁ᵐᵃˣ = 4 * sqrt(μ) * β # compromise of appromximating 1.25 >> 1 and keeping B1max in limits
    ω1, _, φ, TRF_inv = sech_inversion_pulse(TRF=TRF_inv, β=β, μ=μ, ω₁ᵐᵃˣ=ω₁ᵐᵃˣ) # standard Philips inverson pulse, likely hyperbolic secant, as confirmed by Dr. Gowland
    u_inv = u_sp * RF_pulse_propagator(ω1, B1, φ, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)

    # excitation blocks
    α = (8/20:8/20:8) .* π / 180 # pattern confirmed by Dr. Gowland
    TRF_exc = 0.67e-3 # s
    nLobes = 7 # sinc pulses confirmed by Dr. Gowland; number of lobes guessed guessed to approximate the 11.9kHz bandwidth discussed in the paper

    u_exc = [RF_pulse_propagator(sinc_pulse(α[i], TRF_exc; nLobes=nLobes), B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model) for i in eachindex(α)]
    u_te = exp(hamiltonian_linear(0, B1, ω0, TE - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_tr = exp(hamiltonian_linear(0, B1, ω0, TR_FLASH - TE - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc = [u_te * u_exc[i] * u_tr for i in eachindex(u_exc)]
    u_exc_ramp = prod(u_exc[end:-1:1])

    s = similar(TI)
    for iTI in eachindex(TI)
        ti = TI[iTI] - TRF_inv / 2 - (TR_FLASH - TE) - length(α) * TR_FLASH
        tc = TRl - TI[iTI] - Nz * TR_FLASH - TRF_inv / 2 - TE + length(α) * TR_FLASH

        u_ti = exp(hamiltonian_linear(0, B1, ω0, ti, m0s, R1f, R2f, Rx, R1s, 1))
        u_tc = exp(hamiltonian_linear(0, B1, ω0, tc, m0s, R1f, R2f, Rx, R1s, 1))

        # Propagation matrix in temporal order: U = u_tc * u_exc20^(Nz-20) * ... u_exc2 * u_exc1 * u_ti * u_inv
        U = u_exc_ramp * u_ti * u_inv * u_tc * u_exc[end]^(Nz - length(α))
        s[iTI] = steady_state(U)[1] # extract x-magnetization
    end

    function MPRAGE_mz(TI, p)
        T1, M0, α_inv = p

        function hamiltonian_T1(T, R1)
            H = @SMatrix [
                -R1  R1;
                  0   0]
            return H * T
        end
        function pulse_propgator(α)
            U = @SMatrix [
                cos(α)  0;
                     0  1]
            return U
        end
        function steady_state_2D(U)
            Q = U - @SMatrix [1 0; 0 0]
            m = Q \ @SVector [0,1]
            return m
        end

        s = similar(TI)
        for iTI in eachindex(TI)
            ti = TI[iTI] - length(α) * TR_FLASH
            tc = TRl - TI[iTI] - Nz * TR_FLASH + length(α) * TR_FLASH

            u_ti = exp(hamiltonian_T1(ti, 1/T1))
            u_tr = exp(hamiltonian_T1(TR_FLASH, 1/T1))
            u_tc = exp(hamiltonian_T1(tc, 1/T1))

            # Propagation matrix in temporal order: U = u_tc * u_exc20^(Nz-20) * ... u_exc2 * u_exc1 * u_ta * u_inv
            U = u_tr * pulse_propgator(α[end])
            for i in (length(α)-1):-1:1
                U = U * u_tr * pulse_propgator(α[i])
            end
            U = U * u_ti * pulse_propgator(α_inv) * u_tc * (u_tr * pulse_propgator(α[end]))^(Nz - length(α))
            s[iTI] = M0 * steady_state_2D(U)[1] # extract z-magnetization
        end
        return s
    end

    fit = curve_fit(MPRAGE_mz, TI, s, [1, sin(α[end]), 0.9π])
    return fit.param[1]
end

push!(T1a_meas, 0.84) # s
push!(T1a_func, calculate_T1a_MPRAGE_Wright)
push!(seq_name, "MPRAGE Wright et al.")
push!(seq_type, :MP2RAGE)


## #########################################################
# IR w/ adiabatic inversion pulse: Wright et al. (http://doi.org/10.1007/s10334-008-0104-8)
############################################################
function calculate_T1a_IR_EPI_Wright(m0s, R1f, R2f, Rx, R1s, T2s)
    TI = [120, 200, 400, 600, 900, 1500, 2100, 3000, 4000] .* 1e-3 # s
    TR = 35 # s
    TE = 45e-3 # s
    TRF_exc = 7.7e-3 # s
    nLobes = 1 # chose to match 395 Hz bandwidth

    # adiabatic inversion pulse
    TRF_inv = 17.51e-3
    β = 500 # chosen to fit 713Hz bandwidth
    μ = 5   # chosen to fit 713Hz bandwidth
    ω₁ᵐᵃˣ = 2 * sqrt(μ) * β # compromise of appromximating 2 >> 1 and keeping B1max in limits
    ω1, _, φ, TRF_inv = sech_inversion_pulse(TRF=TRF_inv, β=β, μ=μ, ω₁ᵐᵃˣ=ω₁ᵐᵃˣ)
    u_inv = u_sp * RF_pulse_propagator(ω1, B1, φ, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)


    # relaxation blocks
    u_ti = [exp(hamiltonian_linear(0, B1, ω0, iTI - TRF_inv/2 - TRF_exc/2, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]
    u_td = [exp(hamiltonian_linear(0, B1, ω0,  TR - TRF_inv/2 - iTI - TE , m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]

    # excitation block
    u_exc = RF_pulse_propagator(sinc_pulse(π / 2, TRF_exc; nLobes=nLobes), B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
    u_te = exp(hamiltonian_linear(0, B1, ω0, TE - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc = u_te * u_exc

    s = similar(TI)
    for i in eachindex(TI)
        U = u_exc * u_sp * u_ti[i] * u_sp * u_inv * u_sp * u_td[i] * u_sp
        s[i] = steady_state(U)[1]
    end

    model3(t, p) = p[1] .* (1 .- p[2] .* exp.(-p[3] * t)) # p[2] = (1 - cos(α))
    fit = curve_fit(model3, TI, s, [1, 2, 0.8])
    return 1 / fit.param[end]
end

push!(T1a_meas, 0.9) # read from Fig. 5
push!(T1a_func, calculate_T1a_IR_EPI_Wright)
push!(seq_name, "IR EPI Wright et al.")
push!(seq_type, :IR)


## #########################################################
# IR w/ adiabatic inversion pulse: Reynolds et al. (https://doi.org/10.1002/nbm.4936)
############################################################
function calculate_T1a_IRReynolds_adiabatic(m0s, R1f, R2f, Rx, R1s, T2s)
    TRF_exc = 1e-3 # guessed
    nLobes = 3 # guessed
    TI = [5.5, 10.2, 35.8, 66.9, 125, 234, 598, 818, 1118, 1529, 3910, 5348] .* 1e-3 # measured from end to beginning of respective pulse (confirmed by Dr. Reynolds)
    TD = 5 # s
    TE = 10e-3 # s; guessed, but has negligible impact

    # adiabatic inversion pulse
    # ω1, _, φ, TRF_inv = sech_inversion_pulse(TRF=10e-3, ω₁ᵐᵃˣ=13.5e-6 * 267.522e6, μ=10 * 1.154911144648012, β=730)
    ω1, _, φ, TRF_inv = sech_inversion_pulse(TRF=10e-3, ω₁ᵐᵃˣ=13.5e-6 * 267.522e6, μ=1.8380981750265004, β=730)
    u_inv = RF_pulse_propagator(ω1, B1, φ, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)


    # relaxation blocks
    u_ti = [exp(hamiltonian_linear(0, B1, ω0, iTI, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]
    u_td = exp(hamiltonian_linear(0, B1, ω0, TD - TE, m0s, R1f, R2f, Rx, R1s, 1))

    # excitation block
    u_exc = RF_pulse_propagator(sinc_pulse(π / 2, TRF_exc; nLobes=nLobes), B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
    u_te = exp(hamiltonian_linear(0, B1, ω0, TE - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc = u_te * u_exc

    s = similar(TI)
    for i in eachindex(TI)
        U = u_exc * u_sp * u_ti[i] * u_sp * u_inv * u_sp * u_td * u_sp
        s[i] = steady_state(U)[1]
    end

    model3(t, p) = p[1] .* (1 .- p[2] .* exp.(-p[3] * t)) # confirmed by Dr. Reynolds
    fit = curve_fit(model3, TI, s, [1, 2, 0.8])
    return 1 / fit.param[end]
end

push!(T1a_meas, 0.905)
push!(T1a_func, calculate_T1a_IRReynolds_adiabatic)
push!(seq_name, "IR ad. Reynolds et al.")
push!(seq_type, :IR)


## #########################################################
# IR w/ sinc inversion pulse: Reynolds et al. (https://doi.org/10.1002/nbm.4936)
############################################################
function calculate_T1a_IRReynolds_sinc(m0s, R1f, R2f, Rx, R1s, T2s)
    TRF_exc = 1e-3 # guessed
    nLobes_exc = 3 # guessed
    TI = [5.5, 10.2, 35.8, 66.9, 125, 234, 598, 818, 1118, 1529, 3910, 5348] .* 1e-3 # measured from end to beginning of respective pulse (confirmed by Dr. Reynolds)
    TD = 5 # s
    TE = 10e-3 # s; guessed, but has negligible impact

    # excitation block
    u_exc = RF_pulse_propagator(sinc_pulse(π / 2, TRF_exc; nLobes=nLobes_exc), B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
    u_te = exp(hamiltonian_linear(0, B1, ω0, TE - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc = u_te * u_exc

    # sinc inversion pulse
    TRF_inv = 3e-3 # s
    nLobes_inv = 3
    u_inv = RF_pulse_propagator(sinc_pulse(π, TRF_inv; nLobes=nLobes_inv), B1, ω0, TRF_inv, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)

    # relaxation blocks
    u_ti = [exp(hamiltonian_linear(0, B1, ω0, iTI, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]
    u_td = exp(hamiltonian_linear(0, B1, ω0, TD - TE, m0s, R1f, R2f, Rx, R1s, 1))

    s = similar(TI)
    for i in eachindex(TI)
        U = u_exc * u_sp * u_ti[i] * u_sp * u_inv * u_sp * u_td * u_sp
        s[i] = steady_state(U)[1]
    end

    model3(t, p) = p[1] .* (1 .- p[2] .* exp.(-p[3] * t)) # confirmed by Rd. Reynolds
    fit = curve_fit(model3, TI, s, [1, 2, 0.8])
    return 1 / fit.param[end]
end

push!(T1a_meas, 0.861)
push!(T1a_func, calculate_T1a_IRReynolds_sinc)
push!(seq_name, "IR sinc Reynolds et al.")
push!(seq_type, :IR)


## #########################################################
# saturation recovery: Reynolds et al. (https://doi.org/10.1002/nbm.4936)
############################################################
function calculate_T1a_SRReynolds(m0s, R1f, R2f, Rx, R1s, T2s)
    TRF_exc = 1e-3 # s; guessed, but has negligible impact
    nLobes_exc = 5 # guessed, but has negligible impact
    TI = [5.5, 10.2, 35.8, 66.9, 125, 234, 598, 818, 1118, 1529, 3910, 5348] .* 1e-3 # s; measured from end to beginning of respective pulse (confirmed by Dr. Reynolds)
    TD = 5 # s
    TE = 10e-3 # s; guessed, but has negligible impact

    # saturation pulse
    TRF_sat = 0.5 # s
    ω1 = 10 * 2π # rad/s
    u_sat = RF_pulse_propagator(ω1, B1, ω0, TRF_sat, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)

    # relaxation blocks
    u_ti = [exp(hamiltonian_linear(0, B1, ω0, iTI, m0s, R1f, R2f, Rx, R1s, 1)) for iTI ∈ TI]
    u_td = exp(hamiltonian_linear(0, B1, ω0, TD - TE, m0s, R1f, R2f, Rx, R1s, 1))

    # excitation block
    u_exc = RF_pulse_propagator(sinc_pulse(π / 2, TRF_exc; nLobes=nLobes_exc), B1, ω0, TRF_exc, m0s, R1f, R2f, Rx, R1s, T2s, MT_model)
    u_te = exp(hamiltonian_linear(0, B1, ω0, TE - TRF_exc / 2, m0s, R1f, R2f, Rx, R1s, 1))
    u_exc = u_te * u_exc

    s = similar(TI)
    for i in eachindex(TI)
        U = u_exc * u_ti[i] * u_sp * u_sat * u_td * u_sp
        s[i] = steady_state(U)[1]
    end

    model3(t, p) = p[1] .* (1 .- p[2] .* exp.(-p[3] * t)) # confirmed by Dr. Reynolds
    fit = curve_fit(model3, TI, s, [1, 2, 0.8])
    return 1 / fit.param[end]
end

push!(T1a_meas, 1.013)
push!(T1a_func, calculate_T1a_SRReynolds)
push!(seq_name, "SR Reynolds et al.")
push!(seq_type, :SR)
















## #########################################################
# fit models
############################################################
B1 = 1
ω0 = 0
p = plot([0.6, 1.15], [0.6, 1.15], xlabel="T1a simulated", ylabel="T1a measured", legend=:topleft, label=:none)
# marker_list = [:circle, :cross, :diamond, :dtriangle, :rect, :utriangle, :x, :pentagon, :hexagon]
marker_list = [(seq_type_i == :IR) ? (:circle) : ((seq_type_i == :LL) ? (:cross) : ((seq_type_i == :vFA) ? (:diamond) : ((seq_type_i == :SR) ? (:dtriangle) : (:x)))) for seq_type_i in seq_type]

fit_name = String[]
T1a_sim = Array{Float64}[]
ΔAIC_v = Float64[]
ΔBIC_v = Float64[]

## #########################################################
# fit mono-exponential model w/ generalized Bloch model
############################################################
MT_model = Graham() # irrelevant for a mono-exponential model; choosing Graham's model for speed purposes
push!(fit_name, "mono_exp")
m0s, R1f, R2f, Rx, R1s, T2s = WM_param_monoExp()
T2s = 0

function model(iseq, p)
    R1f = p[1]
    T1a = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1a[i] = T1a_func[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1a
end

scatter!(p, model(1:length(T1a_meas), [R1f]), T1a_meas, label="$(fit_name[end]) init", markershape=marker_list, hover=seq_name)

fit_mono = curve_fit(model, 1:length(T1a_meas), T1a_meas, [R1f], x_tol=1e-3, show_trace=true)
fit_mono.param

push!(T1a_sim, model(1:length(T1a_meas), fit_mono.param))
scatter!(p, T1a_sim[end], T1a_meas, label="$(fit_name[end]) fit", markershape=marker_list, hover=seq_name)

n = length(T1a_meas)
k = length(fit_mono.param)
RSS = norm(fit_mono.resid)^2
AIC_0 = n * log(RSS / n) + 2k
BIC_0 = n * log(RSS / n) + k * log(n)

ΔAIC = n * log(RSS / n) + 2k - AIC_0 # = 0
ΔBIC = n * log(RSS / n) + k * log(n) - BIC_0 # = 0

push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)


## #########################################################
# fit unconstr. qMT model w/ Graham's model
############################################################
MT_model = Graham()
push!(fit_name, "unconstr_Graham")
m0s, R1f, R2f, Rx, R1s, T2s = WM_param_qMT()

function model(iseq, p)
    m0s, R1f, R1s = p
    T1a = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1a[i] = T1a_func[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1a
end

scatter!(p, model(1:length(T1a_meas), [m0s, R1f, R1s]), T1a_meas, label="$(fit_name[end]) init", markershape=marker_list, hover=seq_name)

fit_Graham = curve_fit(model, 1:length(T1a_meas), T1a_meas, [m0s, R1f, R1s], x_tol=1e-3, show_trace=true)
fit_Graham.param

push!(T1a_sim, model(1:length(T1a_meas), fit_Graham.param))
scatter!(p, T1a_sim[end], T1a_meas, label="$(fit_name[end]) fit", markershape=marker_list, hover=seq_name)

n = length(T1a_meas)
k = length(fit_Graham.param)
RSS = norm(fit_Graham.resid)^2
ΔAIC = n * log(RSS / n) + 2k - AIC_0
ΔBIC = n * log(RSS / n) + k * log(n) - BIC_0

push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)


## #########################################################
# fit constr. qMT model w/ generalized Bloch model
############################################################
MT_model = gBloch()
push!(fit_name, "constr_gBloch")
m0s, R1f, R2f, Rx, R1s, T2s = WM_param_Stanisz()

function model(iseq, p)
    m0s, R1f = p
    R1s = R1f
    T1a = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1a[i] = T1a_func[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1a
end

scatter!(p, model(1:length(T1a_meas), [m0s, R1f]), T1a_meas, label="$(fit_name[end]) init", markershape=marker_list, hover=seq_name)

fit_constr = curve_fit(model, 1:length(T1a_meas), T1a_meas, [m0s, R1f], x_tol=1e-3, show_trace=true)
fit_constr.param

push!(T1a_sim, model(1:length(T1a_meas), fit_constr.param))
scatter!(p, T1a_sim[end], T1a_meas, label="$(fit_name[end]) fit", markershape=marker_list, hover=seq_name)

n = length(T1a_meas)
k = length(fit_constr.param)
RSS = norm(fit_constr.resid)^2
ΔAIC = n * log(RSS / n) + 2k - AIC_0
ΔBIC = n * log(RSS / n) + k * log(n) - BIC_0

push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)


## #########################################################
# fit unconstr. qMT model w/ generalized Bloch model
############################################################
MT_model = gBloch()
push!(fit_name, "unconstr_gBloch")
m0s, R1f, R2f, Rx, R1s, T2s = WM_param_qMT()

function model(iseq, p)
    m0s, R1f, R1s = p
    T1a = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1a[i] = T1a_func[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1a
end

scatter!(p, model(1:length(T1a_meas), [m0s, R1f, R1s]), T1a_meas, label="$(fit_name[end]) init", markershape=marker_list, hover=seq_name)

fit_uncon = curve_fit(model, 1:length(T1a_meas), T1a_meas, [m0s, R1f, R1s], x_tol=1e-3, show_trace=true)
fit_uncon.param

push!(T1a_sim, model(1:length(T1a_meas), fit_uncon.param))
scatter!(p, T1a_sim[end], T1a_meas, label="$(fit_name[end]) fit", markershape=marker_list, hover=seq_name)

n = length(T1a_meas)
k = length(fit_uncon.param)
RSS = norm(fit_uncon.resid)^2
ΔAIC = n * log(RSS / n) + 2k - AIC_0
ΔBIC = n * log(RSS / n) + k * log(n) - BIC_0

push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)









# not used for the paper...
## #########################################################
# fit unconstr. qMT model w/ Sled's model
############################################################
# MT_model = Sled()
# push!(fit_name, "unconstr_Sled")
# m0s, R1f, R2f, Rx, R1s, T2s = WM_param_qMT()

# function model(iseq, p)
#     m0s, R1f, Rx, R1s = p
#     T1a = similar(iseq, Float64)
#     Threads.@threads for i in eachindex(iseq)
#         T1a[i] = T1a_func[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
#     end
#     return T1a
# end

# scatter!(p, model(1:length(T1a_meas), [m0s, R1f, Rx, R1s]), T1a_meas, label="$(fit_name[end]) init", markershape=marker_list, hover=seq_name)

# fit_Sled = curve_fit(model, (1:length(T1a_meas))[incl_fit], T1a_meas[incl_fit], [m0s, R1f, Rx, R1s], show_trace=true, maxIter=100)
# fit_Sled.param

# push!(T1a_sim, model(1:length(T1a_meas), fit_Sled.param))
# scatter!(p, T1a_sim[end], T1a_meas, label="$(fit_name[end]) fit", markershape=marker_list, hover=seq_name)

# n = length(T1a_meas)
# k = length(fit_Sled.param)
# RSS = norm(fit_Sled.resid)^2
# ΔAIC = n * log(RSS / n) + 2k - AIC_0
# ΔBIC = n * log(RSS / n) + k * log(n) - BIC_0

# push!(ΔAIC_v, ΔAIC)
# push!(ΔBIC_v, ΔBIC)


# ## #########################################################
# # fit constr. qMT model w/ Graham's model
# ############################################################
# MT_model = Graham()
# push!(fit_name, "constr_Graham")
# m0s, R1f, R2f, Rx, R1s, T2s = WM_param_Stanisz()

# function model(iseq, p)
#     m0s, R1f, Rx = p
#     R1s = R1f
#     T1a = similar(iseq, Float64)
#     Threads.@threads for i in eachindex(iseq)
#         T1a[i] = T1a_func[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
#     end
#     return T1a
# end

# scatter!(p, model(1:length(T1a_meas), [m0s, R1f, Rx]), T1a_meas, label="$(fit_name[end]) init", markershape=marker_list, hover=seq_name)

# fit_constr_Graham = curve_fit(model, (1:length(T1a_meas))[incl_fit], T1a_meas[incl_fit], [m0s, R1f, Rx], show_trace=true, maxIter=100, lower=[0.0, 0.0, 0.0])
# fit_constr_Graham.param

# push!(T1a_sim, model(1:length(T1a_meas), fit_constr_Graham.param))
# scatter!(p, T1a_sim[end], T1a_meas, label="$(fit_name[end]) fit", markershape=marker_list, hover=seq_name)

# n = length(T1a_meas)
# k = length(fit_constr_Graham.param)
# RSS = norm(fit_constr_Graham.resid)^2
# ΔAIC = n * log(RSS / n) + 2k - AIC_0
# ΔBIC = n * log(RSS / n) + k * log(n) - BIC_0

# push!(ΔAIC_v, ΔAIC)
# push!(ΔBIC_v, ΔBIC)




## #########################################################
# data analysis
############################################################
using StatsBase
@info "CoV(T1_measured): " variation(T1a_meas)
@info "MAD(T1_measured): " mad(T1a_meas) / median(T1a_meas)
@info "Variability is reduced by " 1 - mad(fit_uncon.resid) / mad(T1a_meas)


@info "Variability is reduced by " mad(fit_mono.resid) / mad(T1a_meas)
@info "Variability is reduced by " mad(fit_Graham.resid) / mad(T1a_meas)
@info "Variability is reduced by " mad(fit_constr.resid) / mad(T1a_meas)
@info "Variability is reduced by " mad(fit_uncon.resid) / mad(T1a_meas)

@info "Variability is reduced by " mad(fit_mono.resid)   / mad(fit_mono.resid)
@info "Variability is reduced by " mad(fit_Graham.resid) / mad(fit_mono.resid)
@info "Variability is reduced by " mad(fit_constr.resid) / mad(fit_mono.resid)
@info "Variability is reduced by " mad(fit_uncon.resid)  / mad(fit_mono.resid)



## #########################################################
# export data
############################################################
using Printf

io = open(expanduser("~/Documents/Paper/2023_T1variablity/Figures/T1.txt"), "w")
write(io, "meas ")
[write(io, "sim_$name_i ") for name_i in fit_name]
write(io, "seq_marker ")
write(io, "\n")

for i_seq in eachindex(T1a_meas)
    write(io, @sprintf("%1.3f ", T1a_meas[i_seq]))
    [write(io, @sprintf("%1.3f ", T1a_sim[i_fit][i_seq])) for i_fit in eachindex(T1a_sim)]
    # write(io, @sprintf("%u ", marker_num[i_seq]))
    write(io, @sprintf("%s ", string(seq_type[i_seq])[1]))
    write(io, "\n")
end
close(io)

##
io = open(expanduser("~/Documents/Paper/2023_T1variablity/Figures/residuals.txt"), "w")
for i_seq in eachindex(fit_mono.resid)
    write(io, "1 ")
    write(io, @sprintf("%1.3f ", fit_mono.resid[i_seq]))
    write(io, @sprintf("%1.3f ", fit_constr.resid[i_seq]))
    write(io, @sprintf("%1.3f ", fit_uncon.resid[i_seq]))
    # write(io, @sprintf("%1.3f ", fit_Sled.resid[i_seq]))
    write(io, @sprintf("%1.3f ", fit_Graham.resid[i_seq]))
    write(io, "\n")
end
close(io)

## AIC/BIC table
println("")
print("model & constraint & \$\\Delta\$AIC & \$\\Delta\$BIC \\\\\n \\midrule \n")
print("mono-exponential   & none            & 0           & 0           \\\\\n")
i = findfirst(fit_name .== "unconstr_Graham")
print(@sprintf("Graham's & none & %1.1f & %1.1f \\\\\n", ΔAIC_v[i], ΔBIC_v[i]))
i = findfirst(fit_name .== "constr_gBloch")
print(@sprintf("generalized Bloch & \$T_1^s = T_1^f\$ & %1.1f & %1.1f \\\\\n", ΔAIC_v[i], ΔBIC_v[i]))
i = findfirst(fit_name .== "unconstr_gBloch")
print(@sprintf("generalized Bloch & none & %1.1f & %1.1f \\\\\n", ΔAIC_v[i], ΔBIC_v[i]))
print("\\bottomrule")


## fit parameter
println("")
print("model & Graham's & gen. Bloch & gen. Bloch \\\\\n \\midrule \n")
print("constraint & none & \$R_1^s = R_1^f\$ & none \\\\\n \\midrule \n")
print(@sprintf("\$m_0^s\$ & %1.2f & %1.2f & %1.2f \\\\\n", fit_Graham.param[1], fit_constr.param[1], fit_uncon.param[1]))
print(@sprintf("\$T_1^f\$ (s) & %1.2f & %1.2f & %1.2f \\\\\n", 1 / fit_Graham.param[2], 1 / fit_constr.param[2], 1 / fit_uncon.param[2]))
print(@sprintf("\$T_1^s\$ (s) & %1.2f & \\g\$T_1^f\$ & %1.2f \\\\\n", 1 / fit_Graham.param[3], 1 / fit_uncon.param[3]))
print(@sprintf("\$T_2^f\$ (ms) & \\g%1.1f & \\g%1.1f & \\g%1.1f \\\\\n", 1e3 / WM_param_qMT()[3], 1e3 / WM_param_Stanisz()[3], 1e3 / WM_param_qMT()[3]))
print(@sprintf("\$T_2^s\$ (\$upmy\$s) & \\g%1.1f & \\g%1.1f & \\g%1.1f \\\\\n", 1e6 * WM_param_qMT()[6], 1e6 * WM_param_Stanisz()[6], 1e6 * WM_param_qMT()[6]))
# print(@sprintf("\$R_\\text{x}\$ (s\$^{-1}\$) & %1.1f & %1.1f & %1.1f \\\\\n", fit_Graham.param[3], fit_constr.param[3], fit_uncon.param[3]))
print(@sprintf("\$R_\\text{x}\$ (s\$^{-1}\$) & \\g%1.1f & \\g%1.1f & \\g%1.1f \\\\\n", WM_param_qMT()[4], WM_param_Stanisz()[4], WM_param_qMT()[4]))
print("\\bottomrule")
