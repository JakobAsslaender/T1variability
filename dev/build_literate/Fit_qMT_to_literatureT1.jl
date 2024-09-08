include("helper_functions.jl")

include("T1_mapping_methods.jl")

p = plot([0.55, 1.15], [0.55, 1.15], xlabel="T1 simulated", ylabel="T1 measured", legend=:topleft, label=:none)
marker_list = [(seq_type_i == :IR) ? (:circle) : ((seq_type_i == :LL) ? (:cross) : ((seq_type_i == :vFA) ? (:diamond) : ((seq_type_i == :SR) ? (:dtriangle) : (:x)))) for seq_type_i in seq_type]

fit_name = String[]
fitted_param = NTuple{6, Float64}[]
T1_simulated = Array{Float64}[]
ΔAIC_v = Float64[]
ΔBIC_v = Float64[]

MT_model = Graham()
push!(fit_name, "mono_exp")

m0s = 0
R1f = 1 / 1.084  # 1/s
R2f = 1 / 0.0769 # 1/s
Rx = 0           # 1/s
R1s = R1f        # 1/s
T2s = 0
ω0 = 0
B1 = 1

function model(iseq, p)
    R1f = p[1]
    T1 = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1[i] = T1_functions[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1
end

fit_mono = curve_fit(model, 1:length(T1_literature), T1_literature, [R1f], x_tol=1e-3)
push!(fitted_param, (m0s, fit_mono.param[1], R2f, Rx, R1s, T2s))
push!(T1_simulated, model(1:length(T1_literature), fit_mono.param))

m0s

T1f_fitted = 1/fit_mono.param[1] # s

1/R1s # s

1/R2f # s

T2s # s

Rx # 1/s

scatter!(p, T1_simulated[end], T1_literature, label="$(fit_name[end]) model", markershape=marker_list, hover=seq_name)

n = length(T1_literature)

k = length(fit_mono.param)

RSS = norm(fit_mono.resid)^2

AIC_mono = n * log(RSS / n) + 2k

BIC_mono = n * log(RSS / n) + k * log(n)

ΔAIC = n * log(RSS / n) + 2k - AIC_mono

ΔBIC = n * log(RSS / n) + k * log(n) - BIC_mono

push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)

MT_model = Graham()
push!(fit_name, "unconstr_Graham")

m0s = 0.25
R1f = 1 / 1.84   # 1/s
R1s = 1 / 0.34   # 1/s
Rx = 13.6        # 1/s
R2f = 1 / 0.0769 # 1/s
T2s = 12.5e-6    # s
ω0 = 0
B1 = 1

function model(iseq, p)
    m0s, R1f, R1s = p
    T1 = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1[i] = T1_functions[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1
end

fit_Graham = curve_fit(model, 1:length(T1_literature), T1_literature, [m0s, R1f, R1s], x_tol=1e-3)
push!(fitted_param, (fit_Graham.param[1], fit_Graham.param[2], R2f, Rx, fit_Graham.param[3], T2s))
push!(T1_simulated, model(1:length(T1_literature), fit_Graham.param))

m0s_fitted = fit_Graham.param[1]

T1f_fitted = 1/fit_Graham.param[2] # s

T1s_fitted = 1/fit_Graham.param[3] # s

1/R2f # s

T2s # s

Rx # 1/s

scatter!(p, T1_simulated[end], T1_literature, label="$(fit_name[end]) model", markershape=marker_list, hover=seq_name)

n = length(T1_literature)

k = length(fit_Graham.param)

RSS = norm(fit_Graham.resid)^2

ΔAIC = n * log(RSS / n) + 2k - AIC_mono

ΔBIC = n * log(RSS / n) + k * log(n) - BIC_mono

push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)

MT_model = gBloch()
push!(fit_name, "constr_gBloch")

m0s = 0.139
R1f = 1 / 1.084  # 1/s
R2f = 1 / 0.0769 # 1/s
Rx = 23          # 1/s
R1s = 1          # 1/s
T2s = 12.5e-6    # s
ω0 = 0
B1 = 1

function model(iseq, p)
    m0s, R1f = p
    R1s = R1f
    T1 = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1[i] = T1_functions[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1
end

fit_constr = curve_fit(model, 1:length(T1_literature), T1_literature, [m0s, R1f], x_tol=1e-3)
push!(fitted_param, (fit_constr.param[1], fit_constr.param[2], R2f, Rx, fit_constr.param[2], T2s))
push!(T1_simulated, model(1:length(T1_literature), fit_constr.param))

m0s_fitted = fit_constr.param[1]

T1f_fitted = 1/fit_constr.param[2] # s

T1s_fitted = 1/fit_constr.param[2] # s

1/R2f # s

T2s # s

Rx # 1/s

scatter!(p, T1_simulated[end], T1_literature, label="$(fit_name[end]) model", markershape=marker_list, hover=seq_name)

n = length(T1_literature)

k = length(fit_constr.param)

RSS = norm(fit_constr.resid)^2

ΔAIC = n * log(RSS / n) + 2k - AIC_mono

ΔBIC = n * log(RSS / n) + k * log(n) - BIC_mono

push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)

MT_model = gBloch()
push!(fit_name, "unconstr_gBloch")

m0s = 0.25
R1f = 1 / 1.84   # 1/s
R1s = 1 / 0.34   # 1/s
Rx = 13.6        # 1/s
R2f = 1 / 0.0769 # 1/s
T2s = 12.5e-6    # s
ω0 = 0
B1 = 1

function model(iseq, p)
    m0s, R1f, R1s = p
    T1 = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1[i] = T1_functions[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1
end

fit_uncon = curve_fit(model, 1:length(T1_literature), T1_literature, [m0s, R1f, R1s], x_tol=1e-3, show_trace=true)
push!(fitted_param, (fit_uncon.param[1], fit_uncon.param[2], R2f, Rx, fit_uncon.param[3], T2s))
push!(T1_simulated, model(1:length(T1_literature), fit_uncon.param))

m0s_fitted = fit_uncon.param[1]

T1f_fitted = 1/fit_uncon.param[2] # s

T1s_fitted = 1/fit_uncon.param[3] # s

1/R2f # s

T2s # s

Rx # 1/s

scatter!(p, T1_simulated[end], T1_literature, label="$(fit_name[end]) model", markershape=marker_list, hover=seq_name)

n = length(T1_literature)

k = length(fit_uncon.param)

RSS = norm(fit_uncon.resid)^2

ΔAIC = n * log(RSS / n) + 2k - AIC_mono

ΔBIC = n * log(RSS / n) + k * log(n) - BIC_mono

push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)

extrema(T1_literature)

variation(T1_literature)

1 - mad(fit_mono.resid;   center=mean(fit_mono.resid))    / mad(T1_literature; center=mean(T1_literature))

1 - mad(fit_Graham.resid; center=mean(fit_Graham.resid))  / mad(T1_literature; center=mean(T1_literature))

1 - mad(fit_constr.resid; center=mean(fit_constr.resid))  / mad(T1_literature; center=mean(T1_literature))

1 - mad(fit_uncon.resid;  center=mean(fit_uncon.resid))   / mad(T1_literature; center=mean(T1_literature))

1 - mean(abs.(fit_mono.resid   .- mean(fit_mono.resid)))   / mean(abs.(T1_literature .- mean(T1_literature)))

1 - mean(abs.(fit_Graham.resid .- mean(fit_Graham.resid))) / mean(abs.(T1_literature .- mean(T1_literature)))

1 - mean(abs.(fit_constr.resid .- mean(fit_constr.resid))) / mean(abs.(T1_literature .- mean(T1_literature)))

1 - mean(abs.(fit_uncon.resid  .- mean(fit_uncon.resid)))  / mean(abs.(T1_literature .- mean(T1_literature)))

1 - std(fit_mono.resid)   / std(T1_literature)

1 - std(fit_Graham.resid) / std(T1_literature)

1 - std(fit_constr.resid) / std(T1_literature)

1 - std(fit_uncon.resid)  / std(T1_literature)
