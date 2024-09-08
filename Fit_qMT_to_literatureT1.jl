# # Global fit
# First, we load the required packages and include some [Helper functions](@ref)
include("helper_functions.jl")
nothing #hide #md

# and load all T₁ mapping methods:
include("T1_mapping_methods.jl")
nothing #hide #md

# ## Initialize plot and output vectors
#src #########################################################
p = plot([0, 2], [0, 2], xlabel="simulated T₁ (s)", ylabel="literature T₁ (s)", legend=:topleft, label=:none, xlim=(0.5, 1.1), ylim=(0.5, 1.1))
marker_list = [(seq_type_i == :IR) ? (:circle) : ((seq_type_i == :LL) ? (:cross) : ((seq_type_i == :vFA) ? (:diamond) : ((seq_type_i == :SR) ? (:dtriangle) : (:x)))) for seq_type_i in seq_type]

fit_name = String[] #hide
fitted_param = NTuple{6, Float64}[]
T1_simulated = Array{Float64}[]
ΔAIC_v = Float64[]
ΔBIC_v = Float64[]
nothing #hide #md

#src #########################################################
# ## Mono-exponential fit
#src #########################################################
# We simulate the mono-exponential model as an MT model with a vanishing semi-solid spin pool. In this case, the underlying MT model is irrelevant and we choose Graham's model for speed purposes:
MT_model = Graham()
push!(fit_name, "mono_exp") #src
nothing #hide #md

# The following parameters are hard-coded with the exception of `R1f`, which serves as an initialization for the global fit.
m0s = 0
R1f = 1 / 1.084  # 1/s
R1s = R1f        # 1/s
R2f = 1 / 0.0769 # 1/s
T2s = 0
Rx = 0           # 1/s
ω0 = 0
B1 = 1
nothing #hide #md

# We define a model for the global fit. It takes the global set of parameters `p` as an input and returns a vector of T₁ estimates that correspond to the T₁ mapping methods described by the vector `T1_functions`
function model(iseq, p)
    R1f = p[1]
    T1 = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1[i] = T1_functions[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1
end

# Perform the fit and save the global set of parameters:
fit_mono = curve_fit(model, 1:length(T1_literature), T1_literature, [R1f], x_tol=1e-3)
push!(fitted_param, (m0s, fit_mono.param[1], R2f, Rx, R1s, T2s))
push!(T1_simulated, model(1:length(T1_literature), fit_mono.param))
nothing #hide #md

# For this model, the global set of parameters is:
m0s
#-
T1f_fitted = 1/fit_mono.param[1] # s
#-
1/R1s # s
#-
1/R2f # s
#-
T2s # s
#-
Rx # 1/s
# where all but `R1f` are fixed. The following plot visualizes the quality of the fit an replicates Fig. 1 in the manuscript:
scatter!(p, T1_simulated[1], T1_literature, label="mono-exponential model", markershape=marker_list, hover=seq_name)
#md Main.HTMLPlot(p) #hide

# ### Akaike (AIC) and Bayesian (BIC) information criteria
# The information criteria depend on the number of measurements `n`, the number of parameters `k`, and the squared sum of the residuals `RSS`:
n = length(T1_literature)
#-
k = length(fit_mono.param)
#-
RSS = norm(fit_mono.resid)^2

# As the AIC and BIC are only informative in the difference, between two models, we use the mono-exponential model as the reference:
AIC_mono = n * log(RSS / n) + 2k
#-
BIC_mono = n * log(RSS / n) + k * log(n)
# In this case, `ΔAIC` is per definition zero:
ΔAIC = n * log(RSS / n) + 2k - AIC_mono
# as is `ΔBIC`:
ΔBIC = n * log(RSS / n) + k * log(n) - BIC_mono
# Store the results:
push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)
nothing #hide #md

#src #########################################################
# ## Unconstrained qMT fit with Graham's model
#src #########################################################
MT_model = Graham()
push!(fit_name, "unconstr_Graham") #src
nothing #hide #md

# The following parameters are hard-coded with the exception of `m0s`, `R1f`, and `R1s`, which serve as an initialization for the global fit.
m0s = 0.25
R1f = 1 / 1.84   # 1/s
R1s = 1 / 0.34   # 1/s
R2f = 1 / 0.0769 # 1/s
T2s = 12.5e-6    # s
Rx = 13.6        # 1/s
ω0 = 0
B1 = 1
nothing #hide #md

# We define a model for the global fit. It takes the global set of parameters `p` as an input and returns a vector of T₁ estimates that correspond to the T₁ mapping methods described by the vector `T1_functions`
function model(iseq, p)
    m0s, R1f, R1s = p
    T1 = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1[i] = T1_functions[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1
end

# Perform the fit and save the global set of parameters:
fit_Graham = curve_fit(model, 1:length(T1_literature), T1_literature, [m0s, R1f, R1s], x_tol=1e-3)
push!(fitted_param, (fit_Graham.param[1], fit_Graham.param[2], R2f, Rx, fit_Graham.param[3], T2s))
push!(T1_simulated, model(1:length(T1_literature), fit_Graham.param))
nothing #hide #md

# For this model, the global set of parameters is:
m0s_fitted = fit_Graham.param[1]
#-
T1f_fitted = 1/fit_Graham.param[2] # s
#-
T1s_fitted = 1/fit_Graham.param[3] # s
#-
1/R2f # s
#-
T2s # s
#-
Rx # 1/s
# where all but `m0s`, `R1f`, and `R1s` are fixed. The following plot visualizes the quality of the fit an replicates Fig. 1 in the manuscript:
scatter!(p, T1_simulated[2], T1_literature, label="Graham's model (unconstrained R₁ˢ)", markershape=marker_list, hover=seq_name)
#md Main.HTMLPlot(p) #hide


# ### Akaike (AIC) and Bayesian (BIC) information criteria
# The information criteria depend on the number of measurements `n`, the number of parameters `k`, and the squared sum of the residuals `RSS`:
n = length(T1_literature)
#-
k = length(fit_Graham.param)
#-
RSS = norm(fit_Graham.resid)^2

# With this information, we can calculate the AIC difference to the mono-exponential model:
ΔAIC = n * log(RSS / n) + 2k - AIC_mono
# and similarly for the BIC:
ΔBIC = n * log(RSS / n) + k * log(n) - BIC_mono
# Store the results:
push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)
nothing #hide #md


#src #########################################################
# ## Constrained qMT fit with the generalized Bloch model
#src #########################################################
MT_model = gBloch()
push!(fit_name, "constr_gBloch") #src
nothing #hide #md

# The following parameters are hard-coded with the exception of `m0s`, and `R1f`, which serve as an initialization for the global fit.
m0s = 0.139
R1f = 1 / 1.084  # 1/s
R1s = 1          # 1/s
R2f = 1 / 0.0769 # 1/s
T2s = 12.5e-6    # s
Rx = 23          # 1/s
ω0 = 0
B1 = 1
nothing #hide #md

# We define a model for the global fit. It takes the global set of parameters `p` as an input and returns a vector of T₁ estimates that correspond to the T₁ mapping methods described by the vector `T1_functions`
function model(iseq, p)
    m0s, R1f = p
    R1s = R1f
    T1 = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1[i] = T1_functions[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1
end

# Perform the fit and save the global set of parameters:
fit_constr = curve_fit(model, 1:length(T1_literature), T1_literature, [m0s, R1f], x_tol=1e-3)
push!(fitted_param, (fit_constr.param[1], fit_constr.param[2], R2f, Rx, fit_constr.param[2], T2s))
push!(T1_simulated, model(1:length(T1_literature), fit_constr.param))
nothing #hide #md

# For this model, the global set of parameters is:
m0s_fitted = fit_constr.param[1]
#-
T1f_fitted = 1/fit_constr.param[2] # s
#-
T1s_fitted = 1/fit_constr.param[2] # s
#-
1/R2f # s
#-
T2s # s
#-
Rx # 1/s
# where all but `m0s`, `R1f`, and are fixed (`R1s = R1f`). The following plot visualizes the quality of the fit an replicates Fig. 1 in the manuscript:
scatter!(p, T1_simulated[3], T1_literature, label="generalized Bloch model (R₁ˢ = R₁ᶠ constraint)", markershape=marker_list, hover=seq_name)
#md Main.HTMLPlot(p) #hide


# ### Akaike (AIC) and Bayesian (BIC) information criteria
# The information criteria depend on the number of measurements `n`, the number of parameters `k`, and the squared sum of the residuals `RSS`:
n = length(T1_literature)
#-
k = length(fit_constr.param)
#-
RSS = norm(fit_constr.resid)^2

# With this information, we can calculate the AIC difference to the mono-exponential model:
ΔAIC = n * log(RSS / n) + 2k - AIC_mono
# and similarly for the BIC:
ΔBIC = n * log(RSS / n) + k * log(n) - BIC_mono
# Store the results:
push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)
nothing #hide #md


#src #########################################################
# Unconstrained qMT fit with the generalized Bloch model
#src #########################################################
MT_model = gBloch()
push!(fit_name, "unconstr_gBloch") #src
nothing #hide #md

# The following parameters are hard-coded with the exception of `m0s`, `R1f`, and `R1s`, which serve as an initialization for the global fit.
m0s = 0.25
R1f = 1 / 1.84   # 1/s
R1s = 1 / 0.34   # 1/s
R2f = 1 / 0.0769 # 1/s
T2s = 12.5e-6    # s
Rx = 13.6        # 1/s
ω0 = 0
B1 = 1
nothing #hide #md

# We define a model for the global fit. It takes the global set of parameters `p` as an input and returns a vector of T₁ estimates that correspond to the T₁ mapping methods described by the vector `T1_functions`
function model(iseq, p)
    m0s, R1f, R1s = p
    T1 = similar(iseq, Float64)
    Threads.@threads for i in eachindex(iseq)
        T1[i] = T1_functions[iseq[i]](m0s, R1f, R2f, Rx, R1s, T2s)
    end
    return T1
end

# Perform the fit and save the global set of parameters:
fit_uncon = curve_fit(model, 1:length(T1_literature), T1_literature, [m0s, R1f, R1s], x_tol=1e-3, show_trace=true)
push!(fitted_param, (fit_uncon.param[1], fit_uncon.param[2], R2f, Rx, fit_uncon.param[3], T2s))
push!(T1_simulated, model(1:length(T1_literature), fit_uncon.param))
nothing #hide #md

# For this model, the global set of parameters is:
m0s_fitted = fit_uncon.param[1]
#-
T1f_fitted = 1/fit_uncon.param[2] # s
#-
T1s_fitted = 1/fit_uncon.param[3] # s
#-
1/R2f # s
#-
T2s # s
#-
Rx # 1/s
# where all but `m0s`, `R1f`, and `R1s` are fixed. The following plot visualizes the quality of the fit an replicates Fig. 1 in the manuscript:
scatter!(p, T1_simulated[4], T1_literature, label="generalized Bloch model (unconstrained R₁ˢ)", markershape=marker_list, hover=seq_name)
#md Main.HTMLPlot(p) #hide


# ## Akaike (AIC) and Bayesian (BIC) information criteria
# The information criteria depend on the number of measurements `n`, the number of parameters `k`, and the squared sum of the residuals `RSS`:
n = length(T1_literature)
#-
k = length(fit_uncon.param)
#-
RSS = norm(fit_uncon.resid)^2

# With this information, we can calculate the AIC difference to the mono-exponential model:
ΔAIC = n * log(RSS / n) + 2k - AIC_mono
# and similarly for the BIC:
ΔBIC = n * log(RSS / n) + k * log(n) - BIC_mono
# Store the results:
push!(ΔAIC_v, ΔAIC)
push!(ΔBIC_v, ΔBIC)
nothing #hide #md




#src #########################################################
# ## Data analysis
#src #########################################################
# The span of T₁ estimates in the literature is
extrema(T1_literature)

# and the coefficient of variation is
variation(T1_literature)

# ### Median absolute deviation
# In the paper, the median absolute deviation wrt. the mean value is used, as it is more robust to outliers compared to the mean absolute deviation or the standard deviation. Note, however, that the median absolute deviation of the mono-exponential fit is dominated by an outlier, artificially inflating the corresponding reduction.

# A mono-exponential model explains the following fraction of the T₁ variability in the literature:
1 - mad(fit_mono.resid;   center=mean(fit_mono.resid))    / mad(T1_literature; center=mean(T1_literature))
# Graham's spectral model without constraints on `R₁ᶠ` explains the following fraction of the T₁ variability in the literature:
1 - mad(fit_Graham.resid; center=mean(fit_Graham.resid))  / mad(T1_literature; center=mean(T1_literature))
# The generalized Bloch model constrained by `R₁ˢ = R₁ᶠ` explains the following fraction of the T₁ variability in the literature:
1 - mad(fit_constr.resid; center=mean(fit_constr.resid))  / mad(T1_literature; center=mean(T1_literature))
# The generalized Bloch model without constraints on `R₁ᶠ` explains the following fraction of the T₁ variability in the literature:
1 - mad(fit_uncon.resid;  center=mean(fit_uncon.resid))   / mad(T1_literature; center=mean(T1_literature))

# ### Mean absolute deviation
# For comparison, the mean absolute deviation is analyzed:
# A mono-exponential model explains the following fraction of the T₁ variability in the literature:
1 - mean(abs.(fit_mono.resid   .- mean(fit_mono.resid)))   / mean(abs.(T1_literature .- mean(T1_literature)))
# Graham's spectral model without constraints on `R₁ᶠ` explains the following fraction of the T₁ variability in the literature:
1 - mean(abs.(fit_Graham.resid .- mean(fit_Graham.resid))) / mean(abs.(T1_literature .- mean(T1_literature)))
# The generalized Bloch model constrained by `R₁ˢ = R₁ᶠ` explains the following fraction of the T₁ variability in the literature:
1 - mean(abs.(fit_constr.resid .- mean(fit_constr.resid))) / mean(abs.(T1_literature .- mean(T1_literature)))
# The generalized Bloch model without constraints on `R₁ᶠ` explains the following fraction of the T₁ variability in the literature:
1 - mean(abs.(fit_uncon.resid  .- mean(fit_uncon.resid)))  / mean(abs.(T1_literature .- mean(T1_literature)))

# ### Standard deviation
# For comparison, the standard deviation is analyzed:
# A mono-exponential model explains the following fraction of the T₁ variability in the literature:
1 - std(fit_mono.resid)   / std(T1_literature)
# Graham's spectral model without constraints on `R₁ᶠ` explains the following fraction of the T₁ variability in the literature:
1 - std(fit_Graham.resid) / std(T1_literature)
# The generalized Bloch model constrained by `R₁ˢ = R₁ᶠ` explains the following fraction of the T₁ variability in the literature:
1 - std(fit_constr.resid) / std(T1_literature)
# The generalized Bloch model without constraints on `R₁ᶠ` explains the following fraction of the T₁ variability in the literature:
1 - std(fit_uncon.resid)  / std(T1_literature)


#src #########################################################
#src export data
#src #########################################################
using Printf #src

#src T1 estimates
io = open(expanduser("~/Documents/Paper/2023_T1variablity/Figures/T1.txt"), "w") #src
write(io, "meas ") #src
[write(io, "sim_$name_i ") for name_i in fit_name] #src
write(io, "seq_marker ") #src
write(io, "\n") #src

for i_seq in eachindex(T1_literature) #src
    write(io, @sprintf("%1.3f ", T1_literature[i_seq])) #src
    [write(io, @sprintf("%1.3f ", T1_simulated[i_fit][i_seq])) for i_fit in eachindex(T1_simulated)] #src
    write(io, @sprintf("%s ", string(seq_type[i_seq])[1])) #src
    write(io, "\n") #src
end #src
close(io) #src

#src residuals (no longer used)
io = open(expanduser("~/Documents/Paper/2023_T1variablity/Figures/residuals.txt"), "w") #src
for i_seq in eachindex(fit_mono.resid) #src
    write(io, "1 ") #src
    write(io, @sprintf("%1.3f ", fit_mono.resid[i_seq])) #src
    write(io, @sprintf("%1.3f ", fit_constr.resid[i_seq])) #src
    write(io, @sprintf("%1.3f ", fit_uncon.resid[i_seq])) #src
    write(io, @sprintf("%1.3f ", fit_Graham.resid[i_seq])) #src
    write(io, "\n") #src
end #src
close(io) #src

#src AIC/BIC table
println("") #src
println("model & \$T_1^s\$ constraint & \$\\Delta\$AIC & \$\\Delta\$BIC \\\\") #src
println("\\midrule") #src
println("mono-exponential   & none            & 0           & 0           \\\\") #src
i = findfirst(fit_name .== "unconstr_Graham") #src
println(@sprintf("Graham's & none & %1.1f & %1.1f \\\\", ΔAIC[i], ΔBIC[i])) #src
i = findfirst(fit_name .== "constr_gBloch") #src
println(@sprintf("generalized Bloch & \$T_1^s = T_1^f\$ & %1.1f & %1.1f \\\\", ΔAIC[i], ΔBIC[i])) #src
i = findfirst(fit_name .== "unconstr_gBloch") #src
println(@sprintf("generalized Bloch & none & %1.1f & %1.1f \\\\", ΔAIC[i], ΔBIC[i])) #src
println("\\bottomrule") #src


#src fitted MT parameter
println("") #src
println("MT model                & \\multicolumn{2}{c|}{Graham's} & \\multicolumn{4}{c}{generalized Bloch} \\\\") #src
println("\\midrule") #src
println("\$T_1^s\$ constraint     & \\multicolumn{2}{c|}{none}     & \\multicolumn{2}{c|}{\$T_1^s = T_1^f\$}  & \\multicolumn{2}{c}{none} \\\\") #src
println("\\midrule") #src
println("study                   & this                          & \\cite{Gelderen2016,Stanisz.2005}      & this                     & \\cite{Assländer.2024,Stanisz.2005} & this   & \\cite{Assländer.2024oxt} \\\\") #src
println("\\midrule") #src
println(@sprintf("\$m_0^s\$ & %1.2f & 0.27 & %1.2f & 0.14 & %1.2f & 0.21 \\\\", fitted_param[2][1], fitted_param[3][1], fitted_param[4][1])) #src
println(@sprintf("\$T_1^f\$ (s) & %1.2f & 2.44 & %1.2f & 1.52 & %1.2f & 1.84 \\\\", 1/fitted_param[2][2], 1/fitted_param[3][2], 1/fitted_param[4][2])) #src
println(@sprintf("\$T_1^s\$ (s) & %1.2f & 0.25 & \\g\$T_1^f\$ & & %1.2f & 0.34 \\\\", 1/fitted_param[2][5], 1/fitted_param[4][5])) #src
println(@sprintf("\$T_2^f\$ (ms) & \\g%1.1f & 69 & \\g%1.1f & 70.1 & \\g%1.1f & 76.9 \\\\", 1e3/fitted_param[2][3], 1e3/fitted_param[3][3], 1e3/fitted_param[4][3])) #src
println(@sprintf("\$T_2^s\$ (\$\\upmu\$s) & \\g%1.1f & 10.0 & \\g%1.1f & & \\g%1.1f & 12.5 \\\\", 1e6*fitted_param[2][6], 1e6*fitted_param[3][6], 1e6*fitted_param[4][6])) #src
println(@sprintf("\$R_\\text{x}\$ (s\$^{-1}\$) & \\g%1.1f & 9.0 & \\g%1.1f & 23.0 & \\g%1.1f & 13.6 \\\\", fitted_param[2][4], fitted_param[3][4], fitted_param[4][4])) #src
println("\\bottomrule") #src