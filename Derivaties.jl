# # Sensitivity analysis
# ## Load code
# First, we load the required packages and include some [Helper functions](@ref):
include("helper_functions.jl")
include("Derivatives_HelperFunctions.jl")
nothing #hide #md

# load all [T₁-mapping methods](@ref):
include("T1_mapping_methods.jl")
nothing #hide #md

# All simulations in this analysis are performed with the generalized Bloch model:
MT_model = gBloch()
nothing #hide #md


# ## Set parameters
# We analyze overall 9 ROIs and use the MT parameters from the paper [Unconstrained quantitative magnetization transfer imaging: Disentangling T₁ of the free and semi-solid spin pools](https://doi.org/10.1162/imag_a_00177):
ROI = [:WM, :anteriorCC, :posteriorCC, :GM, :caudate, :putamen, :pallidum, :thalamus, :hippocampus]
m0s = [0.212, 0.237, 0.235, 0.098, 0.113, 0.118, 0.164, 0.158, 0.097]
T1f = [1.84, 1.77, 1.80, 2.46, 1.95, 1.84, 1.664, 2.02, 2.65]
T2f = [76.9, 69.9, 76.3, 83, 73.3, 67.4, 59.3, 70.8, 91] .* 1e-3
Rex = [13.6, 13.4, 13.5, 14.0, 13.8, 14.9, 15.8, 14.2, 15.3]
T1s = [0.34, 0.349, 0.350, 0.42, 0.432, 0.385, 0.351, 0.396, 0.376]
T2s = [12.5, 14.5, 12.6, 14.4, 15.1, 15.4, 14.9, 13.0, 13.0]
nothing #hide #md

# For consistency, we use times instead of rates throughout
Tex = 1 ./ Rex
nothing #hide #md

# and we neglect B₀ and B₁ inhomogeneities:
ω0 = 0
B1 = 1
nothing #hide #md

# We define vectors of symbols for all MT parameters and the derivatives ∂T₁ᵒ/∂pⱼᴹᵀ:
p = [:m0s, :T1f, :T2f, :Tex, :T1s, :T2s]
j = [:dT1odm0s, :dT1odT1f, :dT1odT2f, :dT1odTex, :dT1odT1s, :dT1odT2s]
nothing #hide #md

# ## Compute derivatives
# Since the calculation of the derivatives is computationally intensive, it is pre-computed and saved in a CSV file. To recompute the derivatives, set the following parameter to `true`:
calculate_jacobian = false
nothing #hide #md

# We store all derivatives in a `DataFrame` list (saved in the file `jacobian.csv`), concatenating ROIs and T₁ mapping methods:
Nrows = length(ROI) * length(T1_functions)

if calculate_jacobian
    df = DataFrame(
        seq_name=Vector{String}(undef, Nrows),
        seq_type=Vector{Symbol}(undef, Nrows),
        ROI=Vector{Symbol}(undef, Nrows),
        m0s=Vector{Float64}(undef, Nrows),
        T1f=Vector{Float64}(undef, Nrows),
        T2f=Vector{Float64}(undef, Nrows),
        Tex=Vector{Float64}(undef, Nrows),
        T1s=Vector{Float64}(undef, Nrows),
        T2s=Vector{Float64}(undef, Nrows),
        dT1odm0s=Vector{Float64}(undef, Nrows),
        dT1odT1f=Vector{Float64}(undef, Nrows),
        dT1odT2f=Vector{Float64}(undef, Nrows),
        dT1odTex=Vector{Float64}(undef, Nrows),
        dT1odT1s=Vector{Float64}(undef, Nrows),
        dT1odT2s=Vector{Float64}(undef, Nrows),
    )

    Threads.@threads for iROI ∈ eachindex(ROI)
        _p = (m0s[iROI], T1f[iROI], T2f[iROI], Tex[iROI], T1s[iROI], T2s[iROI])
        Threads.@threads for iseq ∈ eachindex(T1_functions)
            df_idx = iseq + length(T1_functions) * (iROI - 1)

            ## The following line performs the actual computation of the derivatives.
            ## `T1_functions` takes the rates of several MT parameters, so we have to call it with `1/T`.
            ## `T₂ˢ` is rescaled to ensure stability of the finite difference algorithm.
            j = jacobian(central_fdm(5, 1), p -> T1_functions[iseq](p[1], 1 / p[2], 1 / p[3], 1 / p[4], 1 / p[5], 1e-6p[6]), _p)[1]

            df[df_idx, :seq_name] = seq_name[iseq]
            df[df_idx, :seq_type] = seq_type[iseq]
            df[df_idx, :ROI] = ROI[iROI]
            df[df_idx, :m0s] = m0s[iROI]
            df[df_idx, :T1f] = T1f[iROI]
            df[df_idx, :T2f] = T2f[iROI]
            df[df_idx, :Tex] = Tex[iROI]
            df[df_idx, :T1s] = T1s[iROI]
            df[df_idx, :T2s] = T2s[iROI]
            df[df_idx, :dT1odm0s] = j[1]
            df[df_idx, :dT1odT1f] = j[2]
            df[df_idx, :dT1odT2f] = j[3]
            df[df_idx, :dT1odTex] = j[4]
            df[df_idx, :dT1odT1s] = j[5]
            df[df_idx, :dT1odT2s] = j[6]
        end
    end
    CSV.write("$(get_main_path())/jacobian.csv", df)
else
    df = DataFrame(CSV.File("$(get_main_path())/jacobian.csv"))
end
show(df; allrows=false, allcols=true)

# ## Figure 1
# Fig. 1 is a scatter plot and `_x` ensures the desired alignment along the x-axis and `ymax` ensures a uniform scaling of the plot:
_x = Float64.(map(x -> findfirst(isequal(x), unique(df.ROI)), df.ROI))
_x .+= (map(x -> findfirst(isequal(x), sort(unique(df.seq_type))), df.seq_type) .- 3) ./ 8
ymax = maximum([maximum(abs.(df[!, j[i]] .* mean(df[!, p[i]]))) for i ∈ eachindex(j)]) * 1.1
nothing #hide #md
#-

pall = similar(j, Plots.Plot)
ylabels = ["|∂T₁ᵒ/∂m₀ˢ⋅μ(m₀ˢ)| (s)", "|∂T₁ᵒ/∂T₁ᶠ⋅μ(T₁ᶠ)| (s)", "|∂T₁ᵒ/∂T₂ᶠ⋅μ(T₂ᶠ)| (s)", "|∂T₁ᵒ/∂Tₓ⋅μ(Tₓ)| (s)", "|∂T₁ᵒ/∂T₁ˢ⋅μ(T₁ˢ)| (s)", "|∂T₁ᵒ/∂T₂ˢ⋅μ(T₂ˢ)| (s)"]
for id ∈ eachindex(j)
    pall[id] = scatter(_x, abs.(df[!, j[id]] .* mean(df[!, p[id]])),
        group=df.seq_type,
        hover=df.seq_name,
        xticks=id == length(j) ? (1:9, String.(unique(df.ROI))) : (1:9, ["" for _ ∈ 1:9]),
        ylim=(0, ymax),
        ylabel=ylabels[id],
        legend_position=:none,
    )
end
plt = plot(pall..., layout=(6, 1), size=(800, 1500))
#md Main.HTMLPlot(plt) #hide

# Derivatives of the observed `T₁ᵒ` with respect to the 6 MT parameters. I calculated the derivatives for 9 brain regions of interest (ROIs) and for 25 pulse sequences, grouped into different sequence types (cf. legend). Here, WM denotes white matter, CC the corpus callosum, and GM gray matter. The derivatives `∂T₁ᵒ/∂pᵢᴹᵀ` are normalized by `pᵢᴹᵀ`, averaged over all ROIs, to allow for a comparison between the parameters.


# ## Linear Mixed model
# First, we initialize a few empty `DataFrame` objects that are filled in the for-loop.
df_pred = DataFrame(
    dT1odm0s_observe=Vector{Float64}(undef, Nrows),
    dT1odT1f_observe=Vector{Float64}(undef, Nrows),
    dT1odT2f_observe=Vector{Float64}(undef, Nrows),
    dT1odTex_observe=Vector{Float64}(undef, Nrows),
    dT1odT1s_observe=Vector{Float64}(undef, Nrows),
    dT1odT2s_observe=Vector{Float64}(undef, Nrows),
    dT1odm0s_predict=Vector{Float64}(undef, Nrows),
    dT1odT1f_predict=Vector{Float64}(undef, Nrows),
    dT1odT2f_predict=Vector{Float64}(undef, Nrows),
    dT1odTex_predict=Vector{Float64}(undef, Nrows),
    dT1odT1s_predict=Vector{Float64}(undef, Nrows),
    dT1odT2s_predict=Vector{Float64}(undef, Nrows),
    color=Vector{Int}(undef, Nrows),
    ROI=Vector{Symbol}(undef, Nrows),
)

r2 = DataFrame()
r2[!, Symbol("∂T₁ᵒ / ∂pᵢᴹᵀ")]=Symbol[]
r2[!, Symbol("μ(|∂T₁ᵒ/∂pᵢᴹᵀ|) ⋅ μ(pᵢᴹᵀ)")]=Float64[]
r2[!, Symbol("σ(∂T₁ᵒ/∂pᵢᴹᵀ) / μ(|∂T₁ᵒ/∂pᵢᴹᵀ|)")]=Float64[]
r2[!, Symbol("R²(fixed)")]=Float64[]
r2[!, Symbol("R²(ROI)")]=Float64[]
r2[!, Symbol("R²(seq. type)")]=Float64[]
r2[!, Symbol("R²(ind. seq)")]=Float64[]
r2[!, Symbol("R²(full)")]=Float64[]

r2_fixed = DataFrame()
r2_fixed[!, Symbol("∂T₁ᵒ / ∂pᵢᴹᵀ")]=Symbol[]
r2_fixed[!, Symbol("R²(m₀ˢ)")]=Float64[]
r2_fixed[!, Symbol("R²(T₁ᶠ)")]=Float64[]
r2_fixed[!, Symbol("R²(T₂ᶠ)")]=Float64[]
r2_fixed[!, Symbol("R²(Tₓ)")]=Float64[]
r2_fixed[!, Symbol("R²(T₁ˢ)")]=Float64[]
r2_fixed[!, Symbol("R²(T₂ˢ)")]=Float64[]
r2_fixed[!, Symbol("R²(fixed)")]=Float64[]

fixed_model = DataFrame()
fixed_model[!, Symbol("∂T₁ᵒ / ∂pᵢᴹᵀ")]=Symbol[]
fixed_model[!, Symbol("a₀")]=Float64[]
fixed_model[!, Symbol("a(m₀ˢ)")]=Float64[]
fixed_model[!, Symbol("a(T₁ᶠ) (1/s)")]=Float64[]
fixed_model[!, Symbol("a(T₂ᶠ) (1/s)")]=Float64[]
fixed_model[!, Symbol("a(Tₓ) (1/s)")]=Float64[]
fixed_model[!, Symbol("a(T₁ˢ) (1/s)")]=Float64[]
fixed_model[!, Symbol("a(T₂ˢ) (1/s)")]=Float64[]
fixed_model[!, Symbol("m₀ˢ = 0")]=Float64[]

str_r2_tex = "" #src
str_fixed_r2_tex = "" #src
str_fixed_model_tex = "" #src
j_tex = ["\$\\partial T_1 / \\partial m_0^\\text{s}\$", "\$\\partial T_1 / \\partial T_1^\\text{f}\$", "\$\\partial T_1 / \\partial T_2^\\text{f}\$", "\$\\partial T_1 / \\partial T_\\text{x}\$", "\$\\partial T_1 / \\partial T_1^\\text{s}\$", "\$\\partial T_1 / \\partial T_2^\\text{s}\$"] #src

pall = similar(j, Plots.Plot)
for id ∈ eachindex(j)
    μⱼ = mean(abs.(df[!, j[id]])) * mean(df[!, p[id]])
    cv = std(df[!, j[id]]) / mean(abs.(df[!, j[id]]))

    frm = @formula(derivative ~ 1 + m0s + T1f + T2f + Tex + T1s + T2s + (1 | seq_type) + (1 | seq_name) + (1 | ROI))
    model = fit(MixedModel, term(j[id]) ~ frm.rhs, df)

    pred = sum(param -> model.βs[param] .* df[!, param], Symbol.(frm.rhs[collect(typeof.(frm.rhs) .== Term)]))
    σ²_fixed = var(pred; corrected=false)

    σ²_ROI = model.σs[:ROI][1]^2
    σ²_seqname = model.σs[:seq_name][1]^2
    σ²_seqtype = model.σs[:seq_type][1]^2
    σ²_ϵ = model.σ^2
    σ²_all = σ²_fixed + σ²_ROI + σ²_seqtype + σ²_seqname + σ²_ϵ

    r²_fixed = σ²_fixed / σ²_all
    r²_ROI = σ²_ROI / σ²_all
    r²_seqname = σ²_seqname / σ²_all
    r²_seqtype = σ²_seqtype / σ²_all
    r²_full = (σ²_fixed + σ²_ROI + σ²_seqtype + σ²_seqname) / σ²_all

    fe = fixef(model)
    m0s_intercept = fe[1] + mean(df.T1f) * fe[3] + mean(df.T2f) * fe[4] + mean(df.Tex) * fe[5] + mean(df.T1s) * fe[6] + mean(df.T2s) * fe[7]

    push!(r2, (j[id], μⱼ, cv, r²_fixed, r²_ROI, r²_seqtype, r²_seqname, r²_full))

    Δr² = shapley_regression(df, j[id], p)
    push!(r2_fixed, (j[id], Δr²..., sum(Δr²)))

    push!(fixed_model, (j[id], fe..., m0s_intercept))

    global str_r2_tex *= @sprintf("%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \n", j_tex[id], μⱼ, cv, r²_fixed, r²_ROI, r²_seqtype, r²_seqname, r²_full) #src
    global str_fixed_r2_tex *= @sprintf("%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \n", j_tex[id], Δr²...) #src
    global str_fixed_model_tex *= @sprintf("%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \n", j_tex[id], fe..., m0s_intercept) #src

    ## Plot: observed vs full-model prediction
    dlim = maximum(df[!, j[id]]) - minimum(df[!, j[id]])
    xlim = (minimum(df[!, j[id]]) - 0.15dlim, maximum(df[!, j[id]]) + 0.15dlim)

    available_markers = [:circle, :rect, :diamond, :hexagon, :cross, :xcross, :utriangle, :x, :dtriangle]
    unique_ROI = unique(df.ROI)
    label_to_marker = Dict(label => available_markers[i] for (i, label) in enumerate(unique_ROI))
    markers = [label_to_marker[label] for label in df.ROI]

    pall[id] = scatter(df[!, j[id]], predict(model);
        group=df.seq_type,
        hover=df.seq_name .* "; " .* df.ROI,
        m=markers,
        title = j[id],
        xlabel=id ∈ [5,6] ? "Observed" : "",
        ylabel=id ∈ [1,3,5] ? "Predicted" : "",
        legend_position=:outerbottomright,
        xlim,
        ylim=xlim
    )

    plot!(pall[id], [xlim...], [xlim...], label="Ideal", lc=:white, ls=:dash)
    ## write in df #src
    df_pred[!, String(j[id]) * "_observe"] = df[!, j[id]] #src
    df_pred[!, String(j[id]) * "_predict"] = predict(model) #src
    df_pred[!, :ROI] = df[!, :ROI] #src
    df_pred[!, :color] = levelcode.(CategoricalArray(df[!, :seq_type])) #src
end

# ## Table 1
# Print the results of the mixed effects model fit:
r2
# This table corresponds to Tab. 1 in the [paper](https://arxiv.org/pdf/TODO). The column `μ(|∂T₁ᵒ/∂pᵢᴹᵀ|) ⋅ μ(pᵢᴹᵀ)` denotes the mean derivative, normalized by the average parameter, and serves as a measure for the sensitivity of `T₁ᵒ` to the respective parameter. The column `σ(∂T₁ᵒ/∂pᵢᴹᵀ) / μ(|∂T₁ᵒ/∂pᵢᴹᵀ|)` denotes the coefficient of variation. The coefficients of determination for the full model `R^2(full)` is dissected into its components: `R^2(full) = R²(fixed) + R^2(ROI) + R^2(seq. type) + R^2(ind. seq)`, where `R²(fixed)` captures all fixed effects, that is, the degree to which variations of the `pᵢᴹᵀ` between the ROIs explain the derivatives' variability. `R^2(ROI)` captures the ROI-identifier as a random variable, potentially modeling inter-ROI variations not captured by the linear model of `pᵢᴹᵀ`. `R^2(seq. type)` captures the degree to which the sequence type, that is, the groups inversion-recovery, Look-Locker, saturation-recovery, variable flip angle, and MP²RAGE, explains variability of the derivatives, and `R^2(ind. seq)` captures each sequence by itself.

# ## Table 2
r2_fixed
# This table corresponds to Tab. 2 in the [paper](https://arxiv.org/pdf/TODO) and analyzes the fixed effects. `R²(fixed)` is separated into the individual effects with Shapley regression. Note that `R²(fixed) = R²(m₀ˢ) + R²(T₁ᶠ) + R²(T₂ᶠ) + R²(Tₓ) + R²(T₁ˢ) + R²(T₂ˢ).

# ## Figure 2
plt = plot(pall..., layout=(3, 2), size=(800, 1000))
#md Main.HTMLPlot(plt) #hide

# Validation of the mixed effects model, where "observed" denotes the simulated derivatives, and "predicted" the output of the mixed model. The sequence type is here color-coded, while the maker shape identifies the region of interest. Here, WM denotes white matter, CC the corpus callosum, and GM gray matter. The dotted line represents the perfect fit.

# ## Table A1
fixed_model
# Coefficients of the fixed effects for each derivative's mixed model fit, including the intercept `a₀`. The last column denotes the derivatives assuming `m₀ˢ = 0` and the mean values for the other parameters. For `m₀ˢ = 0`, the MT model reduces to the Bloch model, and a perfect statistical model should result in `∂T₁ᵒ/∂T₁ᶠ = 1` and `∂T₁ᵒ/∂Tₓ = ∂T₁ᵒ/∂T₁ˢ = ∂T₁ᵒ/∂T₂ˢ = 0`. This is clearly not the case, highlighting the limitations of the mixed-effects model when extrapolating.


## #src
println(r2) #src
println(r2_fixed) #src
println(fixed_model) #src
println(str_r2_tex) #src
println(str_fixed_r2_tex) #src
println(str_fixed_model_tex) #src

## Export CSV for plotting in latex #src
_df = deepcopy(df) #src
_df.x = _x #src
_df.color = map(x -> findfirst(isequal(x), sort(unique(df.seq_type))), df.seq_type) #src

for id ∈ eachindex(j) #src
    _df[!, j[id]] = abs.(df[!, j[id]] .* mean(df[!, p[id]])) #src
end #src

CSV.write(expanduser("~/Documents/Paper/2025_T1sensitivity/Figures/derivatives/jacobian.csv"), _df) #src
CSV.write(expanduser("~/Documents/Paper/2025_T1sensitivity/Figures/mixed_model/mixed_model.csv"), df_pred) #src