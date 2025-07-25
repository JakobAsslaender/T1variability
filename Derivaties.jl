include("helper_functions.jl")
include("T1_mapping_methods.jl")
using FiniteDifferences
using DataFrames
using CSV
using MixedModels
using StatsPlots
using Printf
plotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native, size=(800, 1000))
MT_model = gBloch()
calculate_jacobian = false

## define parameters in ROIs
ROI = [:WM, :anteriorCC, :posteriorCC, :GM, :Caudate, :Putamen, :Pallidum, :Thalamus, :Hippocampus]
m0s = [0.212, 0.237, 0.235, 0.098, 0.113, 0.118, 0.164, 0.158, 0.097]
T1f = [1.84, 1.77, 1.80, 2.46, 1.95, 1.84, 1.664, 2.02, 2.65]
T2f = [76.9, 69.9, 76.3, 83, 73.3, 67.4, 59.3, 70.8, 91] .* 1e-3
Rex = [13.6, 13.4, 13.5, 14.0, 13.8, 14.9, 15.8, 14.2, 15.3]
T1s = [0.34, 0.349, 0.350, 0.42, 0.432, 0.385, 0.351, 0.396, 0.376]
T2s = [12.5, 14.5, 12.6, 14.4, 15.1, 15.4, 14.9, 13.0, 13.0]
Tex = 1 ./ Rex

ω0 = 0
B1 = 1

p = [:m0s, :T1f, :T2f, :Tex, :T1s, :T2s]
j = [:dT1dm0s, :dT1dT1f, :dT1dT2f, :dT1dTex, :dT1dT1s, :dT1dT2s]


##
if calculate_jacobian
    Nrows = length(ROI) * length(T1_functions)
    df = DataFrame(
        seq_id=Vector{Int}(undef, Nrows),
        seq_name=Vector{String}(undef, Nrows),
        seq_type=Vector{Symbol}(undef, Nrows),
        ROI=Vector{Symbol}(undef, Nrows),
        m0s=Vector{Float64}(undef, Nrows),
        T1f=Vector{Float64}(undef, Nrows),
        T2f=Vector{Float64}(undef, Nrows),
        Tex=Vector{Float64}(undef, Nrows),
        T1s=Vector{Float64}(undef, Nrows),
        T2s=Vector{Float64}(undef, Nrows),
        dT1dm0s=Vector{Float64}(undef, Nrows),
        dT1dT1f=Vector{Float64}(undef, Nrows),
        dT1dT2f=Vector{Float64}(undef, Nrows),
        dT1dTex=Vector{Float64}(undef, Nrows),
        dT1dT1s=Vector{Float64}(undef, Nrows),
        dT1dT2s=Vector{Float64}(undef, Nrows),
    )

    counter = 0
    Threads.@threads for iROI ∈ eachindex(ROI, m0s, T1f, T2f, Tex, T1s, T2s)
        _p = (m0s[iROI], T1f[iROI], T2f[iROI], Tex[iROI], T1s[iROI], T2s[iROI])
        Threads.@threads for iseq ∈ eachindex(T1_functions)
            idx = iseq + length(T1_functions) * (iROI - 1)

            j = jacobian(central_fdm(5, 1), p -> T1_functions[iseq](p[1], 1 / p[2], 1 / p[3], 1 / p[4], 1 / p[5], 1e-6p[6]), _p)[1]

            df[idx, :seq_id] = iseq
            df[idx, :seq_name] = seq_name[iseq]
            df[idx, :seq_type] = seq_type[iseq]
            df[idx, :ROI] = ROI[iROI]
            df[idx, :m0s] = m0s[iROI]
            df[idx, :T1f] = T1f[iROI]
            df[idx, :T2f] = T2f[iROI]
            df[idx, :Tex] = Tex[iROI]
            df[idx, :T1s] = T1s[iROI]
            df[idx, :T2s] = T2s[iROI]
            df[idx, :dT1dm0s] = j[1]
            df[idx, :dT1dT1f] = j[2]
            df[idx, :dT1dT2f] = j[3]
            df[idx, :dT1dTex] = j[4]
            df[idx, :dT1dT1s] = j[5]
            df[idx, :dT1dT2s] = j[6]

            global counter += 1
            @info "iROI = $iROI/$(length(ROI)), iseq = $iseq/$(length(T1_functions)), counter = $counter/$(size(df, 1))"
        end
    end
    CSV.write("jacobian.csv", df)
else
    df = DataFrame(CSV.File("jacobian.csv"))
end


## plot all data together
_x = Float64.(map(x -> findfirst(isequal(x), unique(df.ROI)), df.ROI))
_x .+= (map(x -> findfirst(isequal(x), sort(unique(df.seq_type))), df.seq_type) .- 3) ./ 8

pall = similar(j, Plots.Plot)
ymax = maximum([maximum(abs.(df[!, j[i]] .* mean(df[!, p[i]]))) for i ∈ eachindex(j)]) * 1.1

for id ∈ eachindex(j)
    pall[id] = scatter(_x, abs.(df[!, j[id]] .* mean(df[!, p[id]])),
        group=df.seq_type,
        hover=df.seq_name,
        xticks=id == length(j) ? (1:9, String.(unique(df.ROI))) : (1:9, ["" for _ ∈ 1:9]),
        ylim=(0, ymax),
        ylabel="|$(j[id]) ⋅ m($(p[id]))| (s)",
        legend_position=:outerbottomright,
    )
end
plt = plot(pall..., layout=(6, 1))


## plot d(q)
d = :dT1dT1f
q = :m0s
scatter(df[!, q], df[!, d], group=df.seq_type, hover=df.seq_name, xlabel=string(q), ylabel=string(d))


## Linear Mixed model
str_contributions = ""
str_contributions_tex = "\$\\partial T_1/\\partial p_i^\\text{MT}\$ & \$\\mu(\\frac{\\partial T_1}{\\partial p_i^\\text{MT}}) \\cdot \\mu(p_i^\\text{MT})\$ & \$\\sigma(\\frac{\\partial T_1}{\\partial p_i^\\text{MT}}) / \\mu(\\frac{\\partial T_1}{\\partial p_i^\\text{MT}})\$ & \$R^2_\\text{fixed}\$ & \$R^2_\\text{ROI}\$ & \$R^2_\\text{seq. type}\$ & \$R^2_\\text{seq. name}\$ & \$R^2_\\text{full}\$ & \$m_0^s = 0\$ \\\\"

str_fixed = ""
str_fixed_tex = ""
j_tex = ["\$\\partial T_1 / \\partial m_0^s\$", "\$\\partial T_1 / \\partial T_1^f\$", "\$\\partial T_1 / \\partial T_2^f\$", "\$\\partial T_1 / \\partial T_\\text{x}\$", "\$\\partial T_1 / \\partial T_1^s\$", "\$\\partial T_1 / \\partial T_2^s\$"]


for id ∈ eachindex(j)
    # mj = abs(mean(df[!, j[id]] .* df[!, p[id]])) #! normalized per data point
    mj = abs(mean(df[!, j[id]]) * mean(df[!, p[id]])) #! normalized separately
    cv = abs(std(df[!, j[id]]) / mean(df[!, j[id]]))

    # frm = @formula(derivative ~ 1 + m0s + T1f + T2f + Tex + T1s + T2s + (1 | seq_type / seq_name) + (1 | ROI))
    frm = @formula(derivative ~ 1 + m0s + T1f + T2f + Tex + T1s + T2s + (1 | seq_type) + (1 | seq_name) + (1 | ROI))
    model = fit(MixedModel, term(j[id]) ~ frm.rhs, df)
    # @info model

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

    str_contributions *= @sprintf("%s: mean_j = %.3f; CoV = %.3f; r²_fixed = %.3f; r²_ROI = %.3f; r²_seqtype = %.3f; r²_seqname = %.3f; r²_full = %.3f; \n", j[id], mj, cv, r²_fixed, r²_ROI, r²_seqtype, r²_seqname, r²_full)
    str_contributions_tex *= @sprintf("%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\ \n", j_tex[id], mj, cv, r²_fixed, r²_ROI, r²_seqtype, r²_seqname, r²_full)


    # r²_m0s = var(model.βs[:m0s] .* m0s, corrected=false) / σ²_all
    # r²_T1f = var(model.βs[:T1f] .* T1f, corrected=false) / σ²_all
    # r²_T2f = var(model.βs[:T2f] .* T2f, corrected=false) / σ²_all
    # r²_Tex = var(model.βs[:Tex] .* Tex, corrected=false) / σ²_all
    # r²_T1s = var(model.βs[:T1s] .* T1s, corrected=false) / σ²_all
    # r²_T2s = var(model.βs[:T2s] .* T2s, corrected=false) / σ²_all

    # str_fixed *= @sprintf("%s; %.3f; %.3f; %.3f; %.3f; %.3f; %.3f; %.3f \\\\ \n", j[id], r²_m0s, r²_T1f, r²_T2f, r²_Tex, r²_T1s, r²_T2s, m0s_intercept)
    # str_fixed_tex *= @sprintf("%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\ \n", j_tex[id], r²_m0s, r²_T1f, r²_T2f, r²_Tex, r²_T1s, r²_T2s, m0s_intercept)


    # Plot: observed vs full-model prediction
    dlim = maximum(df[!, :d]) - minimum(df[!, :d])
    xlim = (minimum(df[!, :d]) - 0.15dlim, maximum(df[!, :d]) + 0.15dlim)
    ylim = xlim

    scatter(df[!, j[id]], pred, group=df.seq_type, hover=df.seq_name,
        xlabel="Observed d", ylabel="Predicted d",
        title=string(j[id]),
        legend_position=:bottomright; xlim, ylim)

    plot!([xlim...], [xlim...], label="Ideal", lc=:white, ls=:dash)
    display(current())
end


println(str_contributions)
println(str_fixed)
# println(str_contributions_tex)
# println(str_fixed_tex)

## Export CSV for plotting in latex
_df = deepcopy(df)
_df.x = _x
_df.color = map(x -> findfirst(isequal(x), sort(unique(df.seq_type))), df.seq_type)

for id ∈ eachindex(j)
    _df[!, j[id]] = abs.(df[!, j[id]] .* mean(df[!, p[id]]))
end

CSV.write(expanduser("~/Documents/Paper/2025_T1sensitivity/Figures/derivatives/jacobian.csv"), _df)




##
# using MixedModelsExtras
# r2(model; conditional=false)
# r2(model; conditional=true)