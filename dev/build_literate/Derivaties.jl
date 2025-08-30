include("helper_functions.jl")
include("Derivatives_HelperFunctions.jl")

include("T1_mapping_methods.jl")

MT_model = gBloch()

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
j = [:dT1odm0s, :dT1odT1f, :dT1odT2f, :dT1odTex, :dT1odT1s, :dT1odT2s]

calculate_jacobian = false

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

            # The following line performs the actual computation of the derivatives.
            # `T1_functions` takes the rates of several MT parameters, so we have to call it with `1/T`.
            # `T₂ˢ` is rescaled to ensure stability of the finite difference algorithm.
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

_x = Float64.(map(x -> findfirst(isequal(x), unique(df.ROI)), df.ROI))
_x .+= (map(x -> findfirst(isequal(x), sort(unique(df.seq_type))), df.seq_type) .- 3) ./ 8
ymax = maximum([maximum(abs.(df[!, j[i]] .* mean(df[!, p[i]]))) for i ∈ eachindex(j)]) * 1.1

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
r2[!, Symbol("μ(∂T₁ᵒ/∂pᵢᴹᵀ) ⋅ μ(pᵢᴹᵀ)")]=Float64[]
r2[!, Symbol("σ(∂T₁ᵒ/∂pᵢᴹᵀ) / μ(∂T₁ᵒ/∂pᵢᴹᵀ)")]=Float64[]
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


pall = similar(j, Plots.Plot)
for id ∈ eachindex(j)
    μⱼ = abs(mean(df[!, j[id]]) * mean(df[!, p[id]]))
    cv = abs(std(df[!, j[id]]) / mean(df[!, j[id]]))

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


    # Plot: observed vs full-model prediction
    dlim = maximum(df[!, j[id]]) - minimum(df[!, j[id]])
    xlim = (minimum(df[!, j[id]]) - 0.15dlim, maximum(df[!, j[id]]) + 0.15dlim)

    available_markers = [:circle, :rect, :diamond, :hexagon, :cross, :xcross, :utriangle, :x, :dtriangle]
    unique_ROI = unique(df.ROI)
    label_to_marker = Dict(label => available_markers[i] for (i, label) in enumerate(unique_ROI))
    markers = [label_to_marker[label] for label in df.ROI]

    pall[id] = scatter(df[!, j[id]], predict(model);
        group=df.seq_type,
        hover=df.seq_name .* df.ROI,
        m=markers,
        title = j[id],
        xlabel=id ∈ [5,6] ? "Observed" : "",
        ylabel=id ∈ [1,3,5] ? "Predicted" : "",
        legend=:none,
        xlim,
        ylim=xlim
    )

    plot!(pall[id], [xlim...], [xlim...], label="Ideal", lc=:white, ls=:dash)
end

r2

r2_fixed

plt = plot(pall..., layout=(3, 2), size=(800, 1200))

fixed_model
