include("helper_functions.jl")
include("T1_mapping_methods.jl")
using FiniteDifferences
using DataFrames
using CSV
using MixedModels
using StatsPlots
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
ymax = maximum([maximum(abs.(df[!, j[i]] .* df[!, p[i]])) for i ∈ eachindex(j)]) * 1.1

for id ∈ eachindex(j)
    pall[id] = scatter(_x, abs.(df[!, j[id]] .* df[!, p[id]]),
        group=df.seq_type,
        hover=df.seq_name,
        xticks=id == length(j) ? (1:9, String.(unique(df.ROI))) : (1:9, ["" for _ ∈ 1:9]),
        ylim=(0, ymax),
        ylabel="$(j[id]) ⋅ $(p[id])",
        # title=id == 1 ? "all ROIs" : "",
        legend_position=:outerbottomright,
    )
end
plt = plot(pall..., layout=(6, 1))

## plot d(q)
d = :dT1dT1f
q = :m0s
scatter(df[!, q], df[!, d], group=df.seq_type, hover=df.seq_name, xlabel=string(q), ylabel=string(d))


## Linear Mixed model
for id ∈ eachindex(j)
    d = j[id]
    df.d = df[!, d]

    # frm = @formula(d ~ 1 + m0s + T1f + T2f + Tex + T1s + T2s + (1 | seq_type / seq_name) + (1 | ROI))
    frm = @formula(d ~ 1 + m0s + T1f + (1 | seq_type / seq_name) + (1 | ROI))
    model = fit(MixedModel, frm, df)
    # @info model

    fe = fixef(model)
    re = ranef(model)

    # pred = fe[1] .+ df.m0s * fe[2] .+ df.T1f * fe[3] .+ df.T2f * fe[4] .+ df.Tex * fe[5] .+ df.T1s * fe[6] .+ df.T2s * fe[7]
    pred = fill(fe[1], size(df, 1))
    for iparam ∈ 2:length(frm.rhs)
        if typeof(frm).parameters[2].parameters[iparam] == Term
            pred .+= fe[iparam] .* df[!, Symbol(frm.rhs[iparam])]
        end
    end

    r2_fixed = (cov(pred, df[!, :d]) / (std(pred) * std(df[!, :d])))^2

    pred = fe[1] .+ re[1][get.(Ref(Dict(x => i for (i, x) in pairs(unique(df.seq_name)))), df.seq_name, missing)]
    r2_seq_name = (cov(pred, df[!, :d]) / (std(pred) * std(df[!, :d])))^2

    pred = fe[1] .+ re[2][get.(Ref(Dict(x => i for (i, x) in pairs(unique(df.ROI)))), df.ROI, missing)]
    r2_ROI = (cov(pred, df[!, :d]) / (std(pred) * std(df[!, :d])))^2

    pred = fe[1] .+ re[3][get.(Ref(Dict(x => i for (i, x) in pairs(unique(df.seq_type)))), df.seq_type, missing)]
    r2_seq_type = (cov(pred, df[!, :d]) / (std(pred) * std(df[!, :d])))^2

    pred = predict(model)
    r2_full = (cov(pred, df[!, :d]) / (std(pred) * std(df[!, :d])))^2

    @printf("%s: r2_fixed = %.3f; r2_seq_type = %.3f; r2_seq_name = %.3f; r2_ROI = %.3f; r2_full = %.3f \n", d, r2_fixed, r2_seq_type, r2_seq_name, r2_ROI, r2_full)

    # Plot: observed vs full-model prediction
    dlim = maximum(df[!, :d]) - minimum(df[!, :d])
    xlim = (minimum(df[!, :d]) - 0.15dlim, maximum(df[!, :d]) + 0.15dlim)
    ylim = xlim

    scatter(df.d, pred, group=df.seq_type, hover=df.seq_name,
        xlabel="Observed d", ylabel="Predicted d",
        title=string(d),
        legend_position=:bottomright; xlim, ylim)

    plot!([xlim...], [xlim...], label="Ideal", lc=:white, ls=:dash)
    display(current())
end




##
# using MixedModelsExtras
# r2(model; conditional=false)
# r2(model; conditional=true)