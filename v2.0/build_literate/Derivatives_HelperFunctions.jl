using FiniteDifferences
using DataFrames
using CategoricalArrays
using CSV
using MixedModels
using StatsPlots
using Printf
using Combinatorics

function shapley_regression(df, lhs, p)
    Δr² = similar(p, Float64)
    Δr² .= 0

    for ipick ∈ eachindex(p)
        p_pick = p[ipick]
        p_rest = filter(x -> x != p_pick, p)
        r_terms = term(1) + (term(1) | term(:seq_type)) + (term(1) | term(:seq_name)) + (term(1) | term(:ROI))

        for nterms = 0:length(p_rest)
            weight = 1 / length(combinations(p_rest, nterms)) / (length(p_rest) + 1)
            for combination ∈ combinations(p_rest, nterms)
                rhs = combination == Symbol[] ? r_terms : sum(t -> term(t), combination) + r_terms
                r²_rest = _r2(df, term(lhs), rhs)

                rhs += term(p_pick)
                r²_pick = _r2(df, term(lhs), rhs)

                Δr²[ipick] += weight * (r²_pick - r²_rest)
            end
        end
    end
    return Δr²
end

function _r2(df, lhs, rhs)
    model = fit(MixedModel, lhs ~ rhs, df)
    fixedeffects = Symbol.(rhs[collect(typeof.(rhs) .== Term)])

    σ²_fixed = isempty(fixedeffects) ? 0 : var(sum(param -> model.βs[param] .* df[!, param], fixedeffects); corrected=false)
    σ²_ROI = model.σs[:ROI][1]^2
    σ²_seqname = model.σs[:seq_name][1]^2
    σ²_seqtype = model.σs[:seq_type][1]^2
    σ²_ϵ = model.σ^2
    σ²_all = σ²_fixed + σ²_ROI + σ²_seqtype + σ²_seqname + σ²_ϵ
    r² = σ²_fixed / σ²_all
    return r²
end

function get_main_path()
    path_parts = splitpath(pwd())
    i_docs = findfirst(==("docs"), path_parts)
    main_path = i_docs === nothing ? pwd() : joinpath(path_parts[1:i_docs-1]...)
    return main_path
end
