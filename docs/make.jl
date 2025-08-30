@info "number of threads: " Threads.nthreads()

using Pkg
Pkg.activate(Base.source_path() * "/../..")
Pkg.instantiate()

using Documenter
using Literate
using Plots

# HTML Plotting Functionality
struct HTMLPlot
    p # :: Plots.Plot
end
const ROOT_DIR = joinpath(@__DIR__, "build")
const PLOT_DIR = joinpath(ROOT_DIR, "plots")

function get_plot_size(path)
    width = 100
    height = 700
    for line in eachline(path)
        # Look for the div line with style attributes
        m = match(r"""style="(width:\d+px;height:\d+px;)""", line)
        m = match(r"""style="width:(\d+)px;height:(\d+)px;""", line)
        if m !== nothing
            width, height = m.captures
            width = parse(Int, width)
            height = parse(Int, height)
            width += 20
            height += 20
        end
    end
    return "width:$(width)px;height:$(height)px;"
end

function Base.show(io::IO, ::MIME"text/html", p::HTMLPlot)
    mkpath(PLOT_DIR)
    path = joinpath(PLOT_DIR, string(UInt32(floor(rand() * 1e9)), ".html"))
    Plots.savefig(p.p, path)

    # Fix font issue in Safari (enable unicode characters)
    full_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
    $(read(path, String))
    </body>
    </html>
    """

    open(path, "w") do f
        write(f, full_html)
    end

    rel_path = get(ENV, "CI", "false") == "true" ? "../.." : ".."
    print(io, "<object type=\"text/html\" data=\"$(rel_path)/$(relpath(path, ROOT_DIR))\" style=\"$(get_plot_size(path))\"></object>")
end

# Notebook hack to display inline math correctly
function notebook_filter(str)
    re = r"(?<!`)``(?!`)"  # Two backquotes not preceded by nor followed by another
    return replace(str, re => "\$")
end

# Literate
OUTPUT = joinpath(@__DIR__, "src/build_literate")

files = [
    "T1_mapping_methods.jl",
    "Fit_qMT_to_literatureT1.jl",
    "helper_functions.jl",
    "Derivaties.jl",
    "Derivatives_HelperFunctions.jl",
]

for file in files
    file_path = joinpath(@__DIR__, "../", file)
    Literate.markdown(file_path, OUTPUT; credit=false)
    # Literate.notebook(file_path, OUTPUT; preprocess=notebook_filter, execute=false, credit=false)
    Literate.script(file_path, OUTPUT; credit=false)
end

makedocs(;
    clean=true,
    draft=false,
    doctest=false,
    authors="Jakob Asslaender <jakob.asslaender@nyumc.org> and contributors",
    sitename="T₁ variability",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JakobAsslaender.github.io/T1variability",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "T₁ variability" => Any[
            "build_literate/T1_mapping_methods.md",
            "build_literate/Fit_qMT_to_literatureT1.md",
            "build_literate/helper_functions.md",
        ],
        "T₁ sensitivity" => Any[
            "build_literate/Derivaties.md",
            "build_literate/Derivatives_HelperFunctions.md",
        ],
    ],
)

# Set dark theme as default independent of the OS's settings
run(`sed -i'.old' 's/var darkPreference = false/var darkPreference = true/g' docs/build/assets/themeswap.js`)

deploydocs(;
    repo="github.com/JakobAsslaender/T1variability",
    push_preview=true,
)
