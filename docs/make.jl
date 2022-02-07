using Copulas
using Documenter

DocMeta.setdocmeta!(Copulas, :DocTestSetup, :(using Copulas); recursive=true)

makedocs(;
    modules=[Copulas],
    authors="Oskar Laverny",
    repo="https://github.com/lrnv/Copulas.jl/blob/{commit}{path}#{line}",
    sitename="Copulas.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lrnv.github.io/Copulas.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lrnv/Copulas.jl",
    devbranch="main",
)
