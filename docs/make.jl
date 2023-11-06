using Copulas
using Documenter

DocMeta.setdocmeta!(Copulas, :DocTestSetup, :(using Copulas); recursive=true)

makedocs(;
    modules=[Copulas],
    authors="Oskar Laverny <oskar.laverny@univ-amu.fr> and contributors",
    repo="https://github.com/lrnv/Copulas.jl/blob/{commit}{path}#{line}",
    sitename="Copulas.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lrnv.github.io/Copulas.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Miscellaneous Copulas" => "miscellaneous.md",
        "Elliptical Copulas" => "ellipticals.md",
        "Archimedean Copulas" => "archimedeans.md",
        "Sklar Distributions" => "sklardist.md"
    ],
)

deploydocs(;
    repo="github.com/lrnv/Copulas.jl",
    devbranch="main",
)
