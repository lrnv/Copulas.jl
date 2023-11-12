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
        collapselevel=3,
    ),
    pages=[
        "Home" => "index.md",
        "Sklar's Theorem" => "sklar.md",
        "Elliptical Copulas" => [
            "elliptical/generalities.md",
            "elliptical/available_models.md",
        ],
        "Archimedean Copulas" => [
            "archimedean/generalities.md",
            "archimedean/available_models.md",
            "archimedean/implement_your_own.md",
        ],
        "Miscellaneous Copulas" => "miscellaneous.md",
        "Exemples" => [
            "exemples/fitting_sklar.md",
            "exemples/turing.md",
        ],
        "Reference" => "reference.md"
    ],
)

deploydocs(;
    repo="github.com/lrnv/Copulas.jl",
    devbranch="main",
)
