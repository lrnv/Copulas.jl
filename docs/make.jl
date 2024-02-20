using Copulas
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(Copulas, :DocTestSetup, :(using Copulas); recursive=true)

bib = CitationBibliography(
    joinpath(@__DIR__,"src","assets","references.bib"),
    style=:numeric
)

makedocs(;
    plugins=[bib],
    modules=[Copulas],
    authors="Oskar Laverny <oskar.laverny@univ-amu.fr> and contributors",
    repo="https://github.com/lrnv/Copulas.jl/blob/{commit}{path}#{line}",
    sitename="Copulas.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lrnv.github.io/Copulas.jl",
        assets=String["assets/citations.css"],
        collapselevel=3,
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "getting_started.md",
            "Sklar's Distributions" => "sklar.md",
            "Elliptical Copulas" => "elliptical/generalities.md",
            "Archimedean Copulas" => "archimedean/generalities.md",
            "Liouville Copulas" => "Liouville.md",
            "Empirical Copulas" => "empirical/generalities.md",
            "Dependence measures" => "dependence_measures.md",
        ],
        
        "Bestiary" => [
            "elliptical/available_models.md",
            "archimedean/available_models.md",
            "empirical/available_models.md",
            "Other Copulas" => "miscellaneous.md",
            "Transformed Copulas" => "transformations.md",
        ],
        "Examples" => [
            "examples/lossalae.md",
            "examples/fitting_sklar.md",
            "examples/turing.md",
            "examples/other_usecases.md"
        ],
        "Dev Roadmap" => "dev_roadmap.md",
        "Package Index" => "idx.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/lrnv/Copulas.jl",
    devbranch="main",
)
