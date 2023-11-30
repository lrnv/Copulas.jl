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
        "Copulas.jl package" => "index.md",
        "Getting Started" => "getting_started.md",
        "Sklar's Distributions" => "sklar.md",
        "Elliptical Copulas" => [
            "elliptical/generalities.md",
            "elliptical/available_models.md",
        ],
        "Archimedean Copulas" => [
            "archimedean/generalities.md",
            "archimedean/available_models.md",
        ],
        "Liouville Copulas" => "Liouville.md",
        "Other Copulas" => "miscellaneous.md",
        "Transformed Copulas" => "transformations.md",
        "Empirical Copulas" => [
            "empirical/generalities.md",
            "empirical/available_models.md",
        ],
        "Dependence measures" => "dependence_measures.md",
        "Exemples" => [
            "exemples/fitting_sklar.md",
            "exemples/turing.md",
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
