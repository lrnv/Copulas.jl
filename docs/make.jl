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
            "Sklar's Distribution" => "sklar.md",
            "Dependence measures" => "dependence_measures.md",
            "Conditioning and Subsetting" => "conditioning_and_subsetting.md",
            "Visualizations" => "visualizations.md",
            "Archimedean family" => "archimedean/generalities.md",
            "Elliptical family" => "elliptical/generalities.md",
            "Extreme Value family" => "extremevalue/generalities.md",
            "Archimax family" => "archimax/generalities.md",
            "Vines family" => "Vines.md",
            "Empirical models" => "empirical/generalities.md",
            "Troubleshooting" => "troubleshooting.md",
        ],
        
        "Bestiary" => [
            "elliptical/available_models.md",
            "archimedean/available_models.md",
            "extremevalue/available_models.md",
            "archimax/available_models.md",
            "empirical/available_models.md",
            "Other Copulas" => "miscellaneous.md",
            "Transformed Copulas" => "transformations.md",
        ],
        "Examples" => [
            "examples/archimedean_radial_estimation.md",
            "examples/lambda_viz.md",
            "examples/lossalae.md",
            "examples/fitting_sklar.md",
            "examples/ifm1.md",
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
