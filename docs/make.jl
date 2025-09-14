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
        "index.md",
        "Manual" => [
            "manual/getting_started.md",
            "manual/sklar.md",
            "manual/dependence_measures.md",
            "manual/conditioning_and_subsetting.md",
            "manual/visualizations.md",
            "manual/archimedean.md",
            "manual/elliptical.md",
            "manual/extremevalues.md",
            "manual/archimax.md",
            "manual/vines.md",
            "manual/empirical.md",
            "manual/troubleshooting.md",
        ],
        "Bestiary" => [
            "bestiary/indep_and_fh_bouds.md",
            "bestiary/elliptical.md",
            "bestiary/archimedean.md",
            "bestiary/extremevalues.md",
            "bestiary/archimax.md",
            "bestiary/empirical.md",
            "Other Copulas" => "bestiary/miscellaneous.md",
            "Transformed Copulas" => "bestiary/transformations.md",
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
        "Package Index" => "idx.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/lrnv/Copulas.jl",
    devbranch="main",
)
