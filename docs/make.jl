using Copulas
using Documenter
using DocumenterCitations
using DocumenterVitepress

DocMeta.setdocmeta!(Copulas, :DocTestSetup, :(using Copulas); recursive=true)

bib = CitationBibliography(
    joinpath(@__DIR__,"src","assets","references.bib"),
    style=:numeric
)

makedocs(;
    plugins=[bib],
    modules=[Copulas],
    repo = Remotes.GitHub("lrnv", "Copulas.jl"),
    authors="Oskar Laverny <oskar.laverny@univ-amu.fr> and contributors",
    sitename="Copulas.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/lrnv/Copulas.jl",
    ),
    pages=[
        "Home"=>"index.md",
        "Manual" => [
            "Introduction"=>"manual/intro.md",
            "Conditioning and subsetting"=>"manual/conditioning_and_subsetting.md",
            "Dependence metrics"=>"manual/dependence_measures.md",
            "Fitting"=>"manual/fitting_interface.md",
            "Fitting"=>"manual/developer_fitting.md",
            "Visualizations"=>"manual/visualizations.md",

        ],
        "Bestiary" => [
            "Elliptical copulas"=>"bestiary/elliptical.md",
            "Archimedean copulas"=>"bestiary/archimedean.md",
            "Extreme Value copulas"=>"bestiary/extremevalues.md",
            "Archimax copulas"=>"bestiary/archimax.md",
            "Empirical copulas"=>"bestiary/empirical.md",
            "Vines copulas"=>"bestiary/vines.md",
            "Other copulas"=>"bestiary/miscellaneous.md",
        ],
        "Examples" => [
            "Nonparametric estimation of the radial law in Archimedean copulas"=>"examples/archimedean_radial_estimation.md",
            "Empirical Kendall function and Archimedean's λ function."=>"examples/lambda_viz.md",
            "Loss-Alae fitting example"=>"examples/lossalae.md",
            "Fitting compound distributions"=>"examples/fitting_sklar.md",
            "Influence of the method of estimation"=>"examples/ifm1.md",
            "Bayesian inference with `Turing.jl`"=>"examples/turing.md",
            "Other known use cases"=>"examples/other_usecases.md"
        ],
        "API" => [
            "Public"=>"api/public.md",
            "Internal (non-stable)"=>"api/internal.md",
        ],
        "References" => "references.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/lrnv/Copulas.jl",
    target = "build", # this is where Vitepress stores its output
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true,
)
