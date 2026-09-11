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
        keep = :patch,
    ),
    pages=[
        "Home"=>"index.md",
        "Manual" => [
            "Introduction"=>"manual/intro.md",
            "Conditioning and subsetting"=>"manual/conditioning_and_subsetting.md",
            "Dependence metrics"=>"manual/dependence_measures.md",
            "Fitting"=>"manual/fitting_interface.md",
            "Hypothesis testing" => "manual/hypothesis_testing.md",
            "Visualizations"=>"manual/visualizations.md",
        ],
        "Bestiary" => [
            "Elliptical copulas"=>"bestiary/elliptical.md",
            "Archimedean copulas"=>"bestiary/archimedean.md",
            "Liouville copulas"=>"bestiary/liouville.md",
            "Nested Archimedean copulas"=>"bestiary/nested.md",
            "Extreme Value copulas"=>"bestiary/extremevalues.md",
            "Archimax copulas"=>"bestiary/archimax.md",
            "Empirical copulas"=>"bestiary/empirical.md",
            "Vines copulas"=>"bestiary/vines.md",
            "Other copulas"=>"bestiary/miscellaneous.md",
        ],
        "Examples" => [
            "Nonparametric radial estimation"=>"examples/archimedean_radial_estimation.md",
            "Empirical Kendall function and Archimedean λ"=>"examples/lambda_viz.md",
            "Fitting compound distributions"=>"examples/fitting_sklar.md",
            "Influence of the estimation method"=>"examples/ifm1.md",
            "Fitting censored observations"=>"examples/censored_fitting.md",
            "Fitting Loss-ALAE"=>"examples/lossalae.md",
            "Mixture models with ExpectationMaximization.jl"=>"examples/expectation_maximization.md",
            "Interoperability with PartitionedDistributions.jl" => "examples/partitioned_distributions.md",
            "Bayesian inference with Turing.jl"=>"examples/turing.md",
            "Other use cases"=>"examples/other_usecases.md",
        ],
        "Public API" => "api/public.md",
        "Development" => [
            "Extending Copulas.jl"=>"dev/developer_guide.md",
            "Internals reference"=>"api/internal.md",
            "Performance benchmarks"=>"dev/benchmarks.md",
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
