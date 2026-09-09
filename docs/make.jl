using Copulas
using Documenter
using DocumenterCitations
using DocumenterVitepress

# Temporary profiling hook; remove after the documentation runtime audit.
if get(ENV, "COPULAS_PROFILE_DOCS", "false") == "true"
    @eval Documenter function expand(doc::Documenter.Document)
        expandfirst = map(normpath, doc.user.expandfirst)
        priority_pages = filter(src -> src in keys(doc.blueprint.pages), expandfirst)
        normal_pages = sort!(filter(src -> !(src in priority_pages), collect(keys(doc.blueprint.pages))))
        for src in Iterators.flatten((priority_pages, normal_pages))
            started = time_ns()
            page = doc.blueprint.pages[src]
            copy!(page.globals.meta, doc.user.meta)
            for node in collect(page.mdast.children)
                Selectors.dispatch(Expanders.ExpanderPipeline, node, page, doc)
                expand_recursively(node, page, doc)
            end
            collect_named_anchors!(page, doc)
            pagecheck(doc, page)
            clear_modules!(page.globals.meta)
            @info "Documentation page expanded" page=src seconds=round((time_ns() - started) / 1e9; digits=3)
        end
        return
    end
end

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
            "Mixture models with ExpectationMaximization.jl"=>"examples/expectation_maximization.md",
            "Bayesian inference with Turing.jl"=>"examples/turing.md",
            "Loss-ALAE fitting"=>"examples/lossalae.md",
            "Other use cases"=>"examples/other_usecases.md",
        ],
        "API" => [
            "Public"=>"api/public.md",
        ],
        "Development" => [
            "Extending Copulas.jl"=>"dev/developer_guide.md",
            "Internal implementation reference"=>"api/internal.md",
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
