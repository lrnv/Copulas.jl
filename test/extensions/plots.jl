# Extension contract: verifies that loading Plots activates the documented
# Copula and SklarDist recipes without requiring a graphical display.
using Plots

@testset "Plots extension" begin
    @test Base.get_extension(Copulas, :CopulasPlotsExt) !== nothing

    C = ClaytonCopula{2}(1.5)
    S = SklarDist(C, (Normal(), Exponential()))

    copula_plot = plot(C; n=0, show_marginals=false)
    @test copula_plot isa Plots.Plot

    sklar_plot = plot(S, :cdf; n=0, overlay_n=5, show_marginals=false)
    @test sklar_plot isa Plots.Plot

    multivariate_plot = plot(ClaytonCopula{3}(1.5); n=2,
                             show_corr=false)
    @test multivariate_plot isa Plots.Plot

    @test_throws ArgumentError plot(S, :cdf; n=0,
                                    show_marginals=false, scale=:invalid)
end
