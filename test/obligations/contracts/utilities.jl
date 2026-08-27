# Public-API contract: checks standalone public functions and data-based
# dependence measures that do not naturally belong to one model contract.
@testset "standalone public utilities" begin
    X = [3.0 1.0 2.0 4.0; 2.0 4.0 1.0 3.0]
    U = pseudos(X)
    @test size(U) == size(X)
    @test all(x -> 0 < x < 1, U)
    @test pseudos(U) == U

    C = ClaytonCopula{2}(1.5)
    @test Copulas.measure(C, zeros(2), ones(2)) == 1
    @test Copulas.measure(C, [0.7, 0.2], [0.4, 0.8]) == 0
    @test 0 <= Copulas.measure(C, [0.2, 0.3], [0.7, 0.8]) <= 1
    @test Copulas.measure(C, (0.2, 0.3), (0.7, 0.8)) ≈
          Copulas.measure(C, [0.2, 0.3], [0.7, 0.8])

    target = [1.0 0.4; 0.4 1.0]
    @test Nataf((Normal(), Normal(2, 3)), target) == target
    @test Nataf((Uniform(), Uniform()), 0.4) ≈ 2sinpi(0.4 / 6)

    generic = Nataf((Gamma(2.0, 1.0), Beta(2.0, 3.0)), 0.2; nodes=8)
    @test -1 < generic < 1
    @test Nataf((Gamma(2.0, 1.0), Beta(2.0, 3.0)),
                [1.0 0.2; 0.2 1.0]; nodes=8)[1, 2] ≈ generic
    @test_throws ArgumentError Nataf((Normal(),), 0.2)
    @test_throws ArgumentError Nataf((Normal(), Normal()), 1.2)
    @test_throws ArgumentError Nataf((Normal(), Normal()), target; nodes=1)

    sample = rand(StableRNG(91), ClaytonCopula{2}(1.5), 80)
    for scalar in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ, Copulas.ι,
                   Copulas.λₗ, Copulas.λᵤ)
        @test scalar(sample) isa Real
    end
    for pairwise in (StatsBase.corkendall, StatsBase.corspearman,
                     Copulas.corblomqvist, Copulas.corgini,
                     Copulas.corentropy, Copulas.corlowertail,
                     Copulas.coruppertail)
        @test size(pairwise(transpose(sample))) == (2, 2)
    end

    observations = transpose(sample)
    @test size(Copulas.corlowertail(
        observations, :SchmidSchmidt, 0.25)) == (2, 2)
    @test size(Copulas.coruppertail(
        observations, :SchmidSchmidt, 0.25)) == (2, 2)

    sample3 = rand(StableRNG(92), ClaytonCopula{3}(1.5), 20)
    for scalar in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ,
                   Copulas.ι, Copulas.λₗ, Copulas.λᵤ)
        @test scalar(sample3) isa Real
    end
    for pairwise in (StatsBase.corkendall, StatsBase.corspearman,
                     Copulas.corblomqvist, Copulas.corgini,
                     Copulas.corentropy, Copulas.corlowertail,
                     Copulas.coruppertail)
        @test size(pairwise(transpose(sample3))) == (3, 3)
    end
end
