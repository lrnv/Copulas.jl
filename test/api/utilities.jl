# Public-API proof: checks standalone public functions and data-based
# dependence measures that do not naturally belong to one model contract.
@testset "standalone public utilities" begin
    X = [3.0 1.0 2.0 4.0; 2.0 4.0 1.0 3.0]
    U = pseudos(X)
    @test size(U) == size(X)
    @test all(x -> 0 < x < 1, U)
    @test pseudos(U) == U
    @test eltype(pseudos(Float32.(X))) === Float32

    Xtied = [1.0 1.0 2.0 4.0; 4.0 3.0 3.0 1.0]
    denominator = 5
    @test pseudos(Xtied; ties=:average) ==
          [1.5 1.5 3.0 4.0; 4.0 2.5 2.5 1.0] ./ denominator
    @test pseudos(Xtied; ties=:first) ==
          [1.0 2.0 3.0 4.0; 4.0 2.0 3.0 1.0] ./ denominator
    @test pseudos(Xtied; ties=:last) ==
          [2.0 1.0 3.0 4.0; 4.0 3.0 2.0 1.0] ./ denominator
    @test pseudos(Xtied; ties=:min) ==
          [1.0 1.0 3.0 4.0; 4.0 2.0 2.0 1.0] ./ denominator
    @test pseudos(Xtied; ties=:max) ==
          [2.0 2.0 3.0 4.0; 4.0 3.0 3.0 1.0] ./ denominator

    random1 = pseudos(Xtied; ties=:random, rng=Xoshiro(93))
    random2 = pseudos(Xtied; ties=:random, rng=Xoshiro(93))
    @test random1 == random2
    @test sort(random1[1, 1:2]) == [1.0, 2.0] ./ denominator
    @test sort(random1[2, 2:3]) == [2.0, 3.0] ./ denominator

    rng = Xoshiro(94)
    control = Xoshiro(94)
    pseudos(Xtied; ties=:average, rng)
    @test rand(rng) == rand(control)

    permutation = [4, 2, 1, 3]
    for ties in (:average, :min, :max)
        ranked = pseudos(Xtied; ties)
        permuted = pseudos(Xtied[:, permutation]; ties)
        @test permuted[:, invperm(permutation)] == ranked
    end

    for ties in (:average, :first, :last, :min, :max, :random)
        @test pseudos(X; ties, rng=Xoshiro(95)) == U
    end
    Utied = pseudos(Xtied)
    @test pseudos(Utied) == Utied
    @test_throws ArgumentError pseudos(Xtied; ties=:dense)


    target = [1.0 0.4; 0.4 1.0]
    @test Nataf((Normal(), Normal(2, 3)), target) == target
    @test Nataf([Normal(), Normal(2, 3)], target) == target
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
