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

    for row in axes(Xtied, 1)
        x = @view Xtied[row, :]
        @test pseudos(reshape(x, 1, :); ties=:average)[1, :] .* denominator ==
              StatsBase.tiedrank(x)
        @test pseudos(reshape(x, 1, :); ties=:first)[1, :] .* denominator ==
              StatsBase.ordinalrank(x)
        @test pseudos(reshape(x, 1, :); ties=:min)[1, :] .* denominator ==
              StatsBase.competerank(x)
    end

    random1 = pseudos(Xtied; ties=:random, rng=Xoshiro(93))
    random2 = pseudos(Xtied; ties=:random, rng=Xoshiro(93))
    @test random1 == random2
    @test sort(random1[1, 1:2]) == [1.0, 2.0] ./ denominator
    @test sort(random1[2, 2:3]) == [2.0, 3.0] ./ denominator

    for ties in (:average, :first, :last, :min, :max)
        rng = Xoshiro(94)
        control = Xoshiro(94)
        pseudos(Xtied; ties, rng)
        @test rand(rng) == rand(control)
    end

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

    @testset "weighted pseudo-observations" begin
        n = size(Xtied, 2)
        # Unit weights reproduce every convention exactly, whatever their scale
        # up to rounding, and a range or an integer vector is accepted.
        for ties in (:average, :first, :last, :min, :max)
            @test pseudos(Xtied; ties, weights=ones(n)) == pseudos(Xtied; ties)
            @test pseudos(Xtied; ties, weights=fill(3, n)) == pseudos(Xtied; ties)
            @test pseudos(Xtied; ties, weights=fill(0.3, n)) ≈ pseudos(Xtied; ties)
        end
        @test pseudos(Xtied; ties=:random, rng=Xoshiro(93), weights=ones(n)) ==
              pseudos(Xtied; ties=:random, rng=Xoshiro(93))
        # The ranks are computed in the type the sample and the weights
        # promote to, so wide weights are not silently narrowed.
        @test eltype(pseudos(Float32.(X); weights=ones(Float32, n))) === Float32
        @test eltype(pseudos(Float32.(X); weights=ones(n))) === Float64
        wide = pseudos(X; weights=big.(ones(n)))
        @test eltype(wide) === BigFloat
        @test Float64.(wide) == pseudos(X)
        @test eltype(pseudos(X; weights=1:n)) === Float64

        # Integer weights that sum to n are counts: each observation takes the
        # mean rank of its copies in the replicated sample, under every
        # convention, and a zero-weight observation sits at the weighted
        # empirical distribution function of its value.
        Xw = [30.0 10.0 20.0 40.0 20.0; 4.0 6.0 5.0 7.0 6.0]
        weights = [2, 0, 1, 1, 1]
        kept = findall(>(0), weights)
        Xrep = hcat((repeat(Xw[:, j], 1, weights[j]) for j in kept)...)
        first_copy = cumsum([0; weights[kept][1:end-1]]) .+ 1
        for ties in (:average, :first, :last, :min, :max)
            P = pseudos(Xw; ties, weights)
            Q = pseudos(Xrep; ties)
            @test all(x -> 0 < x < 1, P)
            for (k, j) in enumerate(kept)
                copies = first_copy[k]:(first_copy[k] + weights[j] - 1)
                @test P[:, j] ≈ vec(Statistics.mean(Q[:, copies]; dims=2))
            end
        end
        @test pseudos(Xw; weights) ≈ pseudos(Xw; weights=2 .* weights)
        # Column 2 has weight zero: alone at the bottom of margin 1, tied with
        # column 5 in margin 2, where the tie block still carries mass.
        @test pseudos(Xw; weights)[:, 2] == [0.5, 4.0] ./ 6
        @test pseudos(Xw; ties=:min, weights)[:, 2] == [0.5, 4.0] ./ 6
        @test pseudos(Xw; ties=:max, weights)[:, 2] == [0.5, 4.0] ./ 6
        @test pseudos(Xw; ties=:first, weights)[:, 2] == [0.5, 3.5] ./ 6
        @test pseudos(Xw; ties=:last, weights)[:, 2] == [0.5, 4.5] ./ 6

        @test_throws DimensionMismatch pseudos(Xtied; weights=ones(n - 1))
        @test_throws ArgumentError pseudos(Xtied; weights=-ones(n))
        @test_throws ArgumentError pseudos(Xtied; weights=zeros(n))
        @test_throws ArgumentError pseudos(Xtied; weights=[1.0, NaN, 1.0, 1.0])
        @test_throws ArgumentError pseudos(Xtied; weights=ones(2, 2))
    end

    kendall_data = [1.0 1.0 2.0 3.0; 1.0 2.0 1.0 3.0]
    kendall = Copulas._kendall_sample(kendall_data)
    @test kendall == [1.0, 2.0, 2.0, 4.0] ./ 5
    @test Copulas._kendall_sample(pseudos(kendall_data)) == kendall
    permuted_kendall = Copulas._kendall_sample(kendall_data[:, permutation])
    @test permuted_kendall[invperm(permutation)] == kendall


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
