# Operation proof for `rand` and `rand!`: public adapters, sampler
# distributional correctness, buffer semantics, and dispatch-route closure.

function sampling_route_key(C)
    Base.@nospecialize C
    rng = StableRNG(51)
    method = which(Distributions._rand!,
                   Tuple{typeof(rng),typeof(C),Matrix{Float64}})
    return (method, length(C) == 2 ? :bivariate : :multivariate)
end

@testset "public sampling contract" begin
    @testset "$(fixture.case.name)" for (seed, fixture) in enumerate(COPULA_FIXTURES)
        case, C = fixture.case, fixture.copula
        d = length(C)

        buffer = zeros(eltype(C), d, 2)
        @test rand!(StableRNG(20_000 + seed), C, buffer) === buffer
        @test all(x -> 0 <= x <= 1, buffer)

        x = rand(StableRNG(30_000 + seed), C)
        @test length(x) == d
        @test eltype(x) == eltype(C)
        @test all(y -> 0 <= y <= 1, x)
    end

    @test_throws ArgumentError rand!(
        StableRNG(42), MissingSamplerContractCopula(), zeros(2, 1))
end

@testset verbose=true "one distributional identity per sampler dispatch" begin
    selected_routes = Set(sampling_route_key(fixture.copula)
                          for fixture in COPULA_FIXTURES)
    tested_routes = Set{Any}()
    for (index, fixture) in pairs(COPULA_FIXTURES)
        case, C = fixture.case, fixture.copula
        key = sampling_route_key(C)
        key in tested_routes && continue

        d = length(C)
        route_rng = StableRNG(400 + index)
        n = 160
        U = rand(route_rng, C, n)
        point = fill(0.72, d)
        theoretical = cdf(C, point)
        empirical = mean(all(U .<= point; dims=1))
        se = sqrt(max(theoretical * (1 - theoretical), eps()) / n)
        @test abs(empirical - theoretical) <= max(6se, 0.08)
        @test all(abs(mean(view(U, i, :)) - 0.5) <= 0.12 for i in 1:d)
        push!(tested_routes, key)
    end
    @test tested_routes == selected_routes
end

@testset "generic numeric sampler buffers" begin
    C = ClaytonCopula{3}(1.0)
    storage = fill(Float32(NaN), 5, 2)
    buffer = @view storage[2:4, :]
    @test rand!(StableRNG(52), C, buffer) === buffer
    @test all(x -> 0 <= x <= 1, buffer)
    @test all(isnan, storage[[1, 5], :])
    @test_throws DimensionMismatch rand!(StableRNG(52), C, zeros(Float32, 2, 1))
end

@testset "default sample element types" begin
    Carch = ClaytonCopula{2}(Float32(1))
    Cev = GalambosCopula{2}(Float32(1))
    Cell = GaussianCopula{2}(Float32(0.3))
    Cliouville = LiouvilleCopula{2}(
        Copulas.ClaytonGenerator(Float32(1)), (Float32(1), Float32(2)))
    Carchimax = ArchimaxCopula{2}(
        Copulas.ClaytonGenerator(Float32(1)),
        Copulas.GalambosTail(Float64(1)),
    )
    Cstudent = TCopula{2}(
        Float32(4), Float32[1 0.3; 0.3 1])
    Cempirical = EmpiricalCopula{2}(
        Float32[0.2 0.4 0.6 0.8; 0.8 0.6 0.4 0.2])

    cases = (
        Carch => Float32,
        Cev => Float32,
        Cell => Float32,
        Cstudent => Float32,
        FGMCopula{2}(Float32(0.5)) => Float32,
        PlackettCopula{2}(Float32(2)) => Float32,
        RafteryCopula{2}(Float32(0.5)) => Float32,
        Cempirical => Float32,
        Cliouville => Float32,
        SurvivalCopula(Carch, (1,)) => Float32,
        subsetdims(ClaytonCopula{3}(Float32(1)), (1, 2)) => Float32,
        Carchimax => Float64,
        IndependentCopula{2}() => Float64,
    )
    for (C, T) in cases
        @test eltype(C) === T
        @test eltype(rand(StableRNG(61), C)) === T
        @test eltype(rand(StableRNG(62), C, 2)) === T
    end

    big_cases = (
        ClaytonCopula{2}(big"1.0"),
        ClaytonCopula{2}(big"0.0"),
        GalambosCopula{2}(big"1.0"),
        PlackettCopula{2}(big"2.0"),
    )
    for C in big_cases
        @test eltype(C) === BigFloat
        @test eltype(rand(StableRNG(65), C, 2)) === BigFloat
    end

    # A Sklar distribution samples on its marginal scales. The copula's
    # probability representation therefore must not widen Float32 margins.
    S = SklarDist(IndependentCopula{2}(),
                  (Normal(Float32(0), Float32(1)),
                   Exponential(Float32(1))))
    @test eltype(S) === Float32
    @test eltype(rand(StableRNG(63), S)) === Float32
    @test eltype(rand(StableRNG(64), S, 2)) === Float32
end
