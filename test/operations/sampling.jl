# Operation proof for `rand` and `rand!`: public adapters, sampler
# distributional correctness, buffer semantics, and dispatch-route closure.

function sampling_route_key(C)
    Base.@nospecialize C
    rng = StableRNG(51)
    method = which(Distributions._rand!,
                   Tuple{typeof(rng),typeof(C),Matrix{Float64}})
    return (method, length(C) == 2 ? :bivariate : :multivariate)
end

@testset verbose=true "public sampling contract" begin
    @testset "$(fixture.case.name)" for (seed, fixture) in enumerate(COPULA_FIXTURES)
        case, C = fixture.case, fixture.copula
        d = length(C)
        test_progress("operations", "sampling", case.name, "contract")

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
        test_progress("operations", "sampling", case.name, "route")
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
