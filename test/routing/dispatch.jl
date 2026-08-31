# Routing obligation: discover every copula method selected by the public
# fixtures and exercise one representative of each distinct dispatch route.

@testset "distribution adapters remain shared" begin
    # These public operations intentionally delegate to the scalar kernels
    # inventoried below. A direct family specialization must come with an
    # equivalence proof and an explicit route before this assertion is relaxed.
    signatures = (
        pdf = C -> Tuple{typeof(C),Vector{Float64}},
        logcdf = C -> Tuple{typeof(C),Vector{Float64}},
        loglikelihood = C -> Tuple{typeof(C),Matrix{Float64}},
    )
    functions = (pdf=Distributions.pdf, logcdf=Distributions.logcdf,
                 loglikelihood=Distributions.loglikelihood)
    for name in keys(signatures)
        selected = Set(which(functions[name], signatures[name](fixture.copula))
                       for fixture in ROUTING_COPULA_FIXTURES)
        @test length(selected) == 1
    end
end

@testset "every scalar dependence route has an oracle" begin
    for measure in SCALAR_DEPENDENCE_MEASURES
        selected = Set(dependence_route_key(measure, fixture.copula)
                       for fixture in ROUTING_COPULA_FIXTURES
                       if _dependence_is_defined(measure, fixture.copula))
        missing = setdiff(selected, PROVEN_DEPENDENCE_ROUTES[measure])
        isempty(missing) || @info "Dependence routes without an oracle" measure missing
        @test isempty(missing)
    end
end

function _exercise_dispatch_path(operation, C)
    Base.@nospecialize operation
    Base.@nospecialize C
    d = length(C)
    u = fill(0.6, d)
    if operation === :cdf
        @test 0 <= cdf(C, u) <= 1
    elseif operation === :logpdf
        @test !isnan(logpdf(C, u))
    elseif operation === :conditioning
        D = condition(C, Tuple(1:(d - 1)), ntuple(_ -> 0.4, d - 1))
        @test 0 <= cdf(D, 0.6) <= 1
    elseif operation === :conditional_joint
        H = condition(C, (1,), (0.4,))
        @test 0 <= cdf(H, fill(0.6, d - 1)) <= 1
    end
end

@testset verbose=true "one representative per copula dispatch mechanism" begin
    models = ROUTING_COPULA_FIXTURES
    operations = (:cdf, :logpdf, :conditioning, :conditional_joint)
    @testset verbose=true "$operation" for operation in operations
        seen = Set{Any}()
        for (; case, copula) in models
            method = dispatch_path(operation, copula)
            isnothing(method) && continue
            key = (method, length(copula) == 2 ? :bivariate : :multivariate)
            key in seen && continue
            push!(seen, key)
            @testset "$(case.name)" begin
                test_progress("routing", operation, case.name)
                _exercise_dispatch_path(operation, copula)
            end
        end
        @test !isempty(seen)
    end
end

@testset verbose=true "every selected deterministic route has a proof" begin
    deterministic = (:cdf, :logpdf, :conditioning, :conditional_joint)
    @testset "$operation" for operation in deterministic
        selected = Set{Any}()
        for fixture in ROUTING_COPULA_FIXTURES
            case, C = fixture.case, fixture.copula
            key = dispatch_route_key(operation, C)
            isnothing(key) || push!(selected, key)
        end
        proven = Set(keys(get(PROVEN_DISPATCH_ROUTES, operation,
                              Dict{Any,Set{Symbol}}())))
        missing = setdiff(selected, proven)
        isempty(missing) || @info "Dispatch routes without a proof" operation missing
        @test isempty(missing)
    end
end
