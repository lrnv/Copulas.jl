# Routing obligation: discover every copula method selected by the public
# fixtures and exercise one representative of each distinct dispatch route.
function _exercise_dispatch_path(operation, C)
    d = length(C)
    u = fill(0.6, d)
    if operation === :cdf
        @test 0 <= cdf(C, u) <= 1
    elseif operation === :logpdf
        @test !isnan(logpdf(C, u))
    elseif operation === :sampling
        @test size(rand(StableRNG(51), C, 2)) == (d, 2)
    elseif operation === :conditioning
        D = condition(C, Tuple(1:(d - 1)), ntuple(_ -> 0.4, d - 1))
        @test 0 <= cdf(D, 0.6) <= 1
    elseif operation === :conditional_joint
        H = condition(C, (1,), (0.4,))
        @test 0 <= cdf(H, fill(0.6, d - 1)) <= 1
    elseif operation === :rosenblatt
        @test size(rosenblatt(C, reshape(u, :, 1))) == (d, 1)
    elseif operation === :inverse_rosenblatt
        @test size(inverse_rosenblatt(C, reshape(u, :, 1))) == (d, 1)
    elseif operation === :subsetting
        @test length(subsetdims(C, d == 2 ? (2, 1) : (1, d))) == 2
    end
end

@testset verbose=true "one representative per copula dispatch mechanism" begin
    models = ROUTING_COPULA_FIXTURES
    operations = (:cdf, :logpdf, :sampling, :conditioning,
                  :conditional_joint, :rosenblatt, :inverse_rosenblatt,
                  :subsetting)
    @testset verbose=true "$operation" for operation in operations
        seen = Set{Any}()
        for (; case, copula) in models
            method = dispatch_path(operation, copula, case)
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
    deterministic = (:cdf, :logpdf, :conditioning, :conditional_joint,
                     :rosenblatt, :inverse_rosenblatt, :subsetting)
    @testset "$operation" for operation in deterministic
        selected = Set{Any}()
        for fixture in ROUTING_COPULA_FIXTURES
            case, C = fixture.case, fixture.copula
            key = dispatch_route_key(operation, C, case)
            isnothing(key) || push!(selected, key)
        end
        proven = Set(keys(get(PROVEN_DISPATCH_ROUTES, operation,
                              Dict{Any,Set{Symbol}}())))
        missing = setdiff(selected, proven)
        isempty(missing) || @info "Dispatch routes without a proof" operation missing
        @test isempty(missing)
    end
end
