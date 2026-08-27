# Routing obligation: discover every copula method selected by the public
# fixtures and exercise one representative of each distinct dispatch route.
_which(f, args...) = which(f, Tuple{typeof.(args)...})

function _dispatch_path(operation, C, case)
    d = length(C)
    u = fill(0.6, d)
    if operation === :cdf
        return _which(Copulas._cdf, C, u)
    elseif operation === :logpdf
        case.kind === :continuous || return nothing
        return _which(Distributions._logpdf, C, u)
    elseif operation === :sampling
        return _which(Distributions._rand!, StableRNG(51), C, zeros(d, 1))
    elseif operation === :conditioning
        js = Tuple(1:(d - 1))
        values = ntuple(_ -> 0.4, d - 1)
        return _which(Copulas.DistortionFromCop, C, js, values, d)
    elseif operation === :rosenblatt
        case.rosenblatt || return nothing
        return _which(Copulas.rosenblatt, C, reshape(u, :, 1))
    elseif operation === :subsetting
        dims = d == 2 ? (2, 1) : (1, d)
        return _which(Copulas.subsetdims, C, dims)
    end
    error("unknown dispatch operation $operation")
end

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
    elseif operation === :rosenblatt
        @test size(rosenblatt(C, reshape(u, :, 1))) == (d, 1)
    elseif operation === :subsetting
        @test length(subsetdims(C, d == 2 ? (2, 1) : (1, d))) == 2
    end
end

@testset "one representative per copula dispatch mechanism" begin
    models = Tuple((case=case, copula=case.build()) for case in COPULA_CASES)
    for operation in (:cdf, :logpdf, :sampling, :conditioning, :rosenblatt, :subsetting)
        seen = Set{Any}()
        for (; case, copula) in models
            method = _dispatch_path(operation, copula, case)
            isnothing(method) && continue
            key = (method, length(copula) == 2 ? :bivariate : :multivariate)
            key in seen && continue
            push!(seen, key)
            @info "Testing dispatch mechanism" operation copula=case.name method
            _exercise_dispatch_path(operation, copula)
        end
        @test !isempty(seen)
    end
end
