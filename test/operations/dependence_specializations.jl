# Dependence-operation equivalence proofs for scalar and pairwise routes.

function _unique_dependence_routes(operation, predicate)
    Base.@nospecialize operation predicate
    seen = Set{Method}()
    routes = NamedTuple[]
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 2 || continue
        predicate(case, C) || continue
        method = operation(case, C)
        method in seen && continue
        push!(seen, method)
        push!(routes, (; case, C, method))
    end
    return routes
end

@testset verbose=true "specialized dependence measures agree with generic definitions" begin
    # Entropy and Gini's gamma use substantially more expensive multidimensional
    # expectations and are covered by their independent identities in
    # correctness/. Kendall's generic definition is stochastic, so singular
    # Kendall formulas keep their exact family identities instead of a noisy,
    # repeated 10_000-observation comparison here. The CDF-only definitions of
    # rho, beta and tail dependence remain valid for singular and mixed laws.
    @testset verbose=true "$(nameof(SCALAR_DEPENDENCE_MEASURES[index]))" for index in (1, 2, 3, 6, 7)
        measure = SCALAR_DEPENDENCE_MEASURES[index]
        routes = _unique_dependence_routes(
            (_, C) -> which(measure, Tuple{typeof(C)}),
            (_, C) -> measure === Copulas.τ ?
                is_absolutely_continuous(C) : true,
        )
        generic_method = which(measure, Tuple{Copulas.Copula{2}})
        for (; case, C, method) in routes
            if method === generic_method
                # The bivariate generic mechanism is proved independently by
                # PolynomialOracleCopula in correctness/mathematical.jl.
                @test dependence_route_key(measure, C) in
                      PROVEN_DEPENDENCE_ROUTES[measure]
                continue
            end
            @testset "$(case.name)" begin
                test_progress("equivalence", nameof(measure), case.name)
                if measure === Copulas.τ &&
                   (C isa GaussianCopula || C isa TCopula)
                    # Kendall's tau is invariant over the radial distribution
                    # of an elliptical copula.  At ρ = 1/2, the exact identity
                    # 2asin(ρ)/π = 1/3 validates both elliptical
                    # specializations without repeatedly evaluating their
                    # expensive numerical CDFs inside a cubature.
                    reference = C isa GaussianCopula ?
                        GaussianCopula{2}(0.5) :
                        TCopula{2}(C.df, [1.0 0.5; 0.5 1.0])
                    @test Copulas.τ(reference) ≈ 1 / 3 atol=2e-15
                elseif measure === Copulas.ρ && C isa GaussianCopula
                    # The bivariate Gaussian identity avoids nesting the
                    # numerical normal CDF inside the generic rho cubature.
                    reference = GaussianCopula{2}(0.5)
                    @test Copulas.ρ(reference) ≈ 6asin(0.25) / π atol=2e-15
                else
                    expected = measure === Copulas.τ ?
                        4 * HCubature.hcubature(u -> cdf(C, u) * pdf(C, u),
                                               zeros(2), ones(2); rtol=1e-5)[1] - 1 :
                        invoke(measure, Tuple{Copulas.Copula}, C)
                    @test isapprox(measure(C), expected; atol=3e-4, rtol=3e-4)
                end
            end
            prove_dependence_route!(measure, C)
        end
    end
end

@testset "limit and subset dependence routes agree with independent identities" begin
    independence = IndependentCopula{2}()
    @test Copulas.γ(independence) == 0
    @test Copulas.ι(independence) == 0
    @test Copulas.γ(MCopula{2}()) == 1
    @test Copulas.ι(MCopula{2}()) == -Inf

    parent = ClaytonCopula{2}(1.5)
    subset = subsetdims(parent, (2, 1))
    # Gamma and entropy use stochastic generic expectations. Their forwarding
    # dispatches are inventoried below; exact value equality is meaningful only
    # for the deterministic measures.
    for measure in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λₗ, Copulas.λᵤ)
        @test measure(subset) == measure(parent)
    end
end

_spectral_matrix(tail::Copulas.DiscreteSpectralTail) = tail.B
_spectral_matrix(tail::Union{Copulas.BC2Tail,Copulas.MOTail}) = tail.spectral.B

function _spectral_curvature_tau(tail)
    # If A(t) = sum_k max(B[1,k]t, B[2,k](1-t)), its second derivative is
    # the discrete measure placing mass B[1,k] + B[2,k] at the corresponding
    # kink.  This is the distributional version of the defining EV Kendall
    # integral and, unlike a sample-concordance check, is exact and noiseless.
    B = _spectral_matrix(tail)
    total = zero(eltype(B))
    for k in axes(B, 2)
        mass = B[1, k] + B[2, k]
        iszero(mass) && continue
        kink = B[2, k] / mass
        total += mass * kink * (1 - kink) / Copulas.A(tail, kink)
    end
    return total
end

_singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.BC2Tail}) =
    _spectral_curvature_tau(C.tail)

function _singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.MOTail})
    # Classical competing-shocks identity. The public bivariate constructor
    # stores private shocks in subset order ([2], [1], [1,2]).
    λ₁, λ₂, λ₁₂ = C.tail.λ[2], C.tail.λ[1], C.tail.λ[3]
    a = λ₁ / (λ₁ + λ₁₂)
    b = λ₂ / (λ₂ + λ₁₂)
    return a * b / (a + b - a * b)
end

_singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.DiscreteSpectralTail}) =
    _spectral_curvature_tau(C.tail)

function _singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.CuadrasAugeTail})
    # Its Pickands function has one kink at 1/2 with slope jump 2θ.
    kink = 0.5
    return 2C.tail.θ * kink * (1 - kink) / Copulas.A(C.tail, kink)
end

# This is the classical bivariate Raftery identity, independently obtained
# from its common-factor mixture representation.
_singular_tau_oracle(C::RafteryCopula{2}) = 2C.θ / (3 - C.θ)
_singular_tau_oracle(::MCopula{2}) = 1
_singular_tau_oracle(::WCopula{2}) = -1
# The order-two Williamson transform of a Dirac radial is
# ϕ(t) = (1 - t/r)₊, hence its bivariate copula is the lower
# Fréchet--Hoeffding bound independently of the positive radius r.
_singular_tau_oracle(
    ::ArchimedeanCopula{2,<:WilliamsonGenerator{<:Dirac}},
) = -1

@testset verbose=true "singular Kendall routes agree with deterministic identities" begin
    routes = _unique_dependence_routes(
        (_, C) -> which(Copulas.τ, Tuple{typeof(C)}),
        (_, C) -> !is_absolutely_continuous(C),
    )
    generic_method = which(Copulas.τ, Tuple{Copulas.Copula{2}})
    compared = 0
    for route in routes
        (; case, C, method) = route
        method === generic_method && continue
        @testset "$(case.name)" begin
            test_progress("equivalence", "singular Kendall", case.name)
            expected = _singular_tau_oracle(C)
            @test Copulas.τ(C) ≈ expected atol=2e-12 rtol=2e-12
        end
        prove_dependence_route!(Copulas.τ, C)
        compared += 1
    end
    @test compared > 0
end

@testset "all gamma and entropy dispatches have an independent proof" begin
    parent = ClaytonCopula{2}(1.5)
    subset = subsetdims(parent, (2, 1))
    candidates = Any[]
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 2 && push!(candidates, C)
    end
    push!(candidates, subset)

    for (measure, checked) in (
        (Copulas.γ, (PolynomialOracleCopula(0.4), IndependentCopula{2}(),
                     MCopula{2}(), subset)),
        (Copulas.ι, (PolynomialOracleCopula(0.4), IndependentCopula{2}(),
                     MCopula{2}(), subset)),
    )
        selected_methods = Set(which(measure, Tuple{typeof(C)}) for C in candidates)
        checked_methods = Set(which(measure, Tuple{typeof(C)}) for C in checked)
        @test selected_methods == checked_methods
        for C in checked
            prove_dependence_route!(measure, C)
        end
    end
end

@testset "every pairwise dependence route reduces to bivariate margins" begin
    scalar = Dict(
        StatsBase.corkendall => Copulas.τ,
        StatsBase.corspearman => Copulas.ρ,
        Copulas.corblomqvist => Copulas.β,
        Copulas.corgini => Copulas.γ,
        Copulas.corentropy => Copulas.ι,
        Copulas.corlowertail => Copulas.λₗ,
        Copulas.coruppertail => Copulas.λᵤ,
    )
    for (pairwise, diagonal) in PAIRWISE_DEPENDENCE_MEASURES
        selected = Set((which(pairwise, Tuple{typeof(fixture.copula)}),
                        length(fixture.copula) == 2 ? :bivariate : :multivariate)
                       for fixture in ROUTING_COPULA_FIXTURES
                       if _dependence_is_defined(pairwise, fixture.copula))
        checked = Set{Any}()
        for fixture in ROUTING_COPULA_FIXTURES
            case, C = fixture.case, fixture.copula
            _dependence_is_defined(pairwise, C) || continue
            key = (which(pairwise, Tuple{typeof(C)}),
                   length(C) == 2 ? :bivariate : :multivariate)
            key in checked && continue
            # Generic gamma and entropy estimators sample internally. Reusing
            # the same RNG state makes this an exact forwarding test instead
            # of comparing two independent Monte Carlo estimates.
            seed = 0x51a7 + hash((pairwise, key))
            Random.seed!(seed)
            observed = pairwise(C)
            Random.seed!(seed)
            if C isa EmpiricalCopula &&
               pairwise in (StatsBase.corkendall, StatsBase.corspearman)
                expected = pairwise(transpose(C.u))
            else
                d = length(C)
                expected = Matrix{Float64}(I, d, d) .* diagonal
                for i in 1:d, j in 1:(i - 1)
                    value = scalar[pairwise](subsetdims(C, (i, j)))
                    expected[i, j] = expected[j, i] = value
                end
            end
            @test observed ≈ expected atol=1e-8
            push!(checked, key)
        end
        @test checked == selected
    end
end

@testset "multivariate Archimedean and Raftery dependence identities" begin
    # These closed forms are dimension-dependent dispatch routes and therefore
    # cannot be represented by the bivariate specialization comparison above.
    clayton = ClaytonCopula{3}(1.5)
    @test Copulas.τ(clayton) ≈ 3 / 7
    # The generator specialization is dimension invariant; its bivariate
    # value is independently checked against the generic integral above.
    @test Copulas.ρ(clayton) == Copulas.ρ(ClaytonCopula{2}(1.5))
    prove_dependence_route!(Copulas.τ, clayton)
    prove_dependence_route!(Copulas.ρ, clayton)

    raftery = RafteryCopula{3}(0.5)
    @test Copulas.τ(raftery) ≈ 0.4
    @test Copulas.ρ(raftery) ≈ 13 / 27
    prove_dependence_route!(Copulas.τ, raftery)
    prove_dependence_route!(Copulas.ρ, raftery)
end
