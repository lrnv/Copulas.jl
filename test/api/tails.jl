# Component proof: exhaustively covers public EV-tail families and
# verifies stable-tail, Pickands, derivative, and reconstruction identities.

struct LogisticOracleTail{T} <: Copulas.BivariatePickandsTail
    θ::T
end
Distributions.params(tail::LogisticOracleTail) = (; θ=tail.θ)
Copulas.ℓ(tail::LogisticOracleTail, x) =
    sum(xᵢ -> xᵢ^tail.θ, x)^(inv(tail.θ))
Copulas.A(tail::LogisticOracleTail, t::Real) =
    Copulas.ℓ(tail, (t, 1 - t))
Copulas._is_valid_in_dim(::LogisticOracleTail, d::Int) = d >= 2

@testset "specialized logistic tail agrees with its generic oracle" begin
    θ = 1.5
    generic_tail = LogisticOracleTail(θ)
    specialized_tail = Copulas.LogTail(θ)
    x = [0.4, 0.7]
    weight = Tuple(x ./ sum(x))
    @test Copulas.ℓ(specialized_tail, x) ≈ Copulas.ℓ(generic_tail, x)
    @test Copulas.A(specialized_tail, weight) ≈ Copulas.A(generic_tail, weight)
    for indices in ((), (1,), (2,), (1, 2))
        @test Copulas.ellpartial(specialized_tail, x, indices) ≈
              Copulas.ellpartial(generic_tail, x, indices) atol=2e-6
    end

    generic = ExtremeValueCopula{2}(generic_tail)
    specialized = LogCopula{2}(θ)
    u = [0.37, 0.68]
    @test cdf(specialized, u) ≈ cdf(generic, u)
    @test pdf(specialized, u) ≈ pdf(generic, u) atol=2e-6

    generic_D = condition(generic, 1, u[1])
    specialized_D = condition(specialized, 1, u[1])
    @test cdf(specialized_D, u[2]) ≈ cdf(generic_D, u[2])
    @test pdf(specialized_D, u[2]) ≈ pdf(generic_D, u[2]) atol=2e-6
    @test quantile(specialized_D, 0.6) ≈ quantile(generic_D, 0.6) atol=2e-6
    @test rosenblatt(specialized, u) ≈ rosenblatt(generic, u)
end
@testset "public tail registry is exhaustive" begin
    public_families = Set(getfield(Copulas, symbol) for symbol in public_symbols()
        if getfield(Copulas, symbol) isa Type &&
           symbol !== :Tail &&
           getfield(Copulas, symbol) <: Copulas.Tail)
    represented = Set(typeof(tail) for (tail, _) in TAIL_CASES)
    @test all(F -> any(T -> T <: F, represented), public_families)
    @test all(T -> any(F -> T <: F, public_families), represented)
end

@testset "discrete spectral partials follow the active atoms" begin
    # Away from a spectral kink, each atom contributes the coefficient of its
    # unique maximizing coordinate to the corresponding first derivative. The
    # STDF is locally linear, hence every mixed derivative of order >= 2 is 0.
    # This is the independent oracle for the non-smooth routes intentionally
    # excluded from finite-difference checks elsewhere in this file.
    for (tail, d) in TAIL_CASES
        tail isa Copulas.DiscreteSpectralBackedTail || continue
        B = Copulas._spectral_tail(tail).B
        x = collect(range(0.37, 1.13; length=d))
        winners = [argmax(B[:, k] .* x) for k in axes(B, 2)]
        for i in 1:d
            expected = sum(B[i, k] for k in axes(B, 2) if winners[k] == i)
            @test Copulas.ellpartial(tail, x, (i,)) ≈ expected
        end
        d > 1 && @test Copulas.ellpartial(tail, x, (1, 2)) ≈ 0 atol=1e-12
    end
end

@testset "public extreme-value tail primitives" begin
    operations = (
        stable_tail = (Copulas.ℓ,
                       (tail, d) -> Tuple{typeof(tail),Vector{Float64}}),
        pickands = (Copulas.A,
                    (tail, d) -> Tuple{typeof(tail),NTuple{d,Float64}}),
        partial = (Copulas.ellpartial,
                   (tail, d) -> Tuple{typeof(tail),Vector{Float64},Tuple{Int}}),
    )
    selected_routes = Dict(name => Set(which(f, signature(tail, d))
        for (tail, d) in TAIL_CASES)
        for (name, (f, signature)) in pairs(operations))
    checked_routes = Dict(name => Set{Method}() for name in keys(operations))
    for (tail, d) in TAIL_CASES
        @testset "$(nameof(typeof(tail))) d=$d" begin
            @test tail isa Copulas.Tail
            x = collect(range(0.4, 1.0; length=d))
            @test params(tail) isa NamedTuple
            value = Copulas.ℓ(tail, x)
            @test maximum(x) <= value <= sum(x)
            @test Copulas.ℓ(tail, 2 .* x) ≈ 2value
            ω = Tuple(x ./ sum(x))
            @test Copulas.A(tail, ω) ≈ value / sum(x)
            for i in 1:d
                e = zeros(d)
                e[i] = 1
                @test Copulas.ℓ(tail, e) ≈ 1
            end
            @test Copulas.ellpartial(tail, x, (1,)) isa Real
            @test Copulas.ellpartial(tail, x, Int[]) == value
            @test Copulas.ellpartial(tail, x, [1]) ≈
                  Copulas.ellpartial(tail, x, (1,))
            if !(tail isa Copulas.DiscreteSpectralBackedTail)
                # HR and extremal-t evaluate ℓ through multivariate Gaussian or
                # Student probabilities. A 1e-5 stencil amplifies the numerical
                # CDF error, especially in the mixed second derivative; use the
                # larger finite-difference scale appropriate to that oracle.
                noisy_cdf = tail isa Union{Copulas.HuslerReissTail,Copulas.tEVTail}
                h = noisy_cdf ? 5e-3 : 1e-5
                xplus, xminus = copy(x), copy(x)
                xplus[1] += h
                xminus[1] -= h
                if noisy_cdf
                    # Differentiating an approximate multivariate CDF amplifies
                    # its stable numerical bias. Compare the unscaled increment
                    # with the integral of the derivative instead.
                    width = 0.05
                    left, right = copy(x), copy(x)
                    left[1] -= width
                    right[1] += width
                    integrated, _ = QuadGK.quadgk(x[1] - width, x[1] + width) do t
                        point = copy(x)
                        point[1] = t
                        Copulas.ellpartial(tail, point, (1,))
                    end
                    @test Copulas.ℓ(tail, right) - Copulas.ℓ(tail, left) ≈
                          integrated atol=3e-5 rtol=2e-4
                else
                    finite_first = (Copulas.ℓ(tail, xplus) -
                                    Copulas.ℓ(tail, xminus)) / (2h)
                    @test Copulas.ellpartial(tail, x, (1,)) ≈
                          finite_first atol=2e-4 rtol=2e-4
                end

                if d > 1
                    xpp, xpm, xmp, xmm = copy(x), copy(x), copy(x), copy(x)
                    xpp[1] += h; xpp[2] += h
                    xpm[1] += h; xpm[2] -= h
                    xmp[1] -= h; xmp[2] += h
                    xmm[1] -= h; xmm[2] -= h
                    finite_mixed = (Copulas.ℓ(tail, xpp) - Copulas.ℓ(tail, xpm) -
                                    Copulas.ℓ(tail, xmp) + Copulas.ℓ(tail, xmm)) /
                                   (4h^2)
                    @test Copulas.ellpartial(tail, x, (1, 2)) ≈ finite_mixed atol=5e-4 rtol=5e-4
                end
            end
            for (name, (f, signature)) in pairs(operations)
                push!(checked_routes[name], which(f, signature(tail, d)))
            end
        end
    end
    @test checked_routes == selected_routes
end

const PICKANDS_CASES = Tuple(tail for (tail, d) in TAIL_CASES
                             if d == 2 && tail isa Copulas.BivariatePickandsTail)

function is_pickands_kink(tail, t, h)
    Base.@nospecialize tail
    left = (Copulas.A(tail, t) - Copulas.A(tail, t - h)) / h
    right = (Copulas.A(tail, t + h) - Copulas.A(tail, t)) / h
    return !isapprox(left, right; atol=1e-3, rtol=1e-3)
end

@testset "bivariate Pickands identities" begin
    selected_routes = Dict(
        :A => Set(which(Copulas.A, Tuple{typeof(tail),Float64})
                  for tail in PICKANDS_CASES),
        :dA => Set(which(Copulas.dA, Tuple{typeof(tail),Float64})
                   for tail in PICKANDS_CASES),
        :d²A => Set(which(Copulas.d²A, Tuple{typeof(tail),Float64})
                    for tail in PICKANDS_CASES),
        :combined => Set(which(Copulas._A_dA_d²A,
                               Tuple{typeof(tail),Float64})
                         for tail in PICKANDS_CASES),
    )
    checked_routes = Dict(name => Set{Method}() for name in keys(selected_routes))
    for tail in PICKANDS_CASES
        @test Copulas.A(tail, 0.0) ≈ 1
        @test Copulas.A(tail, 1.0) ≈ 1
        for t in (0.2, 0.5, 0.8)
            a = Copulas.A(tail, t)
            @test max(t, 1 - t) <= a + 10eps(Float64) <= 1 + 10eps(Float64)
            combined = Copulas._A_dA_d²A(tail, t)
            @test combined[1] ≈ a
            @test combined[2] ≈ Copulas.dA(tail, t)
            @test combined[3] ≈ Copulas.d²A(tail, t)
            h = 1e-5
            finite_dA = (Copulas.A(tail, t + h) - Copulas.A(tail, t - h)) / (2h)
            finite_d²A = (Copulas.dA(tail, t + h) - Copulas.dA(tail, t - h)) / (2h)
            # Spectral atoms are legitimate kinks: classical first and second
            # derivatives need not agree with centered finite differences there.
            if !is_pickands_kink(tail, t, h)
                @test Copulas.dA(tail, t) ≈ finite_dA atol=2e-5
                @test Copulas.d²A(tail, t) ≈ finite_d²A atol=2e-4
            end
        end
        push!(checked_routes[:A], which(Copulas.A, Tuple{typeof(tail),Float64}))
        push!(checked_routes[:dA], which(Copulas.dA, Tuple{typeof(tail),Float64}))
        push!(checked_routes[:d²A], which(Copulas.d²A, Tuple{typeof(tail),Float64}))
        push!(checked_routes[:combined],
              which(Copulas._A_dA_d²A, Tuple{typeof(tail),Float64}))
    end
    @test checked_routes == selected_routes
end
