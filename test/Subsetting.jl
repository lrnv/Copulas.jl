# Full-permutation `subsetdims` (p == d). Reordering *all* coordinates with a
# non-identity permutation of `1:d` previously threw `@assert p < d`; it now
# returns the correctly-reordered copula. (The identity `dims == 1:d` still
# short-circuits to the original copula, and `p == 1` to a `Uniform`.)
@testset "subsetdims full permutation (p == d)" begin

    # Ground truth: cdf(subsetdims(C, perm), u) == cdf(C, v) with v[perm[i]] = u[i].
    permuted_point(perm, u) = (v = similar(u); for (i, j) in enumerate(perm); v[j] = u[i]; end; v)

    @testset "regression: p == d no longer throws" begin
        @test Copulas.subsetdims(ClaytonCopula(3, 2.0), (2, 3, 1)) isa Copulas.Copula
        @test Copulas.subsetdims(GaussianCopula([1.0 0.5 0.2; 0.5 1.0 0.3; 0.2 0.3 1.0]), (3, 1, 2)) isa Copulas.Copula
    end

    @testset "Archimedean (exchangeable) agrees with the parent" begin
        for C in (ClaytonCopula(3, 2.0), FrankCopula(4, 3.0))
            d = length(C); perm = ntuple(i -> mod1(i + 1, d), d)        # cyclic shift (non-identity)
            S = Copulas.subsetdims(C, perm)
            for _ in 1:5
                u = rand(rng, d)
                @test cdf(S, u)    ≈ cdf(C, permuted_point(perm, u))    atol = 1e-8
                @test logpdf(S, u) ≈ logpdf(C, permuted_point(perm, u)) atol = 1e-8
            end
        end
    end

    @testset "Gaussian (asymmetric Σ — permutation is non-trivial)" begin
        Σ = [1.0 0.6 0.2; 0.6 1.0 0.5; 0.2 0.5 1.0]
        C = GaussianCopula(Σ); perm = (2, 3, 1)
        S = Copulas.subsetdims(C, perm)
        @test S.Σ ≈ Σ[collect(perm), collect(perm)]                     # the reordered correlation matrix
        for _ in 1:5
            u = rand(rng, 3)
            @test logpdf(S, u) ≈ logpdf(C, permuted_point(perm, u)) atol = 1e-8
            @test cdf(S, u)    ≈ cdf(C, permuted_point(perm, u))    atol = 1e-2   # MvNormalCDF is Monte-Carlo
        end
    end
end
