
@testset "IndependentCopula conditional"  begin
    # [GenericTests integration]: Yes. This checks condition(X,J,·) reduces to subsetdims for independence; can be generalized and added to GenericTests.
    X = SklarDist(IndependentCopula(3), (Normal(), Exponential(), LogNormal()))
    Y = condition(X, 2, 0.7)
    Z = Copulas.subsetdims(X, (1,3))

    @test length(Y) == 2
    @test Y isa SklarDist
    @test Y.C isa IndependentCopula{2}
    @test Y.m[1] == Normal()
    @test Y.m[2] == LogNormal()

    @test length(Z) == 2
    @test Z isa SklarDist
    @test Z.C isa IndependentCopula{2}
    @test Z.m[1] == Normal()
    @test Z.m[2] == LogNormal()
end

@testset "Generic Distortion vs AD (bivariate small subset)" begin
    # Compare the GENERIC DistortionFromCop (forced via @invoke) against AD-based reference
    # on a tiny, fast subset to validate the generic path independent of family specifics.
    examples = (
        FGMCopula(2, 0.4),
        ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.AsymGalambosTail(0.35, 0.65, 0.3))
    )
    us = (0.2, 0.5, 0.8)
    for C in examples
        # j = conditioned index, i = remaining index
        for j in 1:2
            i = 3 - j
            for v in (0.3, 0.7)
                # Force the generic DistortionFromCop
                Dgen = @invoke Copulas.DistortionFromCop(C::Copulas.Copula{2}, (j,), (v,), i)
                vals_gen = cdf.(Ref(Dgen), us)

                refs = similar(collect(us))
                if j == 1
                    # condition on first coordinate, vary derivative w.r.t u1
                    # numerator at (u1=v, u2=u), denominator at (u1=v, u2≈1)
                    for (k, u) in pairs(us)
                        refs[k] = ForwardDiff.derivative(w -> cdf(C, [w, u]), v)
                    end
                else
                    # j == 2: derivative w.r.t u2; points (u1=u, u2=v) and (u1≈1, u2=v)
                    for (k, u) in pairs(us)
                        refs[k] = ForwardDiff.derivative(t -> cdf(C, [u, t]), v)
                    end
                end

                for (vg, r) in zip(vals_gen, refs)
                    @test isfinite(r) && 0.0 <= r <= 1.0
                    @test isapprox(vg, r; atol=1e-3, rtol=1e-3)
                end
            end
        end
    end
end

@testset "Generic ConditionalCopula vs AD (3D, p=1)" begin
    # Validate the GENERIC ConditionalCopula cdf against an AD-based reference
    # on a tiny 3D subset for two representative families.
    examples = (
        FrankCopula(3, 2.7),
        ClaytonCopula(3, 1.2),
    )
    pts = ((0.2, 0.3), (0.5, 0.5), (0.8, 0.6))
    for C in examples
        js = (3,)
        for w in (0.25, 0.7)
            # Force the GENERIC equivalent to conditioning: 
            CC      = @invoke Copulas.ConditionalCopula(C::Copulas.Copula{3}, js, (w,))
            margin1 = @invoke Copulas.DistortionFromCop(C::Copulas.Copula{3}, js, (w,), 1)
            margin2 = @invoke Copulas.DistortionFromCop(C::Copulas.Copula{3}, js, (w,), 2)
            CondObj = SklarDist(CC, (margin1, margin2))
            for (u1, u2) in pts
                val_fast = cdf(CondObj, [u1, u2])
                # AD reference: ratio of partial derivatives w.r.t. u3 at (u1,u2,w) vs (≈1,≈1,w)
                val_ref = ForwardDiff.derivative(t -> cdf(C, [u1, u2, t]), w)
                @test isfinite(val_ref) && 0.0 <= val_ref <= 1.0
                @test isapprox(val_fast, val_ref; atol=5e-4, rtol=5e-4)
            end
        end
    end
end

@testset "Independent univariate conditional cases"  begin
    # [GenericTests integration]: Yes. Univariate conditional on independent copula should be Uniform; Sklar with independent copula preserves marginal.
    # Suitable for a generic conditional smoke test.
    # Uniform-scale: Independent copula -> Uniform when one dim remains
    C = IndependentCopula(2)
    J = (1,)
    u1 = 0.3
    Ucond = condition(C, J, (u1,))
    @test Ucond isa Distributions.Uniform
    @test cdf(Ucond, 0.1) ≈ 0.1
    @test cdf(Ucond, 0.9) ≈ 0.9
    # Original-scale: Sklar with independent copula -> marginal unaffected
    X = SklarDist(C, (Normal(), Exponential()))
    Y = condition(X, J, (0.0,))  # conditioning value irrelevant for independence
    @test Y isa Distributions.UnivariateDistribution
    for t in (-1.0, 0.0, 1.2)
        @test cdf(Y, t) ≈ cdf(Exponential(), t)
    end
end

@testset "GaussianCopula univariate conditional (uniform scale)"  begin
    # [GenericTests integration]: Yes. This is a model-specific formula but fits an "analytic conditional for Gaussian" block in GenericTests.
    ρ = 0.6
    Σ = [1.0 ρ; ρ 1.0]
    C = GaussianCopula(Σ)
    J = (2,)
    u2 = 0.2
    D = condition(C, J, (u2,))
    @test D isa Distributions.ContinuousUnivariateDistribution
    z2 = quantile(Normal(), u2)
    μ = ρ * z2
    σ = sqrt(1 - ρ^2)
    # For u in (0,1), expected H(u|u2) = Φ((Φ^{-1}(u) - μ)/σ)
    for u in (0.1, 0.4, 0.8)
        expected = cdf(Normal(), (quantile(Normal(), u) - μ)/σ)
        @test isapprox(cdf(D, u), expected; atol=1e-3, rtol=1e-3)
    end
    # Quantile-cdf roundtrip
    for α in (1e-6, 1e-3, 0.5, 0.9, 0.999, 1 - 1e-6)
        q = quantile(D, α)
        @test isapprox(cdf(D, q), α; atol=2e-3, rtol=2e-3)
    end
end

@testset "Bivariate Archimedean conditional (generator formula across families)" begin
    # [GenericTests integration]: Yes. We already added a similar Archimedean conditional check using generator identities in GenericTests.
    # Known bivariate Archimedean identity:
    # H(u | v) = ϕ'(ϕ^{-1}(u) + ϕ^{-1}(v)) / ϕ'(ϕ^{-1}(v))
    # Test it across multiple families by looping instead of duplicating code.
    examples = (
        ClaytonCopula(2,1.2),
        FrankCopula(2, 1.0),
        GumbelCopula(2, 1.2),
    )
    J = (2,)
    tol = 5e-5
    for C in examples
        for v in (0.2, 0.5, 0.8)
            D = condition(C, J, (v,))
            inv_v = Copulas.ϕ⁻¹(C.G, v)
            for u in (1e-6, 0.1, 0.4, 0.8, 1 - 1e-6)
                t = Copulas.ϕ⁻¹(C.G, u) + inv_v
                num = Copulas.ϕ⁽¹⁾(C.G, t)
                den = Copulas.ϕ⁽¹⁾(C.G, inv_v)
                expected = num / den
                @test isfinite(expected) && 0.0 <= expected <= 1.0
                @test isapprox(cdf(D, u), expected; atol=tol, rtol=tol)
            end
        end
    end
end

@testset "GaussianCopula conditional copula vs MVN" begin
    # [GenericTests integration]: Maybe. It depends on MvNormalCDF and is moderately heavy; could be a behind-flag exhaustive check.
    Random.seed!(rng,42)
    d = 4
    # build correlation matrix
    A = randn(rng, d, d)
    Σ = A*A'
    # normalize to correlation
    s = sqrt.(diag(Σ))
    Σ = Symmetric(Σ ./ (s*s'))
    C = GaussianCopula(Matrix(Σ))
    # Choose J and uJ
    J = (2,4)
    uJ = (0.3, 0.8)
    CC = condition(C, J, uJ)
    # Compare to MVNormal conditioning on z-scale
    I = Tuple(setdiff(1:d, J))
    dI = length(I)
    Iv = collect(I); Jv = collect(J)
    ΣII = Σ[Iv, Iv]; ΣJJ = Σ[Jv, Jv]; ΣIJ = Σ[Iv, Jv]; ΣJI = Σ[Jv, Iv]
    L = cholesky(ΣJJ)
    zJ = quantile.(Normal(), collect(uJ))
    y = L \ zJ
    μ = ΣIJ * (L' \ y)
    K = L \ ΣJI
    Σcond = ΣII - ΣIJ * (L'\K)
    for _ in 1:3
        uI = rand(rng, dI)./5 .+ 2/5
        zI = quantile.(Normal(), uI)
        p_mvn = MvNormalCDF.mvnormcdf(vec(μ), Matrix(Σcond), fill(-Inf, dI), zI)[1]
        p_cc = cdf(CC, uI)
        @test isapprox(p_cc, p_mvn; atol=5e-3)
    end
end

@testset "Higher-dim Archimedean conditional (3|2 via generator derivatives)" begin
    # [GenericTests integration]: Yes. This extends the Archimedean conditional identity to higher p; can be parameterized and integrated.
    # For Archimedean C(u) = ϕ(Σ ϕ⁻¹(u_i)), conditioning on J with |J|=p gives
    # H_{I|J}(u_I|u_J) = ϕ^{(p)}(Σ_{i∈I} ϕ⁻¹(u_i) + Σ_{j∈J} ϕ⁻¹(u_j)) / ϕ^{(p)}(Σ_{j∈J} ϕ⁻¹(u_j))
    # We'll test in d=5 with |J|=2, so |I|=3.
    families = [
        (ClaytonCopula, 1.1, 1e-5),
        (FrankCopula,   2.0, 1e-5),
        # (GumbelCopula,  1.5, 5e-5),
    ]
    d = 5
    J = (2, 4)
    p = length(J)
    for (Ctor, θ, tol) in families
        C = Ctor(d, θ)
        # a couple of moderate conditioning points away from 0/1 to avoid singularities
        for uJ in ((0.2, 0.7), (0.3, 0.8))
            CC = condition(C, J, uJ)
            # test a few uI points
            for uI in ((0.1, 0.4, 0.8), (0.25, 0.5, 0.75), (0.2, 0.6, 0.9))
                # Compute expected via generator-derivative ratio
                SJ = sum(Copulas.ϕ⁻¹(C.G, v) for v in uJ)
                SI = sum(Copulas.ϕ⁻¹(C.G, u) for u in uI)
                S_full = SJ + SI
                num = Copulas.ϕ⁽ᵏ⁾(C.G, p, S_full)
                den = Copulas.ϕ⁽ᵏ⁾(C.G, p, SJ)
                expected = num / den
                # Evaluate model
                got = cdf(CC, collect(uI))
                @test isfinite(expected) && 0.0 <= expected <= 1.0
                @test isapprox(got, expected; atol=tol, rtol=tol)
            end
        end
    end
end

@testset "Gaussian Sklar conditional vs MVN with normal marginals" begin
# [GenericTests integration]: Yes. This validates SklarDist conditioning against MVN algebra; belongs in GenericTests under conditioning.
    Random.seed!(rng,43)
    d = 3
    Σ = [1 0.7 0.3;0.7 1 0.7; 0.3 0.7 1]
    C = GaussianCopula(Σ)
    μ = zeros(d)
    
    X = SklarDist(C, Tuple(Normal(μ[i],Σ[i,i]) for i in 1:d))
    X_mock = MvNormal(μ, Σ)

    # check that X and X_mock are indeed the same distribution: 
    for _ in 1:5
        t = rand(rng, 3)
        A, r = mvnormcdf(X_mock, fill(-Inf, d), t)
        B = cdf(X, t)
        @test A ≈ B atol=10sqrt(r)
    end

    
    # Now condition using the known gaussian conditionning algebra: 
    xⱼₛ = [0]
    is, js = 2:3, 1:1 
    μ_Y    = μ[is] .+ Σ[is, js] * inv(Σ[js, js]) * (xⱼₛ - μ[js])
    Σ_Y = Σ[is,is] .- Σ[is,js] * inv(Σ[js,js]) * Σ[js, is]
    Y_mock = MvNormal(μ_Y, Σ_Y)

    # And construct the conditioning using the generic paths: 
    J = Tuple(reverse(collect(js)))
    Y = condition(X, J, xⱼₛ)

    for _ in 1:3
        t = randn(rng, 2)
        A, r = mvnormcdf(Y_mock, fill(-Inf, 2), t)
        B = cdf(Y, t)
        @test A ≈ B atol=10sqrt(r)
    end
end

@testset "Generic fallback sanity (Clayton small d)" begin
    # [GenericTests integration]: Partially. Monotonicity and quantile-roundtrip are generic; keep Clayton-specific here or parameterize family list.
    Random.seed!(rng,44)
    d = 2
    C = ClaytonCopula(d, 0.7)
    m = (Normal(), LogNormal())
    X = SklarDist(C, m)
    J = (1,)
    x1 = 0.0
    Y = condition(X, J, (x1,))
    # basic properties
    t = randn(rng)
    v = cdf(Y, t)
    @test 0.0 <= v <= 1.0
    # Monotonicity: cdf should be non-decreasing
    ts = sort!(randn(rng, 50))
    vs = cdf.(Ref(Y), ts)
    @test all(diff(vs) .>= -1e-10)
    # Quantile-cdf roundtrip
    for α in (1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999, 1 - 1e-6)
        q = quantile(Y, α)
        @test isapprox(cdf(Y, q), α; atol=2e-3, rtol=2e-3)
    end
end
