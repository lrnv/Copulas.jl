
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

@testset "Distortion densities agree with their cdf derivatives" begin
    C = FGMCopula(2, 0.4)
    for j in 1:2
        i = 3 - j
        D = @invoke Copulas.DistortionFromCop(C::Copulas.Copula{2}, (j,), (0.4,), i)
        for u in (0.25, 0.65)
            reference = ForwardDiff.derivative(t -> cdf(D, t), u)
            @test isapprox(pdf(D, u), reference; atol=1e-8, rtol=1e-8)
            @test isapprox(cdf(D, quantile(D, u)), u; atol=1e-6, rtol=1e-6)
        end
        @test pdf(D, -0.1) == 0
        @test logpdf(D, 1.1) == -Inf
    end
end

@testset "Plackett distortion closed-form quantile" begin
    for θ in (0.5, 2.0), j in 1:2
        C = PlackettCopula(θ)
        uⱼ = j == 1 ? 0.3 : 0.7
        D = condition(C, (j,), (uⱼ,))
        @test D isa Copulas.PlackettDistortion
        @test isfinite(D.logden)

        for α in (0.1, 0.5, 0.9)
            q = quantile(D, α)
            @test isapprox(cdf(D, q), α; atol=5e-12, rtol=5e-12)
        end
        for u in (0.2, 0.6)
            reference = ForwardDiff.derivative(t -> cdf(D, t), u)
            @test logpdf(D, u) ≈ log(reference) atol = 2e-11
        end
        @test quantile(D, 0.0) == 0.0
        @test quantile(D, 1.0) == 1.0
        @test quantile(D, big"0.37") isa BigFloat
        @test logpdf(D, -0.1) == -Inf
        @test logpdf(D, 1.1) == -Inf
    end

    Dind = Copulas.PlackettDistortion(1.0, Int8(1), 0.4)
    @test quantile(Dind, 0.37) ≈ 0.37
end

@testset "Algebraic Archimedean distortion quantiles" begin
    copulas = (
        FrankCopula(2, -2.0),
        FrankCopula(2, 3.0),
        AMHCopula(2, -0.5),
        AMHCopula(2, 0.5),
    )
    for C in copulas
        D = condition(C, (1,), (0.4,))
        for α in (0.1, 0.5, 0.9)
            q = quantile(D, α)
            generic = @invoke quantile(D::Copulas.Distortion, α::Real)
            @test isapprox(cdf(D, q), α; atol=2e-11, rtol=2e-11)
            @test isapprox(q, generic; atol=2e-8, rtol=2e-8)
        end
        @test quantile(D, big"0.37") isa BigFloat
    end
end

@testset "Gumbel and Log distortion closed-form quantiles" begin
    for θ in (1.001, 1.2, 2.5, 8.0), uⱼ in (0.25, 0.7)
        Dg = condition(GumbelCopula(2, θ), (1,), (uⱼ,))
        Dl = condition(LogCopula(2, θ), (1,), (uⱼ,))
        for α in (0.1, 0.5, 0.9)
            qg = quantile(Dg, α)
            ql = quantile(Dl, α)
            generic = @invoke quantile(Dg::Copulas.Distortion, α::Real)
            @test isapprox(cdf(Dg, qg), α; atol=2e-11, rtol=2e-11)
            @test isapprox(cdf(Dl, ql), α; atol=2e-11, rtol=2e-11)
            @test isapprox(qg, ql; atol=2e-11, rtol=2e-11)
            @test isapprox(qg, generic; atol=2e-8, rtol=2e-8)
        end
    end
end

@testset "Lambert-W Archimedean distortion quantiles" begin
    copulas = (
        InvGaussianCopula(2, 0.01),
        InvGaussianCopula(2, 0.5),
        InvGaussianCopula(2, 2.0),
        BB9Copula(2, 1.0, 0.8),
        BB9Copula(2, 1.001, 0.8),
        BB9Copula(2, 2.5, 0.8),
    )
    for C in copulas
        D = condition(C, (1,), (0.4,))
        for α in (0.1, 0.5, 0.9)
            q = quantile(D, α)
            generic = @invoke quantile(D::Copulas.Distortion, α::Real)
            @test isapprox(cdf(D, q), α; atol=3e-11, rtol=3e-11)
            @test isapprox(q, generic; atol=2e-8, rtol=2e-8)
        end
        @test quantile(D, big"0.37") isa BigFloat
    end
end

@testset "Gumbel-Barnett distortion closed-form quantile" begin
    for θ in (0.01, 0.2, 0.8), uⱼ in (0.3, 0.7)
        D = condition(GumbelBarnettCopula(2, θ), (1,), (uⱼ,))
        for α in (0.1, 0.5, 0.9)
            q = quantile(D, α)
            generic = @invoke quantile(D::Copulas.Distortion, α::Real)
            @test isapprox(cdf(D, q), α; atol=3e-11, rtol=3e-11)
            @test isapprox(q, generic; atol=2e-8, rtol=2e-8)
        end
        @test quantile(D, big"0.37") isa BigFloat
    end
end

@testset "Gaussian distortion log-scale formulas" begin
    D = condition(GaussianCopula([1.0 0.6; 0.6 1.0]), (1,), (0.3,))
    N = Normal()
    for u in (1e-12, 0.2, 0.5, 0.8)
        q = quantile(N, u)
        z = (q - D.μz) / D.σz
        reference = logpdf(N, z) - log(abs(D.σz)) - logpdf(N, q)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 1e-13
        @test logpdf(D, u) ≈ reference atol = 1e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
    @test logpdf(D, -0.1) == -Inf
end

@testset "Student distortion logcdf" begin
    D = condition(TCopula(4, [1.0 0.5; 0.5 1.0]), (1,), (0.3,))
    @test D.Tu isa TDist
    @test D.Tcond isa TDist
    for u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "Elliptical conditioning shares matrix factorizations" begin
    Σ = [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]
    for C in (GaussianCopula(Σ), TCopula(4, Σ))
        conditioned = condition(C, (1,), (0.35,))
        @test length(conditioned.m) == 2
        for (k, i) in enumerate((2, 3)), u in (0.2, 0.7)
            reference = Copulas.DistortionFromCop(C, (1,), (0.35,), i)
            @test cdf(conditioned.m[k], u) ≈ cdf(reference, u) atol = 2e-12
        end
    end
end

@testset "Extreme-value conditioning caches fixed transforms" begin
    DEV = condition(GalambosCopula(2, 2.5), (1,), (0.3,))
    @test DEV.negloguⱼ == -log(DEV.uⱼ)

    DAM = condition(ArchimaxCopula(2, Copulas.FrankGenerator(0.8),
                                  Copulas.HuslerReissTail(0.6)), (1,), (0.3,))
    @test DAM.yⱼ == Copulas.ϕ⁻¹(DAM.gen, DAM.uⱼ)
    @test DAM.invderivⱼ == Copulas.ϕ⁻¹⁽¹⁾(DAM.gen, DAM.uⱼ)
end

@testset "Archimedean distortion logcdf" begin
    distortions = (
        condition(ClaytonCopula(3, 2.0), (1, 2), (0.3, 0.6)),
        condition(FrankCopula(3, 2.0), (1, 2), (0.3, 0.6)),
        condition(GumbelCopula(3, 2.0), (1, 2), (0.3, 0.6)),
    )
    for D in distortions, u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 3e-12
    end
    @test all(logcdf(D, 0.0) == -Inf for D in distortions)
    @test all(logcdf(D, 1.0) == 0.0 for D in distortions)
end

@testset "Flip distortion logcdf" begin
    S = SurvivalCopula(ClaytonCopula(2, 2.0), (2,))
    D = condition(S, (1,), (0.3,))
    @test D isa Copulas.FlipDistortion
    for u in (1e-12, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-12
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "FGM distortion log-scale formulas" begin
    for θ in (-0.8, 0.8), uⱼ in (0.2, 0.7)
        D = condition(FGMCopula(2, θ), (1,), (uⱼ,))
        for u in (1e-12, 0.2, 0.5, 0.8)
            @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-14
        end
        @test logcdf(D, 0.0) == -Inf
        @test logcdf(D, 1.0) == 0.0
        @test logpdf(D, -0.1) == -Inf
        @test logpdf(D, 1.1) == -Inf
    end
end

@testset "Generic ConditionalCopula density" begin
    C = GaussianCopula([
        1.0 0.35 0.20
        0.35 1.0 0.25
        0.20 0.25 1.0
    ])
    js = (3,)
    ujs = (0.4,)
    generic = @invoke Copulas.ConditionalCopula(C::Copulas.Copula{3}, js, ujs)
    Cgeneric = FGMCopula(3, [0.1, 0.2, 0.3, 0.4])
    conditioned = condition(Cgeneric, js, ujs)
    @test conditioned.C isa Copulas.ConditionalCopula
    @test conditioned.m === conditioned.C.distortions
    @test generic.logden == log(generic.den)
    specialized = Copulas.ConditionalCopula(C, js, ujs)

    for u in ([0.25, 0.35], [0.5, 0.5], [0.75, 0.65])
        @test isapprox(logpdf(generic, u), logpdf(specialized, u); atol=1e-8, rtol=1e-8)
        @test isapprox(pdf(generic, u), pdf(specialized, u); atol=1e-8, rtol=1e-8)
    end
    @test pdf(generic, [-0.1, 0.5]) == 0

    Cclayton = ClaytonCopula(3, 2.0)
    generic_big = @invoke Copulas.ConditionalCopula(
        Cclayton::Copulas.Copula{3},
        (3,),
        (big"0.4",),
    )
    value_big = logpdf(generic_big, BigFloat[0.35, 0.65])
    @test value_big isa BigFloat
    @test isfinite(value_big)
end

@testset "Elementary distortions respect their support" begin
    distortions = (
        Copulas.NoDistortion(),
        Copulas.MDistortion(0.4, Int8(2)),
        Copulas.WDistortion(0.4, Int8(2)),
    )
    for D in distortions
        @test cdf(D, -0.2) == 0
        @test cdf(D, 1.2) == 1
        @test pdf(D, -0.2) == 0
        @test pdf(D, 1.2) == 0
        @test logpdf(D, -0.2) == -Inf
        @test logpdf(D, 1.2) == -Inf
    end
end

@testset "Checkerboard distortion supports multiple conditioning dimensions" begin
    C = CheckerboardCopula(randn(rng, 3, 30); pseudo_values=false)
    D = Copulas.DistortionFromCop(C, (1, 2), (0.3, 0.7), 3)

    @test D isa Copulas.HistogramBinDistortion
    @test all(0 .<= cdf.(Ref(D), (0.2, 0.5, 0.8)) .<= 1)
    @test all(pdf.(Ref(D), (0.2, 0.5, 0.8)) .>= 0)
    @test all(0 .<= quantile.(Ref(D), (0.2, 0.5, 0.8)) .<= 1)
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

@testset "condition accepts non-Float64 reals (BigFloat StackOverflow regression)" begin
    # Regression: condition(C, js, uⱼₛ) hardcoded NTuple{p,Float64}. Because
    # _process_tuples calls float. (which keeps BigFloat/Float32 unchanged), such
    # inputs missed the typed method, fell back to the untyped entry point, and
    # recursed forever (StackOverflow). The typed methods now accept
    # NTuple{p,<:Real}; non-Float64 values are converted to Float64 downstream, so
    # the conditioning result matches the Float64-input result (tolerance allows
    # for the fast-vs-generic distortion method difference on the converted path).
    C3 = ClaytonCopula(3, 2.0)
    C4 = ClaytonCopula(4, 2.0)

    # Copula entry, single conditioned dim (p == D-1) → univariate Distortion.
    r1 = condition(C3, (1, 2), (0.3, 0.4))
    b1 = condition(C3, (1, 2), (big"0.3", big"0.4"))   # must not StackOverflow
    @test b1 isa Copulas.Distortion
    for u in (0.1, 0.5, 0.9)
        @test isapprox(cdf(b1, u), cdf(r1, u); atol=1e-6)
    end

    # Copula entry, scalar BigFloat, multi remaining (p == 1 < D-1) → SklarDist.
    r2 = condition(C3, 1, 0.3)
    b2 = condition(C3, 1, big"0.3")
    @test b2 isa SklarDist
    @test isapprox(cdf(b2.C, [0.5, 0.6]), cdf(r2.C, [0.5, 0.6]); atol=1e-6)

    # Copula entry, tuple BigFloat, multi conditioned (p == 2 < D-1).
    r3 = condition(C4, (1, 2), (0.3, 0.4))
    b3 = condition(C4, (1, 2), (big"0.3", big"0.4"))
    @test b3 isa SklarDist
    @test isapprox(cdf(b3.C, [0.5, 0.6]), cdf(r3.C, [0.5, 0.6]); atol=1e-6)

    # SklarDist entry, BigFloat data-scale conditioning value (3-dim → 2-dim cond).
    X = SklarDist(C3, (Normal(), LogNormal(), Exponential()))
    rS = condition(X, (1,), (0.2,))
    bS = condition(X, (1,), (big"0.2",))
    @test bS isa SklarDist
    @test isapprox(cdf(bS, [0.3, 0.5]), cdf(rS, [0.3, 0.5]); atol=1e-6)

    # Float32 also previously recursed; confirm it is accepted too.
    @test condition(C3, (1, 2), (0.3f0, 0.4f0)) isa Copulas.Distortion
end

@testset "conditioning carries the conditioning eltype (BigFloat flows end-to-end)" begin
    # condition() accepts non-Float64 values, AND the conditioning point now
    # survives into the ConditionalCopula/DistortionFromCop (no Float64 downcast),
    # so BigFloat precision flows through to the conditional CDF.
    C = ClaytonCopula(4, 2.0)
    xf = [0.3, 0.5, 0.4, 0.6]; xb = big.(xf)

    # single-conditioned distortion (p = d-1): the conditional marginal of coord 2
    df = condition(C, (1, 3, 4), Tuple(xf[[1, 3, 4]]))
    db = condition(C, (1, 3, 4), Tuple(xb[[1, 3, 4]]))
    @test db isa Copulas.DistortionFromCop
    @test db.den isa BigFloat                 # value type flows INTO the struct
    @test eltype(db.uⱼₛ) === BigFloat
    @test cdf(db, xb[2]) isa BigFloat          # ... and OUT through the conditional CDF
    @test Float64(cdf(db, xb[2])) ≈ cdf(df, xf[2]) atol = 1e-9

    # multi-conditioned ConditionalCopula (p < d-1)
    mb = condition(C, (1, 3), Tuple(xb[[1, 3]]))
    @test mb.C isa Copulas.ConditionalCopula
    @test mb.C.den isa BigFloat
    @test cdf(mb, xb[[2, 4]]) isa BigFloat
    @test Float64(cdf(mb, xb[[2, 4]])) ≈
          cdf(condition(C, (1, 3), Tuple(xf[[1, 3]])), xf[[2, 4]]) atol = 1e-9
end
