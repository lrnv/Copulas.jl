using Copulas
using Distributions
using LinearAlgebra
using PartitionedDistributions
using Statistics
using Test


@testset "PartitionedDistributions extension" begin

    @test Base.get_extension(
        Copulas,
        :CopulasPartitionedDistributionsExt,
    ) !== nothing


    @testset "PartitionedDistributions API on Copula" begin
        C = GaussianCopula{3}(0.35)
        u = [0.2, 0.4, 0.7]

        # marginal -> subsetdims
        P13 = PartitionedDistributions.marginal(
            C,
            [1, 3],
        )

        C13 = subsetdims(
            C,
            (1, 3),
        )

        @test logpdf(P13, u[[1, 3]]) ≈
              logpdf(C13, u[[1, 3]])

        # Scalar selections retain PartitionedDistributions semantics.
        @test PartitionedDistributions.marginal(
            C,
            2,
        ) isa Uniform

        # Reordered marginal.
        P31 = PartitionedDistributions.marginal(
            C,
            [3, 1],
        )

        C31 = subsetdims(
            C,
            (3, 1),
        )

        @test logpdf(P31, u[[3, 1]]) ≈
              logpdf(C31, u[[3, 1]])

        # Keep coordinates 1 and 3 => condition on coordinate 2.
        Pcond13 = PartitionedDistributions.conditional(
            C,
            u,
            [1, 3],
        )

        Ccond13 = condition(
            C,
            2,
            u[2],
        )

        @test logpdf(Pcond13, u[[1, 3]]) ≈
              logpdf(Ccond13, u[[1, 3]])

        # PartitionedDistributions also allows the kept coordinates
        # to be reordered.
        Pcond31 = PartitionedDistributions.conditional(
            C,
            u,
            [3, 1],
        )

        Ccond31 = subsetdims(
            Ccond13,
            (2, 1),
        )

        @test logpdf(Pcond31, u[[3, 1]]) ≈
              logpdf(Ccond31, u[[3, 1]])

        # Scalar conditional.
        Pcond1 = PartitionedDistributions.conditional(
            C,
            u,
            1,
        )

        Ccond1 = condition(
            C,
            (2, 3),
            (u[2], u[3]),
        )

        @test logpdf(Pcond1, u[1]) ≈
              logpdf(Ccond1, u[1])
    end


    @testset "PartitionedDistributions API on SklarDist" begin
        C = GaussianCopula{3}(0.35)

        S = SklarDist(
            C,
            (
                Normal(0.0, 1.0),
                LogNormal(0.1, 0.5),
                Gamma(2.0, 1.0),
            ),
        )

        x = [0.2, 1.1, 2.0]

        P13 = PartitionedDistributions.marginal(
            S,
            [1, 3],
        )

        S13 = subsetdims(
            S,
            (1, 3),
        )

        @test logpdf(P13, x[[1, 3]]) ≈
              logpdf(S13, x[[1, 3]])

        Pcond13 = PartitionedDistributions.conditional(
            S,
            x,
            [1, 3],
        )

        Scond13 = condition(
            S,
            2,
            x[2],
        )

        @test logpdf(Pcond13, x[[1, 3]]) ≈
              logpdf(Scond13, x[[1, 3]])
    end


    @testset "Copulas API on PartitionedDistributions distributions" begin
        μ = [0.2, -0.3, 0.7]

        Σ = [
            1.0  0.3  0.1
            0.3  1.2  0.25
            0.1  0.25 0.8
        ]

        D = MvNormal(
            μ,
            Σ,
        )

        x = [0.1, -0.4, 1.1]

        # subsetdims -> marginal
        ours13 = subsetdims(
            D,
            (1, 3),
        )

        theirs13 = PartitionedDistributions.marginal(
            D,
            [1, 3],
        )

        @test mean(ours13) ≈ mean(theirs13)
        @test cov(ours13) ≈ cov(theirs13)

        # Copulas semantics collapse a one-element subset to
        # a univariate distribution.
        ours2 = subsetdims(
            D,
            (2,),
        )

        theirs2 = PartitionedDistributions.marginal(
            D,
            2,
        )

        @test mean(ours2) ≈ mean(theirs2)
        @test std(ours2) ≈ std(theirs2)

        # condition on coordinate 2, keep 1 and 3.
        ours_cond13 = condition(
            D,
            2,
            x[2],
        )

        theirs_cond13 = PartitionedDistributions.conditional(
            D,
            x,
            [1, 3],
        )

        @test mean(ours_cond13) ≈ mean(theirs_cond13)
        @test cov(ours_cond13) ≈ cov(theirs_cond13)

        # condition on coordinates 2 and 3, keep coordinate 1.
        ours_cond1 = condition(
            D,
            (2, 3),
            (x[2], x[3]),
        )

        theirs_cond1 = PartitionedDistributions.conditional(
            D,
            x,
            1,
        )

        @test mean(ours_cond1) ≈ mean(theirs_cond1)
        @test std(ours_cond1) ≈ std(theirs_cond1)

        # Sequential transforms are inherited from the same marginal and
        # conditional interface. For a multivariate Gaussian, the independent
        # oracle is the standardized lower-Cholesky representation.
        L = cholesky(Σ).L
        expected = cdf.(Normal(), L \ (x - μ))
        transformed = rosenblatt(D, x)

        @test transformed ≈ expected
        @test inverse_rosenblatt(D, transformed) ≈ x

        X = [x (μ .+ [0.4, -0.2, 0.3])]
        expected_matrix = cdf.(Normal(), L \ (X .- μ))
        transformed_matrix = rosenblatt(D, X)

        @test transformed_matrix ≈ expected_matrix
        @test inverse_rosenblatt(D, transformed_matrix) ≈ X

        @test_throws DimensionMismatch rosenblatt(D, x[1:2])
        @test_throws DimensionMismatch inverse_rosenblatt(D, transformed[1:2])
    end


    @testset "Rosenblatt transforms of a multivariate Student distribution" begin
        D = MvTDist(
            5.0,
            [0.2, -0.3, 0.7],
            [
                1.0 0.3  0.1
                0.3 1.2  0.25
                0.1 0.25 0.8
            ],
        )
        x = [0.1, -0.4, 1.1]

        transformed = rosenblatt(D, x)
        @test all(0 .<= transformed .<= 1)
        @test inverse_rosenblatt(D, transformed) ≈ x

        X = [x [0.4, -0.1, 0.5]]
        transformed_matrix = rosenblatt(D, X)
        @test transformed_matrix[:, 1] ≈ transformed
        @test inverse_rosenblatt(D, transformed_matrix) ≈ X

        @test_throws DimensionMismatch rosenblatt(D, x[1:2])
        @test_throws DimensionMismatch inverse_rosenblatt(D, transformed[1:2])
    end


    @testset "positive-support completion" begin
        D = MvLogNormal(MvNormal([0.1, 0.2], [1.0 0.3; 0.3 1.0]))
        x = [1.2, 0.8]

        # The retained-coordinate placeholder must come from the positive
        # marginal support rather than being an invalid zero.
        @test subsetdims(D, (1,)) isa LogNormal
        @test condition(D, 1, x[1]) isa LogNormal

        transformed = rosenblatt(D, x)
        @test all(0 .<= transformed .<= 1)
        @test inverse_rosenblatt(D, transformed) ≈ x
    end


    @testset "pointwise conditional logpdfs" begin
        C = GaussianCopula{3}(0.35)
        u = [0.2, 0.4, 0.7]

        observed = PartitionedDistributions.pointwise_conditional_logpdfs(
            C,
            u,
        )

        expected = map(1:3) do i
            js = Tuple(
                j for j in 1:3
                if j != i
            )

            ujs = Tuple(
                u[j] for j in js
            )

            logpdf(
                condition(C, js, ujs),
                u[i],
            )
        end

        @test observed ≈ expected
    end

end
