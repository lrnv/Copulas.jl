const COPULA_TEST_RESAMPLES = parse(Int, get(ENV, "COPULAS_TEST_RESAMPLES", "19"))
const COPULA_TEST_TINY_RESAMPLES = min(COPULA_TEST_RESAMPLES, 9)

struct MockHypothesis <: CopulaHypothesis end

Copulas.testname(::MockHypothesis) = "Mock copula hypothesis test"
Copulas.nullhypothesis(::MockHypothesis) = "The mock null hypothesis holds."
Copulas._available_statistics(::MockHypothesis) = (:mean, :sum)
Copulas._available_calibrations(::MockHypothesis, ::Val{:mean}) = (:simulation,)
Copulas._available_calibrations(::MockHypothesis, ::Val{:sum}) = (:simulation,)
Copulas._teststatistic(::MockHypothesis, ::Val{:mean}, U::AbstractMatrix; kwargs...) =
    sum(U) / length(U)
Copulas._teststatistic(::MockHypothesis, ::Val{:sum}, U::AbstractMatrix; kwargs...) = sum(U)

function Copulas._simulation_sample(::MockHypothesis, U::AbstractMatrix, rng::Distributions.AbstractRNG)
    sample = similar(U)
    Random.rand!(rng, sample)
    return sample
end

@testset "Copula hypothesis tests [copula_tests]" begin
    @testset "Extensible framework" begin
        U = rand(Xoshiro(123), 2, 40)
        test = CopulaTest(MockHypothesis(), U; N=COPULA_TEST_TINY_RESAMPLES, rng=Xoshiro(1))

        @test test isa CopulaTest{MockHypothesis}
        @test Copulas.default_statistic(MockHypothesis()) === :mean
        @test Copulas.default_calibration(MockHypothesis(), Val(:mean)) === :simulation
        @test Copulas.testname(test) == "Mock copula hypothesis test"
        @test Copulas.nullhypothesis(test) == "The mock null hypothesis holds."
        @test test.statistic === :mean
        @test test.calibration === :simulation
        @test StatsBase.nobs(test) == 40
        @test isfinite(teststatistic(test))
        @test 0 < pvalue(test) < 1

        io = IOBuffer()
        show(io, MIME("text/plain"), test)
        printed = String(take!(io))
        @test occursin("Mock copula hypothesis test", printed)
        @test occursin("The mock null hypothesis holds.", printed)

        other = CopulaTest(MockHypothesis(), U; statistic=:sum, N=COPULA_TEST_TINY_RESAMPLES, rng=Xoshiro(1))
        @test other isa CopulaTest{MockHypothesis}
        @test other.statistic === :sum
        @test other.calibration === :simulation
        @test isfinite(teststatistic(other))
        @test IndependenceCopulaTest(U; N=2, rng=Xoshiro(1)).statistic === :cvm

        stat_err = try
            CopulaTest(MockHypothesis(), U; statistic=:missing, N=2)
        catch err
            err
        end
        @test stat_err isa ArgumentError
        @test occursin("Statistic :missing", sprint(showerror, stat_err))
        @test occursin("Available statistics: :mean, :sum", sprint(showerror, stat_err))

        cal_err = try
            CopulaTest(MockHypothesis(), U; calibration=:multiplier, N=2)
        catch err
            err
        end
        @test cal_err isa ArgumentError
        @test occursin("Calibration :multiplier", sprint(showerror, cal_err))
        @test occursin("Available calibrations: :simulation", sprint(showerror, cal_err))

        @test !(:testname in names(Copulas))
    end

    @testset "Tied margins are rejected" begin
        Utied = [
            0.1 0.1 0.4 0.7
            0.2 0.3 0.6 0.8
        ]

        err = try
            IndependenceCopulaTest(Utied; N=2, rng=Xoshiro(10),)
        catch e
            e
        end

        @test err isa ArgumentError
        @test occursin("tie-free margins", sprint(showerror, err))
        @test occursin("margin 1", sprint(showerror, err))

        # Supplying pseudo-observations must not bypass the tie check.
        @test_throws ArgumentError IndependenceCopulaTest(Utied; pseudo_values=true, N=2, rng=Xoshiro(11),)
    end

    @testset "IndependenceCopulaTest" begin
        U0 = rand(Xoshiro(123), IndependentCopula(2), 80)
        t0 = IndependenceCopulaTest(U0; N=COPULA_TEST_RESAMPLES, rng=Xoshiro(1))

        @test t0 isa IndependenceCopulaTest
        @test t0.statistic === :cvm
        @test t0.calibration === :simulation
        @test Copulas.testname(t0) == "Copula independence test"
        @test StatsBase.nobs(t0) == 80
        @test t0.dimension == 2
        @test isfinite(teststatistic(t0))
        @test 0 < pvalue(t0) < 1

        U1 = rand(Xoshiro(456), ClaytonCopula(2, 8.0), 80)
        t1 = IndependenceCopulaTest(U1; N=COPULA_TEST_RESAMPLES, rng=Xoshiro(1))

        @test teststatistic(t1) > teststatistic(t0)
        @test pvalue(t1) <= 0.05

        io = IOBuffer()
        show(io, MIME("text/plain"), t1)
        printed = String(take!(io))
        @test occursin("Statistic:", printed)
        @test occursin("Observed value:", printed)

        @test_throws ArgumentError IndependenceCopulaTest(U0; statistic=:ks, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError IndependenceCopulaTest(U0; calibration=:bootstrap, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError IndependenceCopulaTest(U0; N=0, rng=Xoshiro(1))
    end

    @testset "ExchangeabilityCopulaTest" begin
        U2 = rand(Xoshiro(123), ClaytonCopula(2, 3.0), 80)
        t2 = ExchangeabilityCopulaTest(U2; N=COPULA_TEST_TINY_RESAMPLES, rng=Xoshiro(1))

        @test t2 isa ExchangeabilityCopulaTest
        @test t2.statistic === :Sn
        @test t2.calibration === :multiplier
        @test t2.details.permutations === :G2
        @test t2.details.generator == ((2, 1),)
        @test t2.details.weight === :wm2
        @test StatsBase.nobs(t2) == 80
        @test t2.dimension == 2
        @test Copulas.testname(t2) == "Copula exchangeability test"
        @test isfinite(teststatistic(t2))
        @test 0 <= pvalue(t2) <= 1

        tall = ExchangeabilityCopulaTest(U2; permutations=:all, N=COPULA_TEST_TINY_RESAMPLES, rng=Xoshiro(1))
        @test tall.details.generator == ((2, 1),)

        Ue = rand(Xoshiro(234), ClaytonCopula(3, 3.0), 120)
        te = ExchangeabilityCopulaTest(Ue; N=COPULA_TEST_RESAMPLES, rng=Xoshiro(1))
        @test te.details.generator == ((2, 1, 3), (2, 3, 1))
        @test pvalue(te) > 0.05

        tc = ExchangeabilityCopulaTest(Ue; permutations=(2, 1, 3), N=COPULA_TEST_TINY_RESAMPLES, rng=Xoshiro(1))
        @test tc.details.generator == ((2, 1, 3),)

        x = rand(Xoshiro(2), 120)
        y = x .+ 0.04 .* randn(Xoshiro(3), 120)
        z = rand(Xoshiro(4), 120)
        Ua = permutedims(hcat(x, y, z))
        ta = ExchangeabilityCopulaTest(Ua; N=COPULA_TEST_RESAMPLES, rng=Xoshiro(1))
        @test teststatistic(ta) > teststatistic(te)
        @test pvalue(ta) <= 0.05

        io = IOBuffer()
        show(io, MIME("text/plain"), ta)
        printed = String(take!(io))
        @test occursin("Permutations:", printed)
        @test occursin("Weight:", printed)

        @test_throws ArgumentError ExchangeabilityCopulaTest(U2; statistic=:Rn, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError ExchangeabilityCopulaTest(U2; calibration=:randomization, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError ExchangeabilityCopulaTest(U2; weight=:wm, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError ExchangeabilityCopulaTest(U2; permutations=(1, 1), N=9, rng=Xoshiro(1))
        @test_throws ArgumentError ExchangeabilityCopulaTest(U2; permutations=(1, 2), N=9, rng=Xoshiro(1))
        @test_throws ArgumentError ExchangeabilityCopulaTest(U2; N=0, rng=Xoshiro(1))
    end

    @testset "RadialSymmetryCopulaTest" begin
        Us = rand(Xoshiro(123), GaussianCopula(3, 0.5), 100)
        ts = RadialSymmetryCopulaTest(Us; N=COPULA_TEST_RESAMPLES, rng=Xoshiro(1))

        @test ts isa RadialSymmetryCopulaTest
        @test ts.statistic === :Sn
        @test ts.calibration === :randomization
        @test ts.details.reflection_probability == 0.5
        @test StatsBase.nobs(ts) == 100
        @test ts.dimension == 3
        @test Copulas.testname(ts) == "Copula radial symmetry test"
        @test isfinite(teststatistic(ts))
        @test 0 < pvalue(ts) < 1

        Ua = rand(Xoshiro(456), ClaytonCopula(3, 4.0), 100)
        ta = RadialSymmetryCopulaTest(Ua; N=COPULA_TEST_RESAMPLES, rng=Xoshiro(1))

        @test teststatistic(ta) > teststatistic(ts)

        io = IOBuffer()
        show(io, MIME("text/plain"), ta)
        printed = String(take!(io))
        @test occursin("Reflection probability:", printed)
        @test occursin("The copula is radially symmetric.", printed)

        @test_throws ArgumentError RadialSymmetryCopulaTest(Us; statistic=:cvm, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError RadialSymmetryCopulaTest(Us; calibration=:multiplier, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError RadialSymmetryCopulaTest(Us; N=0, rng=Xoshiro(1))
    end

    @testset "ExtremeValueCopulaTest" begin
        Uev = rand(Xoshiro(123), GumbelCopula(3, 3.0), 100)
        tev = ExtremeValueCopulaTest(Uev; N=COPULA_TEST_RESAMPLES, rng=Xoshiro(1))

        @test tev isa ExtremeValueCopulaTest
        @test tev.statistic === :Sn
        @test tev.calibration === :multiplier
        @test tev.details.powers == (3.0, 4.0, 5.0)
        @test tev.details.multiplier === :exponential
        @test tev.details.derivative_bandwidth == inv(sqrt(100))
        @test StatsBase.nobs(tev) == 100
        @test tev.dimension == 3
        @test Copulas.testname(tev) == "Extreme-value copula test"
        @test isfinite(teststatistic(tev))
        @test 0 < pvalue(tev) < 1

        Ucl = rand(Xoshiro(456), ClaytonCopula(3, 3.0), 100)
        tcl = ExtremeValueCopulaTest(Ucl; N=COPULA_TEST_RESAMPLES, rng=Xoshiro(1))

        @test teststatistic(tcl) > teststatistic(tev)
        @test pvalue(tcl) <= 0.05

        tp = ExtremeValueCopulaTest(Uev; powers=2, N=COPULA_TEST_TINY_RESAMPLES, rng=Xoshiro(1))
        @test tp.details.powers == (2.0,)

        io = IOBuffer()
        show(io, MIME("text/plain"), tcl)
        printed = String(take!(io))
        @test occursin("Powers:", printed)
        @test occursin("The copula belongs to the extreme-value class.", printed)

        @test_throws ArgumentError ExtremeValueCopulaTest(Uev; statistic=:cvm, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError ExtremeValueCopulaTest(Uev; calibration=:simulation, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError ExtremeValueCopulaTest(Uev; powers=1, N=9, rng=Xoshiro(1))
        @test_throws ArgumentError ExtremeValueCopulaTest(Uev; powers=(), N=9, rng=Xoshiro(1))
        @test_throws ArgumentError ExtremeValueCopulaTest(Uev; N=0, rng=Xoshiro(1))
    end

    @testset "GOFCopulaTest" begin
        @testset "Published Sn normalization" begin
            Udet = [
                0.2 0.5 0.8
                0.3 0.6 0.9
            ]
            Cdet = IndependentCopula(2)

            expected = 647 / 2250

            @test Copulas._gof_sn_statistic(Udet, Cdet) ≈ expected
            Tdet = GOFCopulaTest(Cdet, Udet; pseudo_values=true, N=1, rng=Xoshiro(777),)
            @test teststatistic(Tdet) ≈ expected
        end

        U = rand(Xoshiro(123), ClaytonCopula(2, 3.0), 60)
        Ts = GOFCopulaTest(ClaytonCopula(2, 3.0), U;
            N=COPULA_TEST_TINY_RESAMPLES, rng=Xoshiro(1))

        @test Ts isa GOFCopulaTest
        @test Ts.hypothesis.kind === :simple
        @test Ts.statistic === :Sn
        @test Ts.calibration === :parametric_bootstrap
        @test StatsBase.nobs(Ts) == 60
        @test Ts.dimension == 2
        @test Copulas.testname(Ts) == "Copula goodness-of-fit test"
        @test isfinite(teststatistic(Ts))
        @test 0 < pvalue(Ts) < 1

        M = fit(CopulaModel, ClaytonCopula, U; vcov=false)
        Tc = GOFCopulaTest(M; N=COPULA_TEST_TINY_RESAMPLES, rng=Xoshiro(1))
        @test Tc.hypothesis.kind === :composite
        @test Tc.hypothesis.model === M
        @test 0 < pvalue(Tc) < 1

        @testset "Composite GOF preserves runtime fitting structure" begin
            # NestedArchimedeanCopula stores its tree structure and inner
            # generator families in the instance, so refitting from typeof(C)
            # is insufficient.
            Cnested = NestedArchimedeanCopula(Copulas.ClaytonGenerator(1.5); leaves=[1], children=[ClaytonCopula(2, 3.0) => [2, 3]],)
            Unested = rand(Xoshiro(901), Cnested, 30)

            Mnested = fit(CopulaModel, Cnested, Unested; vcov=false, derived_measures=false,)

            Mnested_refit = Copulas._gof_refit(Mnested, Unested)
            Cnested_refit = Copulas._copula_of(Mnested_refit)

            @test Cnested_refit isa NestedArchimedeanCopula
            @test Cnested_refit.leafdims == Cnested.leafdims
            @test length(Cnested_refit.children) == length(Cnested.children)
            @test Cnested_refit.children[1][2] == Cnested.children[1][2]
            @test typeof(Cnested_refit.G).name.wrapper === typeof(Cnested.G).name.wrapper
            @test typeof(Cnested_refit.children[1][1].G).name.wrapper === typeof(Cnested.children[1][1].G).name.wrapper

            # SurvivalCopula stores the flip pattern in the instance rather
            # than in its concrete type.
            Csurvival = SurvivalCopula(ClaytonCopula(3, 2.5), (1, 3))
            Usurvival = rand(Xoshiro(902), Csurvival, 40)

            Msurvival = fit(CopulaModel, typeof(Csurvival), Usurvival; method=:itau, flips=Csurvival.flipmask, vcov=false, derived_measures=false,)

            Msurvival_refit = Copulas._gof_refit(Msurvival, Usurvival)
            Csurvival_refit = Copulas._copula_of(Msurvival_refit)

            @test Csurvival_refit isa SurvivalCopula
            @test Csurvival_refit.flipmask == Csurvival.flipmask
            @test Csurvival_refit.flipmask == (true, false, true)
        end

        io = IOBuffer()
        show(io, MIME("text/plain"), Tc)
        printed = String(take!(io))
        @test occursin("Hypothesis:", printed)
        @test occursin("Fitted model:", printed)

        @test_throws ArgumentError GOFCopulaTest(ClaytonCopula(2, 3.0), U; statistic=:ks, N=2)
        @test_throws ArgumentError GOFCopulaTest(M, U; calibration=:multiplier, N=2)
        @test_throws ArgumentError GOFCopulaTest(M, U; N=0)
        @test_throws DimensionMismatch GOFCopulaTest(ClaytonCopula(3, 3.0), U; N=2)
    end
end
