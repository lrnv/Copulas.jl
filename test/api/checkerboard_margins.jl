@testset "CheckerboardCopula validates completed marginal masses" begin
    concentrated = [0.1 0.1; 0.1 0.1]
    @test_throws ArgumentError CheckerboardCopula(concentrated; m=2)
    @test_throws ArgumentError fit(CheckerboardCopula, concentrated; m=2)
    error = try
        CheckerboardCopula(concentrated; m=2)
    catch err
        err
    end
    @test occursin("margin 1", sprint(showerror, error))
    @test occursin("pseudos(X)", sprint(showerror, error))
    @test occursin("pseudo_values=false", sprint(showerror, error))

    # The first margin is valid, but the second is not; missing bins must
    # contribute zero to the marginal sums rather than escape validation.
    @test_throws ArgumentError CheckerboardCopula([0.1 0.7; 0.1 0.2]; m=2)
    @test_throws ArgumentError CheckerboardCopula(
        [0.1 0.2 0.3 0.7; 0.1 0.2 0.7 0.8]; m=2)
    for bad in (-0.1, 1.1, NaN, Inf)
        @test_throws DomainError CheckerboardCopula([bad 0.8; 0.2 0.7]; m=2)
    end
    @test_throws ArgumentError CheckerboardCopula(zeros(2, 0); m=2)

    raw = [10 20 30 40 50 60; 60 20 40 10 50 30; 30 10 60 20 40 50]
    for T in (Float32, Float64), resolution in (2, (2, 3, 6))
        U = T.(pseudos(raw))
        C = CheckerboardCopula(U; m=resolution)
        C_raw = CheckerboardCopula(raw; m=resolution, pseudo_values=false)
        @test C.boxes == C_raw.boxes
        for row in 1:3, u in (0.0, 0.13, 0.5, 0.87, 1.0)
            point = ones(3)
            point[row] = u
            @test cdf(C, point) ≈ u atol=1e-14
        end
    end
    # Tied pseudo-values are allowed when the selected coarser grid still
    # has uniform margins. Validity is a property of bin masses, not ranks.
    tied = CheckerboardCopula([0.1 0.1 0.8 0.8; 0.2 0.2 0.7 0.7]; m=2)
    @test cdf(tied, [0.5, 1.0]) ≈ 0.5
    endpoints = CheckerboardCopula([0.0 1.0; 1.0 0.0]; m=2)
    @test cdf(endpoints, [0.5, 1.0]) ≈ 0.5
    # Summation roundoff must not reject a valid finer rank grid.
    fine = CheckerboardCopula(pseudos(raw); m=6)
    @test cdf(fine, [0.31, 1.0, 1.0]) ≈ 0.31
    fitted = fit(CheckerboardCopula, pseudos(raw); m=6)
    @test cdf(fitted, [0.31, 1.0, 1.0]) ≈ 0.31
end
