# Correctness obligation: exhaustively covers public EV-tail families and
# verifies stable-tail, Pickands, derivative, and reconstruction identities.
const TAIL_CASES = (
    (Copulas.AsymGalambosTail(1.0, 0.4, 0.6), 2),
    (Copulas.AsymLogTail(1.5, 0.4, 0.6), 2),
    (Copulas.AsymMixedTail(0.3, 0.2), 2),
    (Copulas.BC2Tail(0.5, 0.3), 2),
    (Copulas.CuadrasAugeTail(0.5), 2),
    (Copulas.GalambosTail(1.0), 3),
    (Copulas.HuslerReissTail(1.0), 3),
    (Copulas.LogTail(1.5), 3),
    (Copulas.MixedTail(0.5), 2),
    (Copulas.MOTail(0.2, 0.3, 0.4), 2),
    (Copulas.TawnTail(2.0, [0.6, 0.7, 0.8]), 3),
    (Copulas.tEVTail(4.0, 0.5), 2),
    (EmpiricalEVCopula{2}(_FIXTURE_DATA; method=:cfg, pseudo_values=false).tail, 2),
    (EmpiricalEVCopula{3}(_FIXTURE_DATA3;
        degree=1, pseudo_values=false).tail, 3),
    (DiscreteSpectralTail([0.7 0.3; 0.2 0.8]), 2),
)

@testset "public tail registry is exhaustive" begin
    public_families = Set(getfield(Copulas, symbol) for symbol in PUBLIC_SYMBOLS
        if getfield(Copulas, symbol) isa Type &&
           symbol !== :Tail &&
           getfield(Copulas, symbol) <: Copulas.Tail)
    represented = Set(typeof(tail) for (tail, _) in TAIL_CASES)
    @test all(F -> any(T -> T <: F, represented), public_families)
    @test all(T -> any(F -> T <: F, public_families), represented)
end

@testset "public extreme-value tail primitives" begin
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
                h = 1e-5
                xplus, xminus = copy(x), copy(x)
                xplus[1] += h
                xminus[1] -= h
                finite_first = (Copulas.ℓ(tail, xplus) -
                                Copulas.ℓ(tail, xminus)) / (2h)
                @test Copulas.ellpartial(tail, x, (1,)) ≈ finite_first

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
        end
    end
end

const PICKANDS_CASES = (
    Copulas.AsymGalambosTail(1.0, 0.4, 0.6),
    Copulas.AsymLogTail(1.5, 0.4, 0.6),
    Copulas.AsymMixedTail(0.3, 0.2),
    Copulas.BC2Tail(0.5, 0.3),
    Copulas.CuadrasAugeTail(0.5),
    Copulas.GalambosTail(1.0),
    Copulas.HuslerReissTail(1.0),
    Copulas.LogTail(1.5),
    Copulas.MixedTail(0.5),
    Copulas.MOTail(0.2, 0.3, 0.4),
    Copulas.tEVTail(4.0, 0.5),
    EmpiricalEVCopula{2}(_FIXTURE_DATA; method=:cfg, pseudo_values=false).tail,
)

function is_spectral_kink(tail, t)
    tail isa Copulas.DiscreteSpectralBackedTail || return false
    B = Copulas._spectral_tail(tail).B
    return any(axes(B, 2)) do k
        mass = B[1, k] + B[2, k]
        !iszero(mass) && isapprox(t, B[2, k] / mass; atol=10eps(Float64))
    end
end

@testset "bivariate Pickands identities" begin
    for tail in PICKANDS_CASES
        @test Copulas.A(tail, 0.0) ≈ 1
        @test Copulas.A(tail, 1.0) ≈ 1
        for t in (0.2, 0.5, 0.8)
            a = Copulas.A(tail, t)
            @test max(t, 1 - t) <= a + 10eps(Float64) <= 1 + 10eps(Float64)
            h = 1e-5
            finite_dA = (Copulas.A(tail, t + h) - Copulas.A(tail, t - h)) / (2h)
            finite_d²A = (Copulas.dA(tail, t + h) - Copulas.dA(tail, t - h)) / (2h)
            # Spectral atoms are legitimate kinks: classical first and second
            # derivatives need not agree with centered finite differences there.
            if !is_spectral_kink(tail, t)
                @test Copulas.dA(tail, t) ≈ finite_dA atol=2e-5
                @test Copulas.d²A(tail, t) ≈ finite_d²A atol=2e-4
            end
        end
    end
end
