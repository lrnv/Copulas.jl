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
)

@testset "public extreme-value tail primitives" begin
    for (tail, d) in TAIL_CASES
        @testset "$(nameof(typeof(tail))) d=$d" begin
            x = collect(range(0.4, 1.0; length=d))
            value = Copulas.ℓ(tail, x)
            @test maximum(x) <= value <= sum(x)
            @test Copulas.ℓ(tail, 2 .* x) ≈ 2value
            for i in 1:d
                e = zeros(d)
                e[i] = 1
                @test Copulas.ℓ(tail, e) ≈ 1
            end
            @test Copulas.ellpartial(tail, x, (1,)) isa Real
        end
    end
end
