# Mechanism-path layer: exercises one representative of each important generic
# or specialized sampling, conditioning, subsetting, and numerical dispatch path.
@testset "representative dispatch paths" begin
    for (name, C) in pairs(PATH_CASES)
        @testset "$name" begin
            d = length(C)
            u = fill(0.6, d)
            @test 0 <= cdf(C, u) <= 1
            @test size(rand(StableRNG(51), C, 2)) == (d, 2)
            js = Tuple(1:(d - 1))
            D = condition(C, js, ntuple(_ -> 0.4, d - 1))
            @test 0 <= cdf(D, 0.6) <= 1
        end
    end
end


@testset "generic numeric sampler buffers" begin
    C = ClaytonCopula{3}(1.0)
    storage = fill(Float32(NaN), 5, 2)
    buffer = @view storage[2:4, :]
    @test rand!(StableRNG(52), C, buffer) === buffer
    @test all(x -> 0 <= x <= 1, buffer)
    @test all(isnan, storage[[1, 5], :])
    @test_throws DimensionMismatch rand!(StableRNG(52), C, zeros(Float32, 2, 1))
end
