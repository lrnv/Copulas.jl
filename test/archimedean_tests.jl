
@testitem "Test of τ ∘ τ_inv bijection" begin
    using Random 
    Random.seed!(123)
    taus = [0.0, 0.1, 0.5, 0.9, 1.0]

    for T in (
        # AMHCopula,
        ClaytonCopula,
        # FrankCopula,
        GumbelCopula,
        # IndependentCopula,
        # JoeCopula
    )
        for τ in taus
            @test Copulas.τ(T(2,Copulas.τ⁻¹(T,τ))) ≈ τ
        end
    end
end


@testitem "AMHCopula - Test sampling all cases" begin
    for d in 2:10
        for θ ∈ [-1.0,-rand(),0.0,rand()]
            rand(AMHCopula(d,θ),10)
        end
    end
    @test true
end
@testitem "ClaytonCopula - Test sampling all cases" begin
    rand(ClaytonCopula(2,-1),10)
    for d in 2:10
        for θ ∈ [-1/(d-1) * rand(),0.0,-log(rand()), Inf]
            rand(ClaytonCopula(d,θ),10)
        end
    end
    @test true
end
@testitem "FrankCopula - Test sampling all cases" begin
    rand(FrankCopula(2,-Inf),10)
    for d in 2:10
        for θ ∈ [log(rand()),0.0,rand(),1.0,-log(rand()), Inf]
            rand(FrankCopula(d,θ),10)
        end
    end
    @test true
end
@testitem "GumbelCopula - Test sampling all cases" begin
    for d in 2:10
        for θ ∈ [1.0,1-log(rand()), Inf]
            rand(GumbelCopula(d,θ),10)
        end
    end
    @test true
end
@testitem "JoeCopula - Test sampling all cases" begin
    for d in 2:10
        for θ ∈ [1.0,1-log(rand()), Inf]
            rand(JoeCopula(d,θ),10)
        end
    end
    @test true
end

