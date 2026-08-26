struct CopulaContractContext{TU,TM}
    u::TU
    U::TM
end

function CopulaContractContext(C, seed)
    d = length(C)
    u = collect(range(0.31, 0.69; length=d))
    return CopulaContractContext(u, rand(StableRNG(seed), C, 4))
end

function test_copula_contract(case, seed)
    @testset "$(case.name)" begin
        C = case.build()
        ctx = CopulaContractContext(C, seed)
        test_distribution_contract(C, ctx)
        test_density_contract(C, ctx, case.kind)
        test_subsetting_contract(C, ctx)
        test_conditioning_contract(C, ctx)
        test_rosenblatt_contract(C, ctx, case.rosenblatt)
        test_dependence_contract(C)
    end
end

@testset "public copula contract" begin
    for (i, case) in pairs(COPULA_CASES)
        test_copula_contract(case, 10_000 + i)
    end
end
