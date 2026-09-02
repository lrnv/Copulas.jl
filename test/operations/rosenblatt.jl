# Operation proof for `rosenblatt` and `inverse_rosenblatt`: public contract,
# generic mathematics, specialized equivalence, and dispatch-route closure.

function rosenblatt_route_key(C)
    Base.@nospecialize C
    is_absolutely_continuous(C) || return nothing
    d = length(C)
    method = which(Copulas.rosenblatt,
                   Tuple{typeof(C),Matrix{Float64}})
    return (method, d == 2 ? :bivariate : :multivariate)
end

function inverse_rosenblatt_route_key(C)
    Base.@nospecialize C
    is_absolutely_continuous(C) || return nothing
    d = length(C)
    method = which(Copulas.inverse_rosenblatt,
                   Tuple{typeof(C),Matrix{Float64}})
    return (method, d == 2 ? :bivariate : :multivariate)
end

function test_rosenblatt_contract(C, u, U)
    Base.@nospecialize C u U

    R = rosenblatt(C, U)
    @test size(R) == size(U)
    @test all(x -> 0 <= x <= 1, R)
    @test rosenblatt(C, u) ≈ vec(rosenblatt(C, reshape(u, :, 1)))
    @test inverse_rosenblatt(C, R) ≈ U atol=2e-5 rtol=2e-5
    @test inverse_rosenblatt(C, rosenblatt(C, u)) ≈ u atol=2e-5 rtol=2e-5
end

@testset "public Rosenblatt contract" begin
    @testset "$(fixture.case.name)" for (seed, fixture) in enumerate(COPULA_FIXTURES)
        C = fixture.copula
        is_absolutely_continuous(C) || continue
        test_progress("operations", "rosenblatt", fixture.case.name, "contract")
        u = copula_contract_point(C)
        U = rand(StableRNG(10_000 + seed), C, 4)
        test_rosenblatt_contract(C, u, U)
    end
end

@testset "Rosenblatt coordinates are conditional distribution functions" begin
    generic = PolynomialOracleCopula(0.4)
    generic_u = [0.37, 0.68]
    generic_R = rosenblatt(generic, generic_u)
    @test generic_R ≈ [generic_u[1],
                       _oracle_conditional_cdf(generic, generic_u[1], generic_u[2])]
    @test inverse_rosenblatt(generic, generic_R) ≈ generic_u atol=2e-6

    C = GaussianCopula{3}(0.3)
    u = [0.31, 0.52, 0.74]
    R = rosenblatt(C, u)
    @test R[1] ≈ u[1]
    @test R[2] ≈ cdf(condition(C, 1, u[1]).m[1], u[2])
    @test R[3] ≈ cdf(condition(C, (1, 2), (u[1], u[2])), u[3])
    @test inverse_rosenblatt(C, R) ≈ u atol=2e-6 rtol=2e-6

    independent = IndependentCopula{3}()
    @test rosenblatt(independent, u) == u
    @test inverse_rosenblatt(independent, u) == u
end

@testset "specialized Rosenblatt implementations agree with the generic path" begin
    u = [0.2 0.7; 0.4 0.6; 0.8 0.3]
    for C in (
        ClaytonCopula{3}(1.5),
        GaussianCopula{3}([1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
        TCopula{3}(5, [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
    )
        specialized = rosenblatt(C, u)
        generic = invoke(Copulas.rosenblatt,
            Tuple{Copulas.Copula{3},AbstractMatrix{<:Real}}, C, u)
        @test specialized ≈ generic atol=3e-10
        @test inverse_rosenblatt(C, specialized) ≈ u atol=3e-10
    end
end

@testset verbose=true "every Rosenblatt route equals sequential conditioning" begin
    selected_forward = Set(key
        for fixture in COPULA_FIXTURES
        for key in (rosenblatt_route_key(fixture.copula),)
        if !isnothing(key))
    selected_inverse = Set(key
        for fixture in COPULA_FIXTURES
        for key in (inverse_rosenblatt_route_key(fixture.copula),)
        if !isnothing(key))
    tested_forward = Set{Any}()
    tested_inverse = Set{Any}()

    for fixture in COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        is_absolutely_continuous(C) || continue
        forward_key = rosenblatt_route_key(C)
        inverse_key = inverse_rosenblatt_route_key(C)
        forward_done = forward_key in tested_forward
        inverse_done = isnothing(inverse_key) || inverse_key in tested_inverse
        forward_done && inverse_done && continue

        d = length(C)
        u = collect(range(0.31, 0.73; length=d))
        R = rosenblatt(C, u)
        expected = similar(R)
        expected[1] = u[1]
        for i in 2:d
            js = Tuple(1:(i - 1))
            values = Tuple(u[1:(i - 1)])
            expected[i] = cdf(Copulas.DistortionFromCop(C, js, values, i), u[i])
        end
        if !forward_done
            test_progress("operations", "rosenblatt", case.name, "route")
            @test R ≈ expected atol=2e-6 rtol=2e-6
            push!(tested_forward, forward_key)
        end
        if !inverse_done
            @test inverse_rosenblatt(C, R) ≈ u atol=2e-6 rtol=2e-6
            push!(tested_inverse, inverse_key)
        end
    end

    @test tested_forward == selected_forward
    @test tested_inverse == selected_inverse
end
