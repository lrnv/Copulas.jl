# Exhaustive branch-level regression coverage for the analytical nesting
# certificates used by NestedArchimedeanCopula. Every case below exercises a
# branch that explicitly returns VALID or INVALID; UNSUPPORTED regions are not
# promoted to mathematical claims.

const _VALID = Copulas._NESTING_VALID
const _INVALID = Copulas._NESTING_INVALID

function _check_nesting_cases(cases, expected)
    for (name, parent, child, d) in cases
        @testset "$name" begin
            @test Copulas._nested_status(parent, child, d) === expected
        end
    end
end

@testset "analytical nested Archimedean certificates" begin
    valid_cases = [
        ("Independent parent", Copulas.IndependentGenerator(), Copulas.ClaytonGenerator(2.0), 2),
        ("AMH independence parent", Copulas.AMHGenerator(0.0), Copulas.ClaytonGenerator(2.0), 2),
        ("Clayton independence parent", Copulas.ClaytonGenerator(0.0), Copulas.GumbelGenerator(2.0), 2),
        ("Frank independence parent", Copulas.FrankGenerator(0.0), Copulas.ClaytonGenerator(2.0), 2),
        ("Gumbel independence parent", Copulas.GumbelGenerator(1.0), Copulas.ClaytonGenerator(2.0), 2),
        ("Gumbel-Barnett independence parent", Copulas.GumbelBarnettGenerator(0.0), Copulas.ClaytonGenerator(2.0), 2),
        ("inverse-Gaussian independence parent", Copulas.InvGaussianGenerator(0.0), Copulas.ClaytonGenerator(2.0), 2),
        ("Joe independence parent", Copulas.JoeGenerator(1.0), Copulas.ClaytonGenerator(2.0), 2),

        ("AMH ordered", Copulas.AMHGenerator(0.2), Copulas.AMHGenerator(0.6), 2),
        ("Clayton ordered", Copulas.ClaytonGenerator(1.0), Copulas.ClaytonGenerator(2.0), 3),
        ("Frank ordered", Copulas.FrankGenerator(1.0), Copulas.FrankGenerator(3.0), 2),
        ("Gumbel ordered", Copulas.GumbelGenerator(1.5), Copulas.GumbelGenerator(2.5), 2),
        ("Gumbel-Barnett reverse order", Copulas.GumbelBarnettGenerator(0.7), Copulas.GumbelBarnettGenerator(0.3), 2),
        ("inverse-Gaussian ordered", Copulas.InvGaussianGenerator(0.5), Copulas.InvGaussianGenerator(1.2), 2),
        ("Joe ordered", Copulas.JoeGenerator(1.5), Copulas.JoeGenerator(3.0), 2),

        ("AMH to Clayton", Copulas.AMHGenerator(0.4), Copulas.ClaytonGenerator(1.2), 2),
        ("AMH to BB1", Copulas.AMHGenerator(0.4), Copulas.BB1Generator(1.2, 1.5), 2),
        ("AMH to BB2", Copulas.AMHGenerator(0.4), Copulas.BB2Generator(1.2, 1.0), 2),
        ("Clayton to BB1", Copulas.ClaytonGenerator(0.5), Copulas.BB1Generator(1.0, 2.0), 2),
        ("Clayton to BB2", Copulas.ClaytonGenerator(0.5), Copulas.BB2Generator(1.0, 1.0), 2),

        ("BB1 parent Clayton slice", Copulas.BB1Generator(1.0, 1.0), Copulas.ClaytonGenerator(2.0), 2),
        ("BB3 parent Clayton slice", Copulas.BB3Generator(1.0, 1.0), Copulas.ClaytonGenerator(2.0), 2),
        ("BB6 parent Joe slice", Copulas.BB6Generator(1.5, 1.0), Copulas.JoeGenerator(2.0), 2),
        ("BB6 parent Gumbel slice", Copulas.BB6Generator(1.0, 1.5), Copulas.GumbelGenerator(2.0), 2),
        ("BB7 parent Clayton slice", Copulas.BB7Generator(1.0, 1.0), Copulas.ClaytonGenerator(2.0), 2),
        ("BB8 independence slice", Copulas.BB8Generator(1.0, 0.5), Copulas.ClaytonGenerator(2.0), 2),
        ("BB8 parent Joe slice", Copulas.BB8Generator(1.5, 1.0), Copulas.JoeGenerator(2.0), 2),
        ("BB9 independence slice", Copulas.BB9Generator(1.0, 1.0), Copulas.ClaytonGenerator(2.0), 2),
        ("BB9 parent inverse-Gaussian slice", Copulas.BB9Generator(2.0, 0.5), Copulas.InvGaussianGenerator(1.0), 2),
        ("BB10 independence slice", Copulas.BB10Generator(2.0, 0.0), Copulas.ClaytonGenerator(2.0), 2),
        ("BB10 parent AMH slice", Copulas.BB10Generator(1.0, 0.2), Copulas.AMHGenerator(0.5), 2),

        ("AMH to BB3 Clayton slice", Copulas.AMHGenerator(0.3), Copulas.BB3Generator(1.0, 1.5), 2),
        ("Clayton to BB3 Clayton slice", Copulas.ClaytonGenerator(0.5), Copulas.BB3Generator(1.0, 1.5), 2),
        ("Gumbel to BB6 Gumbel slice", Copulas.GumbelGenerator(1.5), Copulas.BB6Generator(1.0, 2.0), 2),
        ("AMH to BB7 Clayton slice", Copulas.AMHGenerator(0.3), Copulas.BB7Generator(1.0, 1.5), 2),
        ("Clayton to BB7 Clayton slice", Copulas.ClaytonGenerator(0.5), Copulas.BB7Generator(1.0, 1.5), 2),
        ("Joe to BB8 Joe slice", Copulas.JoeGenerator(1.5), Copulas.BB8Generator(2.0, 1.0), 2),
        ("inverse-Gaussian to BB9 slice", Copulas.InvGaussianGenerator(0.5), Copulas.BB9Generator(2.0, 1.0), 2),
        ("AMH to BB10 AMH slice", Copulas.AMHGenerator(0.2), Copulas.BB10Generator(1.0, 0.5), 2),

        ("Gumbel to BB3", Copulas.GumbelGenerator(1.5), Copulas.BB3Generator(2.0, 1.0), 2),
        ("Joe to BB6", Copulas.JoeGenerator(1.5), Copulas.BB6Generator(2.0, 2.0), 2),

        ("BB1 equal theta", Copulas.BB1Generator(1.0, 1.5), Copulas.BB1Generator(1.0, 2.0), 2),
        ("BB2 equal theta", Copulas.BB2Generator(1.0, 1.0), Copulas.BB2Generator(1.0, 2.0), 2),
        ("BB2 Nelsen-20 slice", Copulas.BB2Generator(0.5, 1.0), Copulas.BB2Generator(1.0, 1.0), 2),
        ("BB3 equal theta", Copulas.BB3Generator(2.0, 1.0), Copulas.BB3Generator(2.0, 2.0), 2),
        ("BB6 independence corner", Copulas.BB6Generator(1.0, 1.0), Copulas.BB6Generator(2.0, 2.0), 2),
        ("BB6 equal theta", Copulas.BB6Generator(2.0, 1.2), Copulas.BB6Generator(2.0, 2.0), 2),
        ("BB6 Joe-parent branch", Copulas.BB6Generator(1.5, 1.0), Copulas.BB6Generator(2.0, 2.0), 2),
        ("BB7 equal theta", Copulas.BB7Generator(2.0, 1.0), Copulas.BB7Generator(2.0, 2.0), 2),
        ("BB8 independence branch", Copulas.BB8Generator(1.0, 0.5), Copulas.BB8Generator(3.0, 0.2), 2),
        ("BB8 common delta", Copulas.BB8Generator(1.5, 0.5), Copulas.BB8Generator(2.5, 0.5), 2),
        ("BB9 independence branch", Copulas.BB9Generator(1.0, 1.0), Copulas.BB9Generator(3.0, 2.0), 2),
        ("BB9 inverse-Gaussian slice", Copulas.BB9Generator(2.0, 0.5), Copulas.BB9Generator(2.0, 1.0), 2),
        ("BB9 common delta", Copulas.BB9Generator(1.5, 1.0), Copulas.BB9Generator(3.0, 1.0), 2),
        ("BB10 independence branch", Copulas.BB10Generator(2.0, 0.0), Copulas.BB10Generator(1.0, 0.8), 2),
        ("BB10 equal theta", Copulas.BB10Generator(2.0, 0.2), Copulas.BB10Generator(2.0, 0.8), 2),
    ]

    invalid_cases = [
        ("AMH reversed", Copulas.AMHGenerator(0.6), Copulas.AMHGenerator(0.2), 2),
        ("Clayton reversed", Copulas.ClaytonGenerator(2.0), Copulas.ClaytonGenerator(1.0), 2),
        ("Frank reversed", Copulas.FrankGenerator(3.0), Copulas.FrankGenerator(1.0), 2),
        ("Gumbel reversed", Copulas.GumbelGenerator(2.5), Copulas.GumbelGenerator(1.5), 2),
        ("Gumbel-Barnett reversed", Copulas.GumbelBarnettGenerator(0.3), Copulas.GumbelBarnettGenerator(0.7), 2),
        ("inverse-Gaussian reversed", Copulas.InvGaussianGenerator(1.2), Copulas.InvGaussianGenerator(0.5), 2),
        ("Joe reversed", Copulas.JoeGenerator(3.0), Copulas.JoeGenerator(1.5), 2),

        ("BB1 parent Clayton reversed", Copulas.BB1Generator(3.0, 1.0), Copulas.ClaytonGenerator(2.0), 2),
        ("BB3 parent Clayton reversed", Copulas.BB3Generator(1.0, 3.0), Copulas.ClaytonGenerator(2.0), 2),
        ("BB6 parent Joe reversed", Copulas.BB6Generator(3.0, 1.0), Copulas.JoeGenerator(2.0), 2),
        ("BB6 parent Gumbel reversed", Copulas.BB6Generator(1.0, 3.0), Copulas.GumbelGenerator(2.0), 2),
        ("BB7 parent Clayton reversed", Copulas.BB7Generator(1.0, 3.0), Copulas.ClaytonGenerator(2.0), 2),
        ("BB8 parent Joe reversed", Copulas.BB8Generator(3.0, 1.0), Copulas.JoeGenerator(2.0), 2),
        ("BB9 parent inverse-Gaussian reversed", Copulas.BB9Generator(2.0, 2.0), Copulas.InvGaussianGenerator(1.0), 2),
        ("BB10 parent AMH reversed", Copulas.BB10Generator(1.0, 0.7), Copulas.AMHGenerator(0.3), 2),

        ("Clayton to BB3 reversed", Copulas.ClaytonGenerator(2.0), Copulas.BB3Generator(1.0, 1.0), 2),
        ("Gumbel to BB6 reversed", Copulas.GumbelGenerator(3.0), Copulas.BB6Generator(1.0, 2.0), 2),
        ("Clayton to BB7 reversed", Copulas.ClaytonGenerator(2.0), Copulas.BB7Generator(1.0, 1.0), 2),
        ("Joe to BB8 reversed", Copulas.JoeGenerator(3.0), Copulas.BB8Generator(2.0, 1.0), 2),
        ("inverse-Gaussian to BB9 reversed", Copulas.InvGaussianGenerator(2.0), Copulas.BB9Generator(2.0, 1.0), 2),
        ("AMH to BB10 reversed", Copulas.AMHGenerator(0.7), Copulas.BB10Generator(1.0, 0.3), 2),

        ("Gumbel to BB3 reversed", Copulas.GumbelGenerator(3.0), Copulas.BB3Generator(2.0, 1.0), 2),
        ("BB3 to Gumbel", Copulas.BB3Generator(2.0, 1.0), Copulas.GumbelGenerator(2.0), 2),
        ("Joe to BB6 Joe slice reversed", Copulas.JoeGenerator(3.0), Copulas.BB6Generator(2.0, 1.0), 2),
        ("Clayton to Frank", Copulas.ClaytonGenerator(1.0), Copulas.FrankGenerator(2.0), 2),
        ("Clayton to Gumbel", Copulas.ClaytonGenerator(1.0), Copulas.GumbelGenerator(2.0), 2),
        ("Clayton to Joe", Copulas.ClaytonGenerator(1.0), Copulas.JoeGenerator(2.0), 2),

        ("BB1 delta reversal", Copulas.BB1Generator(1.0, 2.5), Copulas.BB1Generator(1.0, 2.0), 2),
        ("BB1 asymptotic reversal", Copulas.BB1Generator(2.0, 1.5), Copulas.BB1Generator(1.0, 2.0), 2),
        ("BB2 theta reversal", Copulas.BB2Generator(2.0, 1.0), Copulas.BB2Generator(1.0, 2.0), 2),
        ("BB2 delta reversal", Copulas.BB2Generator(1.0, 2.0), Copulas.BB2Generator(1.0, 1.0), 2),
        ("BB3 theta reversal", Copulas.BB3Generator(3.0, 1.0), Copulas.BB3Generator(2.0, 2.0), 2),
        ("BB3 delta reversal", Copulas.BB3Generator(2.0, 2.0), Copulas.BB3Generator(2.0, 1.0), 2),
        ("BB6 delta reversal", Copulas.BB6Generator(2.0, 2.0), Copulas.BB6Generator(3.0, 1.5), 2),
        ("BB7 theta reversal", Copulas.BB7Generator(3.0, 1.0), Copulas.BB7Generator(2.0, 2.0), 2),
        ("BB7 delta reversal", Copulas.BB7Generator(2.0, 2.0), Copulas.BB7Generator(2.0, 1.0), 2),
        ("BB8 common-delta reversal", Copulas.BB8Generator(3.0, 0.5), Copulas.BB8Generator(2.0, 0.5), 2),
        ("BB9 inverse-Gaussian reversal", Copulas.BB9Generator(2.0, 2.0), Copulas.BB9Generator(2.0, 1.0), 2),
        ("BB9 common-delta reversal", Copulas.BB9Generator(3.0, 1.0), Copulas.BB9Generator(2.0, 1.0), 2),
        ("BB9 equal-theta delta reversal", Copulas.BB9Generator(3.0, 2.0), Copulas.BB9Generator(3.0, 1.0), 2),
        ("BB10 equal-theta reversal", Copulas.BB10Generator(2.0, 0.8), Copulas.BB10Generator(2.0, 0.2), 2),
    ]

    _check_nesting_cases(valid_cases, _VALID)
    _check_nesting_cases(invalid_cases, _INVALID)
end
