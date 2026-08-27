# Infrastructure layer: applies Aqua's package-level hygiene checks. This is
# independent of the public behavioral and mathematical contracts below.
@testset "Aqua.jl" begin
  Aqua.test_all(
    Copulas;
    ambiguities = false,
  )
end
