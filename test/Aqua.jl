@testitem "Aqua.jl" begin
  using Aqua
  Aqua.test_all(
    Copulas;
    persistent_tasks = VERSION != v"1.10.10", # Disable persistent tasks only on Julia 1.10.10 (workaround for that release)
    ambiguities = false,
  )
end