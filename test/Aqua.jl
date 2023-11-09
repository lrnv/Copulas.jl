@testitem "Aqua.jl" begin
    using Aqua
    Aqua.test_all(
      Copulas;
      ambiguities=false,
    )
end