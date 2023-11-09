@testitem "Aqua.jl" begin
    using Aqua
    Aqua.test_all(
      YourPackage;
      ambiguities=false,
      unbound_args=true,
      undefined_exports=true,
      project_extras=true,
      stale_deps=(ignore=[:SomePackage],),
      deps_compat=(ignore=[:SomeOtherPackage],),
      project_toml_formatting=true,
      piracy=false,
    )
end