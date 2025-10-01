```@meta
CurrentModule = Copulas
```

# [Archimax Models](@id available_archimax_models)

Archimax copulas are built by pairing an Archimedean generator with an extreme value tail bahavior. Beyond the named families below, **any Archimedean model in this package can be combined with any EV model** via the generic constructor `ArchimaxCopula(gen, tail)`.

!!! info "Bivariate only"
The current implementation is only bivariate since we only have bivariate extreme value tails. If you’re interested in contributing a multivariate extensions, please reach out.

## `BB4Copula`

```@docs; canonical=false
BB4Copula
```

## `BB5Copula`

```@docs; canonical=false
BB5Copula
```

## Build-your-own Archimax

Use any Archimedean generator `G<:Generator` and any extreme value tail `E<:Tail`:

```julia
gen = ClaytonGenerator(7.0)   # any Archimedean generator
tail = GalambosTail(3.2)      # any extreme value tail
C = ArchimaxCopula(gen, tail) # bivariate Archimax copula
samples = rand(C,1000)        # sampling
cdf(C,samples)                # cdf
pdf(C,samples)                # pdf
```

**Building blocks:**

* Archimedean generators → see \[available Archimedean generators]\(@ref available\_archimedean\_models).
* Extreme-value tails → see \[available extreme-value tails]\(@ref available\_extreme\_models).

Parameter validity is handled by the underlying types; no extra checks are needed at the Archimax level.

!!! tip "Sampling"
The provided sampler for Archimax uses the **frailty** representation: `M ∼ frailty(gen)` with Laplace transform `ϕ`, and EV draws `V` from `tail`. It returns `U = ϕ.( -log.(V) ./ M )`.
This requires the generator to be **completely monotone** (so that a frailty distribution exists). If a generator is only 2-monotone, `cdf`/`pdf` work as usual, but `rand` may not be available.

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
