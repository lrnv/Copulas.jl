```@meta
CurrentModule = Copulas
```

# [Archimax Models](@id available_archimax_models)

Archimax copulas are built by pairing an Archimedean generator with an extreme-value (EV) copula. Beyond the named families below, **any Archimedean model in this package can be combined with any EV model** via the generic constructor `ArchimaxCopula(gen, evd)`.

!!! note "Bivariate only"
The current implementation is bivariate. If you’re interested in contributing multivariate extensions, please reach out.

## `BB4Copula`

```@docs
BB4Copula
```

## `BB5Copula`

```@docs
BB5Copula
```

## Build-your-own Archimax

Use any Archimedean generator `G<:Generator` and any EV copula `E<:ExtremeValueCopula`:

```@example
using Copulas

gen = ClaytonGenerator(θ)      # any Archimedean generator
evd = GalambosEV(κ)            # any EV copula

C = ArchimaxCopula(gen, evd)   # bivariate Archimax copula

samples = rand(C,1000)   # sampling
cdf(C,samples)           # cdf
pdf(C,samples)           # pdf
```

**Building blocks:**

* Archimedean generators → see \[available Archimedean generators]\(@ref available\_archimedean\_models).
* Extreme-value copulas → see \[available extreme-value models]\(@ref available\_extreme\_models).

Parameter validity is handled by the underlying types; no extra checks are needed at the Archimax level.

!!! tip "Sampling"
The provided sampler for Archimax uses the **frailty** representation: `M ∼ frailty(gen)` with Laplace transform `ϕ`, and EV draws `V` from `evd`. It returns `U = ϕ.( -log.(V) ./ M )`.
This requires the generator to be **completely monotone** (so that a frailty distribution exists). If a generator is only 2-monotone, `cdf`/`pdf` work as usual, but `rand` may not be available.

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
