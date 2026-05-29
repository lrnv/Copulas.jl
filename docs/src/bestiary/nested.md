```@meta
CurrentModule = Copulas
```

# Nested Archimedean copulas

A *nested* (hierarchical) Archimedean copula glues several Archimedean copulas
together under an outer Archimedean generator, letting different blocks of
variables share a stronger within-block dependence while still being coupled
across blocks. This is the natural model for grouped or hierarchical
dependence — for example several organ systems within a patient, or several
assets within a sector.

[`NestedArchimedeanCopula`](@ref) provides the density (and an optional
per-variable censored / survival likelihood) of such trees, following the
algorithm of Yang & Li
([arXiv:2605.23134](https://arxiv.org/abs/2605.23134)).

## Definition

With an outer generator ``\phi_0`` over inner copulas ``C_1, \dots, C_m`` on
disjoint coordinate blocks ``I_1, \dots, I_m`` (and possibly some bare
coordinates attached directly to the root), the CDF is

```math
C(\mathbf u) = \phi_0\!\left(
  \sum_{i \in \text{root leaves}} \phi_0^{-1}(u_i)
  \;+\; \sum_{k=1}^m \phi_0^{-1}\bigl(C_k(\mathbf u_{I_k})\bigr)
\right),
```

and each child ``C_k`` is itself Archimedean or, recursively, nested
Archimedean — so trees nest to arbitrary depth.

The density is the mixed partial of this CDF over the differentiated
coordinates. Differentiating the composition of generators is exactly Faà di
Bruno's formula; the implementation carries the partial Bell polynomials through
truncated Taylor series over the generator tree, building only on the package's
generator interface (`ϕ`, `ϕ⁻¹`, `ϕ⁽ᵏ⁾`).

## Building a tree

```@example nested
using Copulas, Distributions
using Copulas: ClaytonGenerator, JoeGenerator

# Outer Clayton(2) over two inner Clayton panels on dims 1:2 and 3:4.
C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
        children = [ClaytonCopula(2, 5.0), ClaytonCopula(2, 6.0)])
logpdf(C, [0.3, 0.5, 0.4, 0.6])
```

Children are auto-placed on consecutive free dimension blocks in declaration
order. You can instead pin a child to explicit dimensions with a `Pair`, and you
can attach bare coordinates to the root with `leaves`:

```@example nested
# Root Clayton with a bare leaf on dim 1 and a Gumbel panel on dims 2:4.
C2 = NestedArchimedeanCopula(ClaytonGenerator(2.0);
         leaves = [1], children = [GumbelCopula(3, 2.0) => [2, 3, 4]])
logpdf(C2, [0.25, 0.4, 0.55, 0.7])
```

Mixed families and arbitrary nesting depth are supported:

```@example nested
inner = NestedArchimedeanCopula(JoeGenerator(3.0); children = [JoeCopula(2, 4.0)])
C3 = NestedArchimedeanCopula(ClaytonGenerator(1.5);
         children = [GumbelCopula(2, 2.0), FrankCopula(2, 3.0), inner])
logpdf(C3, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
```

A purely flat declaration (only `leaves`, no `children`) returns the package's
native [`ArchimedeanCopula`](@ref) so its fast specialised density is used.

!!! note "Validity is the caller's responsibility"
    The constructor does not check the nesting condition. For same-family
    nestings the standard sufficient condition is that the inner generator be at
    least as dependent as the outer one (e.g. for Clayton/Gumbel/Joe, the inner
    parameter ``\ge`` the outer parameter). Mixed-family nestings are accepted
    but must be validated by the user.

## Precision

The recursion is generic in the value type. `logpdf` works on `Float64`
out of the box; passing `BigFloat` (or `Double64`) coordinates carries that
precision through the whole recursion, which is recommended for adversarial
high-dimensional or deep-tail inputs where the alternating-sign Faà di Bruno sum
can lose `Float64` precision:

```@example nested
logpdf(C, big.([0.3, 0.5, 0.4, 0.6]))
```

## Censored / survival likelihood

For partially observed multivariate event times, the survival likelihood lives
at the [`SklarDist`](@ref) level — it is a statement about the data and the
margins, not the pure dependence on ``[0,1]^d``. Wrap the copula in a
`SklarDist` with its marginal distributions and pass a Boolean censoring mask
via the `censored` keyword of `logpdf`. A right-censored coordinate (`true`)
contributes its survival information only: it enters the copula CDF as a plain
argument and its marginal density is dropped, so the result is

```math
\sum_{i\,:\,\text{observed}} \log f_i(x_i)
\;+\; \log \frac{\partial^{|O|} C(\mathbf u)}{\prod_{i\in O}\partial u_i},
\qquad \mathbf u = (F_1(x_1),\dots,F_d(x_d)),
```

the observed marginal densities times the copula's mixed partial over the
observed coordinates ``O``:

```@example nested
using Distributions
Csurv = NestedArchimedeanCopula(ClaytonGenerator(2.0);
            children = [ClaytonCopula(3, 5.0), ClaytonCopula(3, 6.0)])
S = SklarDist(Csurv, ntuple(_ -> Exponential(1.0), 6))
x = [0.7, 0.3, 0.9, 0.5, 0.4, 1.1]
logpdf(S, x; censored = [false, true, false, false, false, true])
```

With `censored` omitted (all observed) this reduces to the ordinary joint
density; with all coordinates censored it reduces to ``\log F(\mathbf x)``. This
path works for any copula that supplies the mixed partial over observed
coordinates — flat [`ArchimedeanCopula`](@ref) as well as nested trees — so it
is a general per-variable censoring facility, not nested-specific.

!!! warning "Do not use `Distributions.censored` margins"
    `logpdf(SklarDist(C, (…, censored(m), …)), x)` returns `-Inf`: a censored
    margin places an atom at the censoring time, which Sklar's theorem maps to
    the copula boundary ``u = 1`` — off the open domain ``(0,1)^d`` where the
    copula density is defined. The `censored` keyword above is the correct
    replacement.

The copula factor alone is available as `censored_logpdf(C, u, δ)` on
copula-scale pseudo-observations ``u`` — the lower-level building block used by
the `SklarDist` method.
