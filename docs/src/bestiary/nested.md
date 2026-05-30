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

[`NestedArchimedeanCopula`](@ref) provides the density of such trees (and, via
the standard [`condition`](@ref) / [`subsetdims`](@ref) framework, a
per-variable censored / survival likelihood), following the algorithm of Yang &
Li ([arXiv:2605.23134](https://arxiv.org/abs/2605.23134)).

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

Per-variable (right-)censoring is an **emergent capability** of the standard
[`condition`](@ref) + [`subsetdims`](@ref) framework — there is no bespoke
censored-likelihood function. Split the coordinates into an observed set ``O``
(events) and a censored set ``C`` (the survival times we only know exceed their
observation). The per-variable censored likelihood factorises as the *gist
recipe*

```math
\underbrace{\log f_{O}(x_O)}_{\texttt{logpdf(subsetdims(X,O),\,x_O)}}
\;+\;
\underbrace{\log P(X_C \le x_C \mid X_O = x_O)}_{\texttt{logcdf(condition(X,O,x_O),\,x_C)}},
```

which equals the observed-marginal density times the copula's mixed partial over
the observed coordinates,

```math
\sum_{i\in O} \log f_i(x_i)
\;+\; \log \frac{\partial^{|O|} C(\mathbf u)}{\prod_{i\in O}\partial u_i},
\qquad \mathbf u = (F_1(x_1),\dots,F_d(x_d)),
```

because the denominator ``c_O`` in `condition` cancels against the
`subsetdims` marginal density. Both factors route through the Faà di Bruno tree
walk via the `subsetdims` / `condition` specialisations for this type — no
ForwardDiff for the observed-marginal density nor for the conditional CDF (for
any number of censored coordinates).

```@example nested
using Distributions
Csurv = NestedArchimedeanCopula(ClaytonGenerator(2.0);
            children = [ClaytonCopula(3, 5.0), ClaytonCopula(3, 6.0)])
S = SklarDist(Csurv, ntuple(_ -> Exponential(1.0), 6))
x = [0.7, 0.3, 0.9, 0.5, 0.4, 1.1]
O = (1, 3, 4, 5)        # observed (events)
C = (2, 6)              # right-censored
logpdf(subsetdims(S, O), x[collect(O)]) +
    log(cdf(condition(S, O, x[collect(O)]), x[collect(C)]))
```

When a *single* coordinate is censored, `condition(S, O, x_O)` returns a
univariate conditional distribution and you use `logcdf(condition(...), x_C)`
(a scalar `x_C`); when several are censored it returns a conditional joint
distribution and you use `log(cdf(condition(...), x_C))` as above. With ``C``
empty (all observed) the recipe reduces to the ordinary joint density
`logpdf(S, x)`; with ``O`` empty (all censored) it reduces to
``\log F(\mathbf x)``. The recipe works for any copula with `condition` /
`subsetdims` support — flat [`ArchimedeanCopula`](@ref) as well as nested trees.

!!! note "Multi-censored conditional CDF"
    With two or more censored coordinates the conditional CDF is the mixed
    partial of the nested CDF over the *observed* coordinates. The generic path
    takes this by nesting one `ForwardDiff.derivative` per observed coordinate —
    cost exponential in the number of observed dims, infeasible in high
    dimension. A `_partial_cdf` specialisation routes it instead through the same
    polynomial Faà di Bruno tree walk as the single-censored case (selected on the
    conditional copula's concrete nested inner type), for any number of censored
    coordinates.

    At high differentiation order for fast-tail generators the `Float64` sum can
    lose precision; pass `BigFloat` coordinates to recover the exact value (as for
    the density). End-to-end `BigFloat` through `condition()` is not yet enabled —
    upstream stores the conditioning values as `Float64`.

!!! warning "Do not use `Distributions.censored` margins"
    `logpdf(SklarDist(C, (…, censored(m), …)), x)` returns `-Inf`: a censored
    margin places an atom at the censoring time, which Sklar's theorem maps to
    the copula boundary ``u = 1`` — off the open domain ``(0,1)^d`` where the
    copula density is defined. The `condition` + `subsetdims` recipe above is the
    correct replacement.
