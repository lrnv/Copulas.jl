```@meta
CurrentModule = Copulas
```

# Nested Archimedean copulas

A **nested** (hierarchical) Archimedean copula glues several Archimedean copulas
together under an outer Archimedean generator, letting different blocks of
variables share a stronger within-block dependence while still being coupled
across blocks. This is the natural model for grouped or hierarchical
dependence — for example several organ systems within a patient, or several
assets within a sector.

[`NestedArchimedeanCopula`](@ref) provides the density of such trees (and, via
the standard [`condition`](@ref) / [`subsetdims`](@ref) framework, lower-tail
conditional likelihood contributions for partially observed coordinates),
following the algorithm of Yang & Li
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

To facilitate notation of leaves, we assume ``C_k(u_{I_k}) = u_{I_k}`` whenever ``\left|I_k\right|=1``.

The density is the mixed partial of this CDF over the differentiated
coordinates. Differentiating the composition of generators is exactly Faà di
Bruno's formula; the implementation carries the partial Bell polynomials through
truncated Taylor series over the generator tree, building only on the package's
generator machinery.

## Building a tree
```@example nested
using Copulas, Distributions
using StatsBase: coef
using Copulas: AMHGenerator, ClaytonGenerator
# Outer Clayton(2) over two inner Clayton panels on dims 1:2 and 3:4.
C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
        children = [ClaytonCopula(2, 5.0), ClaytonCopula(2, 6.0)])
logpdf(C, [0.3, 0.5, 0.4, 0.6])
```

Children are auto-placed on consecutive free dimension blocks in declaration
order. You can instead pin a child to explicit dimensions with a `Pair`, and you
can attach bare coordinates to the root with `leaves`:

```@example nested
# Root Clayton with a bare leaf on dim 1 and a stronger Clayton panel on dims 2:4.
C2 = NestedArchimedeanCopula(ClaytonGenerator(2.0);
         leaves = [1], children = [ClaytonCopula(3, 4.0) => [2, 3, 4]])
logpdf(C2, [0.25, 0.4, 0.55, 0.7])
```

Mixed families and arbitrary nesting depth are supported:

```@example nested
inner = NestedArchimedeanCopula(ClaytonGenerator(2.0); children = [ClaytonCopula(2, 4.0)])
C3 = NestedArchimedeanCopula(AMHGenerator(0.3);
         children = [ClaytonCopula(2, 1.5), inner])
logpdf(C3, [0.2, 0.3, 0.4, 0.5])
```

A purely flat declaration (only `leaves`, no `children`) returns the package's
native [`ArchimedeanCopula`](@ref) so its fast specialised density is used.

## Construction and fitting validity

The public constructor validates the **tree structure**: dimensions must be
consistent, child blocks and root leaves must be placed without overlap, and
each node generator must support the dimension in which it is used. It does
not attempt to certify the mathematical parent-child nesting relation. Explicit
construction therefore remains permissive, as it was before the fitting
parameter-space machinery was introduced.

Template fitting is deliberately stricter. For
`fit(template::NestedArchimedeanCopula, data)`, Copulas.jl asks
`Paramorph.param_space(template)` for the supported dependent parameter
geometry. The optimiser works only in unconstrained coordinates and
`Paramorph.constrain` reconstructs candidate trees inside that geometry. An
initial template outside the supported region, or a tree for which no fitting
geometry is implemented, is rejected before optimisation.

This separation is intentional: **construction describes a model; Paramorph
describes the subset that the generic fitter knows how to optimise safely**.
Consequently, a tree being unsupported by template fitting does not imply that
the constructor should reject it. Additional fitting support belongs in the
Paramorph geometry and its regression tests rather than in constructor-level
nesting rules.


### Nesting validity and fitting geometry

For template fitting, each supported edge is encoded directly in a
`Paramorph.DependentProduct`. The child's intrinsic generator domain is intersected
with the parent-child nesting constraint, so every finite optimiser coordinate maps
to a nesting that satisfies the implemented rule.

The currently implemented one-parameter rules are:

| Parent generator | Child generator | Fitting-valid region |
| --- | --- | --- |
| `IndependentGenerator` | any generator with an available local parameter space | no additional constraint |
| AMH | AMH | ``\theta_c \ge \theta_p`` |
| Clayton | Clayton | ``\theta_c \ge \theta_p`` |
| Frank | Frank | ``\theta_c \ge \theta_p`` |
| Gumbel | Gumbel | ``\theta_c \ge \theta_p`` |
| Gumbel-Barnett | Gumbel-Barnett | ``\theta_c \le \theta_p`` |
| inverse-Gaussian | inverse-Gaussian | ``\theta_c \ge \theta_p`` |
| Joe | Joe | ``\theta_c \ge \theta_p`` |
| AMH | Clayton | ``\theta_c \ge 1`` |

Here ``\theta_p`` and ``\theta_c`` denote the parent and child parameters. The
usual parametric representations of independence such as AMH(0), Clayton(0),
Frank(0), Gumbel(1), Gumbel-Barnett(0), inverse-Gaussian(0), and Joe(1) are also
mathematically valid above arbitrary valid children. They are not exposed as
cross-family fitting rules because fixing the parent exactly at independence is a
lower-dimensional boundary of the corresponding parametric family; use
`IndependentGenerator` when that is the intended parent model.

Some invalid regions are already known analytically. In particular:

- for the homogeneous AMH, Clayton, Frank, Gumbel, inverse-Gaussian and Joe
  families, ``\theta_c < \theta_p`` violates the implemented nesting condition;
- for homogeneous Gumbel-Barnett, ``\theta_c > \theta_p`` violates it;
- a strictly positive Clayton parent cannot contain a finite Frank, Gumbel, or Joe
  child (for child dimension at least two); these are certified invalid, not merely
  unsupported.

These negative results are intentionally different from a missing rule: absence
from the table above does **not** imply invalidity.

::: todo "Help complete the nesting-rule table"

    The validity classification is not yet exhaustive, especially for heterogeneous
    generator pairs and for the multi-parameter BB families. We already know some
    additional valid and invalid BB slices analytically, but they are not yet
    represented by the fitting geometry. Contributions are welcome to:

    - prove additional parent-child validity or impossibility results;
    - translate newly proved one-parameter rules into `GreaterThan`, `LowerThan`,
      or fixed-bound Paramorph geometries;
    - design suitable dependent parameter geometries for multi-parameter families;
    - add regression tests documenting the exact parameter region covered by each
      new rule.

    Please keep unsupported cases distinct from certified-invalid cases in the
    mathematical discussion, even though template fitting rejects both until a
    fitting geometry is implemented.

:::

# Fitting

`fit` performs maximum-likelihood estimation of the generator parameters on a
**fixed tree**: the leaf layout and the generator family at each node come from a
template instance, while the supported free parameters are determined by that
template's Paramorph space. Pass the template and a `d×n` matrix of
pseudo-observations (columns are observations).

```@example nested
using Random
Ctrue = NestedArchimedeanCopula(ClaytonGenerator(2.0);
            children = [ClaytonCopula(2, 6.0), ClaytonCopula(2, 8.0)])
U = rand(Random.MersenneTwister(1), Ctrue, 300)

# Fit from a deliberately wrong same-shape template:
Cstart = NestedArchimedeanCopula(ClaytonGenerator(1.0);
             children = [ClaytonCopula(2, 3.0), ClaytonCopula(2, 3.0)])
M = fit(CopulaModel, Cstart, U)
fitted_distribution(M)
```

The optimiser runs in an unconstrained space through a **parametrisation** — a map
`α -> NestedArchimedeanCopula` decoupled from the generator objects. Above we fit
a **template** tree. For full control, pass your own map and its initial point,
`fit(CopulaModel, reparam, init, U)` — no template needed, since `reparam` builds
the whole tree. This lets you share parameters across nodes, fit on a different
scale, or encode an application-specific constraint.

For instance, a custom map can make each child's ``\theta`` a non-negative
increment over its parent's:

```@example nested
softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))
nest = α -> NestedArchimedeanCopula(ClaytonGenerator(exp(α[1]));
    children = [ClaytonCopula(2, exp(α[1]) + softplus(α[2])),
                ClaytonCopula(2, exp(α[1]) + softplus(α[3]))])
Mn = fit(CopulaModel, nest, [0.0, 0.0, 0.0], U)
fitted_distribution(Mn)
```

Or share one ``\theta`` across the root and both panels — a single free parameter:

```@example nested
recon = α -> (θ = exp(α[1]);
    NestedArchimedeanCopula(ClaytonGenerator(θ);
        children = [ClaytonCopula(2, θ), ClaytonCopula(2, θ)]))
Ms = fit(CopulaModel, recon, [0.0], U)
fitted_distribution(Ms) # the root and both panels share one parameter by construction
coef(Ms)                 # one fitted free coordinate
```

`fit(C0, U)` is a shorthand returning just the fitted copula; for the custom form
use `fitted_distribution(fit(CopulaModel, reparam, init, U))`.

## Precision

The recursion is generic in the value type. `logpdf` works on `Float64`
out of the box; passing `BigFloat` (or `Double64`) coordinates carries that
precision through the whole recursion, which is recommended for adversarial
high-dimensional or deep-tail inputs where the alternating-sign Faà di Bruno sum
can lose `Float64` precision:

```@example nested
logpdf(C, big.([0.3, 0.5, 0.4, 0.6]))
```

## Partial-observation likelihood

Lower-tail partial-observation likelihoods are an **emergent capability** of the
standard [`condition`](@ref) + [`subsetdims`](@ref) framework — there is no
bespoke likelihood function. Split the coordinates into an observed set ``O``
and an unobserved/lower-tail set ``C``. The contribution factorises as the *gist
recipe*

```math
\underbrace{\log f_{O}(x_O)}_{\text{observed-marginal density}}
\;+\;
\underbrace{\log P(X_C \le x_C \mid X_O = x_O)}_{\text{conditional lower-tail probability}},
```

In code these two terms are `logpdf(subsetdims(X, O), x_O)` and
`logcdf(condition(X, O, x_O), x_C)`.

which equals the observed-marginal density times the copula's mixed partial over
the observed coordinates,

```math
\sum_{i\in O} \log f_i(x_i)
\;+\; \log \frac{\partial^{|O|} C(\mathbf u)}{\prod_{i\in O}\partial u_i},
\qquad \mathbf u = (F_1(x_1),\dots,F_d(x_d)),
```

because the denominator ``c_O`` in `condition` cancels against the
`subsetdims` marginal density.

```@example nested
using Distributions
Cpart = NestedArchimedeanCopula(ClaytonGenerator(2.0);
            children = [ClaytonCopula(3, 5.0), ClaytonCopula(3, 6.0)])
S = SklarDist(Cpart, ntuple(_ -> Exponential(1.0), 6))
x = [0.7, 0.3, 0.9, 0.5, 0.4, 1.1]
O = (1, 3, 4, 5)        # observed
C = (2, 6)              # lower-tail coordinates
logpdf(subsetdims(S, O), x[collect(O)]) +
    log(cdf(condition(S, O, x[collect(O)]), x[collect(C)]))
```

For right-censored coordinates, flip those coordinates with
[`SurvivalCopula`](@ref) and apply the same recipe to the survival-scale
coordinates. On the copula scale this computes
``P(U_C > u_C \mid U_O = u_O)`` as a lower-tail probability of the flipped
coordinates:

```@example nested
margins = last(params(S))
u = [cdf(margins[i], x[i]) for i in 1:6]
Cs = SurvivalCopula(Cpart, C)
logpdf(subsetdims(Cpart, O), u[collect(O)]) +
    log(cdf(condition(Cs, O, u[collect(O)]), 1 .- u[collect(C)]))
```

On the data scale, add the observed marginal log densities
``\sum_{i\in O}\log f_i(x_i)`` to this copula-scale contribution.

When a **single** coordinate is in ``C``, `condition(S, O, x_O)` returns a
univariate conditional distribution and you use `logcdf(condition(...), x_C)`
(a scalar `x_C`); when several are in ``C`` it returns a conditional joint
distribution and you use `log(cdf(condition(...), x_C))` as above. With ``C``
empty (all observed) the recipe reduces to the ordinary joint density
`logpdf(S, x)`; with ``O`` empty it reduces to
``\log F(\mathbf x)``. The recipe works for any copula with `condition` /
`subsetdims` support — flat [`ArchimedeanCopula`](@ref) as well as nested trees.

!!! note "Multi-coordinate conditional CDF"

    With two or more lower-tail coordinates the conditional CDF is the mixed
    partial of the nested CDF over the **observed** coordinates. Its cost grows
    quickly with the number of observed coordinates. At high differentiation
    order, `Float64` calculations can also lose precision; end-to-end `BigFloat`
    conditioning is not currently supported.
