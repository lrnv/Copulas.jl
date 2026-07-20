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

Be carefull about validity of the wanted nesting tree. Let us recall from
[hofert2012nesting](@cite) and [mcneil2008](@cite) the following central results.


!!! theorem "Two-generator nesting condition"
    Consider one parent generator ``\phi_0`` and one child generator ``\phi_1``,
    with a child block of dimension ``d_1 \ge 2``. Write
    ``h = \phi_0^{-1} \circ \phi_1``. Assuming the parent and child generators
    are themselves valid Archimedean generators for their respective dimensions,
    the two-generator nested construction is valid if and only if ``h'`` is
    ``d_1``-monotone on ``(0,\infty)`` [hofert2012nesting](@cite).

    The often-used *sufficient nesting condition* asks instead that ``h'`` be
    completely monotone [mcneil2008](@cite). This stronger condition is convenient
    because it is dimension-free, but it is not necessary in finite dimensions.
    For a full tree, the finite-dimensional condition must hold on every
    parent-child edge, using the dimension of the child subtree.

!!! note "Validity is the caller's responsibility"
    The constructor does not check the nesting condition, you have to check them yourself.

!!! tip "Same-families rule of thumb"
    For same-family nestings the standard sufficient condition is that the inner generator be at
    least as dependent as the outer one (e.g. for Clayton/Gumbel/Joe, the inner
    parameter ``\ge`` the outer parameter). Mixed-family nestings are accepted
    but must be validated by the user.


## Fitting

`fit` performs maximum-likelihood estimation of the generator parameters on a
**fixed tree**: the leaf layout and the generator family at each node come from a
template instance, and only the scalar ``\theta`` of each node is optimised. Pass
the template and a `d×n` matrix of pseudo-observations (columns are observations).

```@example nested
using Random
Ctrue = NestedArchimedeanCopula(ClaytonGenerator(2.0);
            children = [ClaytonCopula(2, 6.0), ClaytonCopula(2, 8.0)])
U = rand(Random.MersenneTwister(1), Ctrue, 2000)

# Fit from a deliberately wrong same-shape template:
Cstart = NestedArchimedeanCopula(ClaytonGenerator(1.0);
             children = [ClaytonCopula(2, 3.0), ClaytonCopula(2, 3.0)])
M = fit(CopulaModel, Cstart, U)
M.result
```

The optimiser runs in an unconstrained space through a *parametrisation* — a map
`α -> NestedArchimedeanCopula` decoupled from the generator objects. Above we fit
a **template** tree. For full control, pass your own map and its initial point,
`fit(CopulaModel, reparam, init, U)` — no template needed, since `reparam` builds
the whole tree. This lets you share parameters across nodes, fit on a different
scale, or encode a constraint.

For instance, enforce the nesting condition by building each child's ``\theta`` as
a non-negative increment over its parent's, so every step is a valid nesting:

```@example nested
softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))
nest = α -> NestedArchimedeanCopula(ClaytonGenerator(exp(α[1]));
    children = [ClaytonCopula(2, exp(α[1]) + softplus(α[2])),
                ClaytonCopula(2, exp(α[1]) + softplus(α[3]))])
Mn = fit(CopulaModel, nest, [0.0, 0.0, 0.0], U)
(Mn.result.G.θ, Mn.result.children[1][1].G.θ)   # inner θ ≥ outer θ, by construction
```

Or share one ``\theta`` across the root and both panels — a single free parameter:

```@example nested
recon = α -> (θ = exp(α[1]);
    NestedArchimedeanCopula(ClaytonGenerator(θ);
        children = [ClaytonCopula(2, θ), ClaytonCopula(2, θ)]))
Ms = fit(CopulaModel, recon, [0.0], U)
(Ms.result.G.θ, Ms.result.children[1][1].G.θ)   # equal — the shared parameter
```

`fit(C0, U)` is a shorthand returning just the fitted copula; for the custom form
use `fit(CopulaModel, reparam, init, U).result`.

## Precision

The recursion is generic in the value type. `logpdf` works on `Float64`
out of the box; passing `BigFloat` (or `Double64`) coordinates carries that
precision through the whole recursion, which is recommended for adversarial
high-dimensional or deep-tail inputs where the alternating-sign Faà di Bruno sum
can lose `Float64` precision:

```@example nested
logpdf(C, big.([0.3, 0.5, 0.4, 0.6]))
```

## Edge-composition method

Each parent→child edge in the Faà di Bruno recursion needs the truncated Taylor
expansion of the inner-to-outer link ``h = \phi^{-1}_{\text{outer}} \circ
\phi_{\text{inner}}`` at the child's argument. It goes through the overloadable
hook `composition_taylor(outer, inner, t₀, d)`, selected by dispatch exactly as
you override `ϕ⁽ᵏ⁾` — most-specific method wins, no keyword or flag. Three methods
are available:

**1. Direct (default).** `composition_taylor_direct` puts a single jet through
the explicit composition. Fast and accurate for ordinary inputs. It requires both
``\phi`` and ``\phi^{-1}`` to accept a `Taylor1` argument.

**2. Implicit.** `composition_taylor_implicit` solves
``\phi_{\text{outer}}(h(t)) = \phi_{\text{inner}}(t)`` order-by-order, using only
the scalar derivatives ``\phi^{(k)}`` of both generators and a single scalar
``\phi^{-1}_{\text{outer}}`` — it never puts a `Taylor1` through ``\phi^{-1}``.
Use it when a generator's ``\phi^{-1}`` has no `Taylor1` method (for instance, an
inverse defined only through root-finding), where the direct jet cannot run.
Select it globally by redefining the generic method:

```julia
Copulas.composition_taylor(o::Copulas.Generator, i::Copulas.Generator, t₀, d) =
    Copulas.composition_taylor_implicit(o, i, t₀, d)
```

(Redefining the shipped default prints a benign "method overwritten" warning.)

**3. Closed form (per generator pair).** Register a more-specific method when you
know ``h`` analytically — the fastest and most robust option. The package ships
Clayton/Clayton (in `Generator/ClaytonGenerator.jl`),
``h(t) = ((1+\theta_{\text{in}} t)^{\theta_{\text{out}}/\theta_{\text{in}}} - 1)/
\theta_{\text{out}}``; add your own the same way:

```julia
function Copulas.composition_taylor(outer::MyGenerator, inner::MyGenerator, t₀, d::Int)
    # return [h'(t₀)/1!, …, h⁽ᵈ⁾(t₀)/d!]
end
```

If you implement your own closed-form for rather-standard cases, do not hesitate to share them with others through a PR on the package. 

The method choice is about *availability and speed*, not accuracy: all three are
exact in exact arithmetic. For deep-tail or very high-``d`` inputs where a
`Float64` jet can overflow or lose precision, pass `BigFloat` coordinates (see
[Precision](#Precision)) — that is the precision fix, independent of which
composition method is in use.

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
`subsetdims` marginal density. Both factors route through the Faà di Bruno tree
walk via the `subsetdims` / `condition` specialisations for this type — no
ForwardDiff for the observed-marginal density nor for the conditional CDF (for
any number of lower-tail coordinates).

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
u = [cdf(S.m[i], x[i]) for i in 1:6]
Cs = SurvivalCopula(Cpart, C)
logpdf(subsetdims(Cpart, O), u[collect(O)]) +
    log(cdf(condition(Cs, O, u[collect(O)]), 1 .- u[collect(C)]))
```

On the data scale, add the observed marginal log densities
``\sum_{i\in O}\log f_i(x_i)`` to this copula-scale contribution.

When a *single* coordinate is in ``C``, `condition(S, O, x_O)` returns a
univariate conditional distribution and you use `logcdf(condition(...), x_C)`
(a scalar `x_C`); when several are in ``C`` it returns a conditional joint
distribution and you use `log(cdf(condition(...), x_C))` as above. With ``C``
empty (all observed) the recipe reduces to the ordinary joint density
`logpdf(S, x)`; with ``O`` empty it reduces to
``\log F(\mathbf x)``. The recipe works for any copula with `condition` /
`subsetdims` support — flat [`ArchimedeanCopula`](@ref) as well as nested trees.

!!! note "Multi-coordinate conditional CDF"
    With two or more lower-tail coordinates the conditional CDF is the mixed
    partial of the nested CDF over the *observed* coordinates. The generic path
    takes this by nesting one `ForwardDiff.derivative` per observed coordinate —
    cost exponential in the number of observed dims, infeasible in high
    dimension. A `_partial_cdf` specialisation routes it instead through the same
    polynomial Faà di Bruno tree walk as the single-coordinate case (selected on
    the conditional copula's concrete nested inner type), for any number of
    lower-tail coordinates.

    At high differentiation order for fast-tail generators the `Float64` sum can
    lose precision; pass `BigFloat` coordinates to recover the exact value (as for
    the density). End-to-end `BigFloat` through `condition()` is not yet enabled —
    upstream stores the conditioning values as `Float64`.
