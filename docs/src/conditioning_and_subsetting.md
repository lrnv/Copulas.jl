```@meta
CurrentModule = Copulas
```

# Conditioning and subsetting copulas

## Conditioning

This page introduces conditional distributions under a copula model and shows how to
construct them programmatically using `condition`.

The same interface works either on the uniform scale (copula only) or on the original
scale (via `SklarDist`).

### Overview

Given a D-variate copula `C` (the joint CDF of `U ∈ [0,1]^D` with uniform marginals),
consider a split of the indices into `I = setdiff(1:D, J)` and `J ⊂ {1,…,D}` with `|J| = p`.

- The conditional joint distribution on the uniform scale is the function

```math
H_{I\mid J}(\mathbf u_I\mid\mathbf u_J)
:= \frac{\partial^p C(\mathbf u_I, \mathbf u_J)}{\partial \mathbf u_J}
\Big/ \frac{\partial^p C(\mathbf 1_I, \mathbf u_J)}{\partial \mathbf u_J},
```

which is a proper distribution on `[0,1]^{|I|}`.

- Each conditional marginal (uniform scale) can be expressed as a “distortion”
`H_{i|J}(· | u_J)` acting on `[0,1]`:

```math
H_{i\mid J}(u\mid\mathbf u_J)
:= \frac{\partial^p C(\mathbf u^{(i)}(u), \mathbf u_J)}{\partial \mathbf u_J}
\Big/ \frac{\partial^p C(\mathbf 1, \mathbf u_J)}{\partial \mathbf u_J},
```

where `\mathbf u^{(i)}(u)` has coordinate `u` at index `i` and `1` elsewhere in `I`.

- On the original scale for a compound distribution `X = SklarDist(C, (F_1,…,F_D))`,
conditioning on values `x_J` is obtained by mapping to uniforms `u_J = (F_j(x_j))_{j∈J}`
then pushing forward each distortion through the corresponding marginal:

```math
F_{X_i\mid X_J}(x\mid \mathbf x_J) = H_{i\mid J}\big(F_i(x)\mid \mathbf u_J\big).
```

The copula of the conditional vector `U_I | U_J = u_J` is a genuine copula denoted
`C_{I|J}(·|u_J)`, which is the copula of `H_{I|J}`; in the package this is materialized
by `ConditionalCopula(C, J, u_J)` and used internally by `condition`.

### Using `condition`

- `condition(C::Copula, js, u_js)` returns the conditional distribution on the uniform
  scale for `I = setdiff(1:D, js)`.
  - If `length(I) == 1`, the result is a `Distortion` on `[0,1]`.
  - Otherwise, it is a `SklarDist(ConditionalCopula, distortions)` on `[0,1]^{|I|}`.
- `condition(X::SklarDist, js, x_js)` returns the conditional distribution on the
  original scale by pushing forward each distortion through the corresponding marginal.
- A specialization `condition(C::Copula{2}, j, u_j)` provides a fast path for the
  common 2D, single-index case.

!!! tip "Performance"
    For best performance, pass `js` and `u_js` as NTuples so that `p = length(js)` is known at compile time.

#### Example (uniform scale, D = 2)

```@example cond1
using Copulas, Distributions
C = ClaytonCopula(2, 1.5)
D = condition(C, 2, 0.3)  # distortion for U₁ | U₂ = 0.3
cdf(D, 0.7), quantile(D, 0.9)
```

Here is the full CDF of this distortion on [0,1]:

```@example cond1
using Plots
ts = range(0.0, 1.0; length=401)
plt = plot(ts, cdf.(Ref(D), ts);
          xlabel="u", ylabel="H_{1|2}(u | 0.3)",
          title="Conditional CDF on the uniform scale",
          legend=false)
plt
```

Overlay the empirical CDF from Monte Carlo sampling via the distortion’s quantile (exact conditional sampling on the uniform scale):

```@example cond1
using StatsBase
N = 2000
αs = rand(N)
us = Distributions.quantile.(Ref(D), αs)
ECDF = ecdf(us)
plot!(ts, ECDF.(ts); seriestype=:steppost, label="empirical", alpha=0.6, color=:black)
plot!(ts, cdf.(Ref(D), ts); label="analytic", color=:blue)
plt
```

#### Example (original scale via SklarDist)

```@example cond1
C = ClaytonCopula(2, 1.5)
X = SklarDist(C, (Normal(), Normal()))
X1_given_X2 = condition(X, 2, 0.0) # distribution of X₁ | X₂ = 0.0
cdf(X1_given_X2, 1.0), quantile(X1_given_X2, 0.95)
```

Overlay analytic vs empirical CDF on the original scale:

```@example cond1
xs = rand(X1_given_X2, 2000)
Fx = ecdf(xs)
xs_grid = range(quantile(X1_given_X2, 0.001), quantile(X1_given_X2, 0.999); length=401)
plot(xs_grid, Distributions.cdf.(Ref(X1_given_X2), xs_grid);
  xlabel="x", ylabel="F_{X₁|X₂}(x|0)", title="Original-scale conditional CDF", label="analytic")
plot!(xs_grid, Fx.(xs_grid); seriestype=:steppost, label="empirical", alpha=0.6, color=:black)
```

#### Example (uniform scale, D = 3, |J| = 2)

```@example cond1
C = ClaytonCopula(3, 1.2)
H = condition(C, (2, 3), (0.25, 0.8))
cdf(H, [0.4, 0.6])
```

You can also visualize a marginal conditional distortion in a higher-dimensional example.
For instance, in 3D, conditioning on two coordinates returns a 1D distortion for the
remaining coordinate; we can plot its CDF as well:

```@example cond1
C3 = ClaytonCopula(3, 1.2)
D3 = condition(C3, (1, 2), (0.25, 0.8))  # distortion for U₃ | (U₁,U₂) = (0.25,0.8)
ts = range(0.0, 1.0; length=401)
plot(ts, cdf.(Ref(D3), ts);
  xlabel="u", ylabel="H_{3|1,2}(u | 0.25,0.8)",
  title="Conditional CDF in 3D (uniform scale)",
  legend=false)
```

Overlay with an empirical CDF obtained via the distortion’s quantile:

```@example cond1
αs = rand(2000)
u3s = Distributions.quantile.(Ref(D3), αs)
EC3 = ecdf(u3s)
plot!(ts, EC3.(ts); seriestype=:steppost, label="empirical", alpha=0.6, color=:black)
```

### Special cases

- Independence: distortions are the identity (`u ↦ u`) and the conditional copula is
  independence; conditioning leaves marginals unchanged.

- Gaussian: if `U = Φ(Z)` with `Z ~ Normal(0, Σ)`, then `Z_I | Z_J` is normal with the
  usual conditional mean/covariance. On the uniform scale this yields a closed-form
  `GaussianDistortion(μ_z, σ_z)` for each marginal; on the original scale, applying it
  to `Normal(μ, σ)` returns `Normal(μ + σ μ_z, σ σ_z)`.

- Archimedean: for generator `ϕ` and `S_J = ∑_{j∈J} ϕ^{-1}(u_j)`, the uniform-scale
  distortion for any `i ∉ J` is

```math
H_{i\mid J}(u\mid \mathbf u_J) = \frac{\,\,\, ϕ^{(p)}\big(ϕ^{-1}(u) + S_J\big)}{ϕ^{(p)}(S_J)}.
```

This yields a fast-path for Archimedean conditioning without numerical differentiation.

!!! tip "Missing fast-paths?"
    If you find a conditional that should admit a faster closed-form or semi-analytic
    path but currently falls back to the generic construction, please open an issue, 
    we’ll happily implement it :)

### Relation to the conditional copula

The conditional copula `C_{I|J}(·|u_J)` is the copula of the conditional distribution
`H_{I|J}(·|u_J)`. In the implementation it is represented by
`ConditionalCopula(C, js, u_js)` and is used as the copula of the conditional joint
when `|I| > 1`. When `condition` returns a `SklarDist` (i.e., `|I| > 1`), you can
access this copula directly via the `.C` field of the returned object:

```julia
H = condition(C, js, u_js)  # returns a SklarDist
H.C                    # the conditional copula
H.m                    # a Tuple containing the distorded marginals. 
```

### Implementation

```@docs
condition
Distortion
DistortionFromCop
DistortedDist
ConditionalCopula
```

On top of this main implementation, there are fast-paths for some copulas that are implemented.

### See also

- [`condition`](@ref) — reference documentation with all calling syntaxes
- [`SklarDist`](@ref) — compound distributions via Sklar’s theorem
- [`rosenblatt`](@ref) — sequential transforms (related but different)


## Subsetting

Subsetting extracts the dependence structure among a subset of coordinates. Given a
copula `C` of dimension `d` and an index tuple `dims::NTuple{p,Int}`, the function
`subsetdims` returns a copula on those `p` dimensions that preserves the original
dependence restricted to `dims`.

There are two entry points:

- `subsetdims(C::Copula, dims)` returns a `Copula{p}` (or `Uniform()` when `p == 1`).
- `subsetdims(X::SklarDist, dims)` returns a `SklarDist` with copula `subsetdims(C, dims)`
  and marginals `(m[i] for i in dims)`.

Internally, we materialize subsetting with a small wrapper type `SubsetCopula{p}(C, dims)`
which delegates `cdf`, `pdf`, and sampling to the base copula by saturating non-selected
coordinates at 1. For many families we provide specialized constructors that return the
natural reduced-parameter form instead of a wrapper (e.g., elliptical copulas return the
appropriate submatrix, Archimedean keeps the same generator with reduced dimension, etc.).

### Usage

```@example subset1
using Copulas, Distributions
C = GaussianCopula([1.0 0.6 0.2; 0.6 1.0 0.3; 0.2 0.3 1.0])
S = subsetdims(C, (1,3))    # 2D copula on coordinates 1 and 3
length(S), typeof(S)
```

```@example subset1
X = SklarDist(C, (Normal(), Normal(1,2), LogNormal()))
X13 = subsetdims(X, (1,3))  # keeps marginals (Normal(), LogNormal()) and reduces the copula
length(X13.C), length(X13.m)
```

### What gets returned?

The result depends on the family:

- Elliptical (Gaussian, t): specialized `Copula{p}` with correlation matrix equal to the
  `p×p` submatrix of the original; in the t case, degrees of freedom are unchanged for
  subsetting.
- Archimedean: same generator, reduced dimension.
- Independence: `IndependentCopula(p)`.
- Frechet bounds: `MCopula(p)`; for `WCopula` (only defined in 2D) you must subset to `p=2`.
- Empirical: keeps only the selected rows of the pseudo-observations.
- FGM: builds the reduced parameter vector θ′ consisting of coefficients whose index
  sets are fully contained in `dims`, ordered by increasing order k=2..p and lexicographic
  within each order, matching the original convention.
- Survival: first subsets the base copula, then remaps flip indices to positions within
  the new coordinate set; flips not in `dims` are dropped.

If no specialization exists, a `SubsetCopula` wrapper is returned. It’s fully usable
and equivalent from an API perspective; specialized forms simply yield better performance
and clearer display.

### Interactions with conditioning

Subsetting and conditioning commute in the obvious way: conditioning on coordinates `J`
and then extracting a subset of the remaining coordinates is equivalent to subsetting
the base copula first and then conditioning on the corresponding indices. In code, if
`S = subsetdims(C, dims)`, conditioning on indices `js` within `S` is implemented by
mapping `js` to indices in the base copula and delegating to `ConditionalCopula(C, ·, ·)`;
the resulting conditional copula of `S` is either the base conditional copula (when all
remaining coordinates are kept) or a further `SubsetCopula` of it.

### Examples

```@example subset1
# Archimedean example
C = ClaytonCopula(3, 2.0)
S = subsetdims(C, (1,2))        # still a ClaytonCopula with the same parameter
rand(S, 3)                      # sample 3 points
cdf(S, [0.7, 0.9])
```

```@example subset1
# Survival example with flips remapped
base = GaussianCopula([1.0 0.7 0.2; 0.7 1.0 0.1; 0.2 0.1 1.0])
S = SurvivalCopula(base, (2,))
S13 = subsetdims(S, (1,3))      # flip on 2 drops; no flips remain
typeof(S13), S13 isa SurvivalCopula
```

```@docs
Copulas.subsetdims
Copulas.SubsetCopula
```

## Rosenblatt transformations

### Definition and usefulness

!!! definition "Definition (Rosenblatt transformation):"
    The Rosenblatt transformation considers a random vector ``X`` distributed according to a certain multivariate cumulative distribution function ``F_{X}(x)``, and maps it back to a uniform distribution on the unit hypercube.

    More formally, consider the map ``R_X(x)`` defined as follows:

    ```math
    R_X(x_1, ..., x_d) = (r_1 = F_{X_1}(x_1), r_2 = F_{X_2 | X_1}(x_2 | x_1), ..., r_{d} = F_{X_d | X_1, ..., X_{d-1}}(x_d | x_1, ..., x_{d-1}))
    ```

References:
* [rosenblatt1952](@cite) Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics, 23(3), 470-472.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press. (Section 2.10)
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.

In certain circumstances, in particular for Archimedean copulas, this map simplifies to tractable expressions. It has a few nice properties:

* ``R_X(X) \sim \texttt{Uniform(Unit Hypercube)}``
* ``R_X`` is a bijection. 

These two properties are leveraged in some cases to construct the inverse Rosenblatt transformations, which map random noise to proper samples from the copula. In some cases, this is the best sampling algorithm available. 

### Implementation

As soon as the random vector ``X`` is represented by an object `X` that subtypes `SklarDist` or `Copula`, you have access to the `rosenblatt(X, x)` and `inverse_rosenblatt(X, x)` operators, which both have a straightforward interpretation from their names. 

```@docs
rosenblatt
inverse_rosenblatt
```

Once again, since the rosenblatt transform leverages the conditioning mechanisme, some fast-paths might be missing in the implementation.

### Sanity check plot

You can validate that the Rosenblatt transform maps samples to independent uniforms by checking the marginal ECDFs against the 45° line.

```@example rosen1
using Copulas, Plots, StatsBase
# pick a nontrivial copula
C = ClaytonCopula(3, 1.5)

# draw samples and apply Rosenblatt transform coordinate-wise
U = rand(C, 3000)                 # size (3, N)
S = reduce(hcat, (rosenblatt(C, U[:, i]) for i in 1:size(U, 2)))  # size (3, N)

ts = range(0.0, 1.0; length=401)
layout = @layout [a b c]
plt = plot(layout=layout, size=(900, 280), legend=false)
for k in 1:3
  Ek = ecdf(S[k, :])
  plot!(plt[k], ts, Ek.(ts); seriestype=:steppost, color=:black,
      title="ECDF of s$(k)", xlabel="u", ylabel="ECDF")
  plot!(plt[k], ts, ts; color=:blue, alpha=0.7)
end
plt
```

```@bibliography
Pages = [@__FILE__]
Canonical = false
```

