```@meta
CurrentModule = Copulas
```
# [Elliptical Copulas](@id elliptical_copulas_header)

## Definition

The easiest families of copulas are the one derived from known families of random vectors, and the first presented one are, generally, the Elliptical families (in particular, the Gaussian and Student families are very standard in the litterature). 

!!! definition "Definition (Spherical and elliptical random vectors):" 
    A random vector $\bm X$ is said to be spherical if for all orthogonal matrix $\bm A \in O_d(\mathbb R)$, $\bm A\bm X \sim \bm X$. 

    For every matrix $\bm B$ and vector $\bm c$, the random vector $\bm B \bm X + \bm c$ is then said to be elliptical.


Spherical random vectors have several interesting properties. First, the shape of the distribution must be the same in every direction since it is stable by rotations. Moreover, their characteristic functions (c.f.) only depend on the norm of their arguments. Indeed, for any $\bm A \in O_d(\mathbb R)$, 
```math
\phi(\bm t) = \mathbb E\left(e^{\langle \bm t, \bm X \rangle}\right)= \mathbb E\left(e^{\langle \bm t, \bm A\bm X \rangle}\right) = \mathbb E\left(e^{\langle \bm A\bm t, \bm X \rangle}\right) = \phi(\bm A\bm t).
```

We can therefore express this characteristic function as $\phi(\bm t) = \psi(\lVert \bm t \rVert_2^2)$, where $\psi$ is a function that characterizes the spherical family, called the *generator* of the family. Any characteristic function that can be expressed as a function of the norm of its argument is the characteristic function of a spherical random vector, since $\lVert \bm A \bm t \rVert_2 = \lVert \bm t \rVert_2$ for any orthogonal matrix $\bm A$. 

This class contains the (multivariate) Normal and Student distributions, and it is easy to construct others if needed. This is a generalization of the family of Gaussian random vectors, and they benefit from several nice properties of the former, among which, particularly interesting, the stability by convolution. Indeed, convolutions correspond to product of characteristic functions, and
```math
\phi(\bm t) = \prod_{i=1}^n \phi_i(\bm t) = \prod_{i=1}^n \psi_i(\lVert \bm t \rVert_2^2) = \psi(\lVert \bm t \rVert_2^2),
```
which is still a function of only the norm of $\bm t$. 

To fix ideas, for Gaussian random vectors, $\psi(t) = e^{-\frac{t^2}{2}}$.

!!! note "Sampling with `Distributions.jl`"
    Elliptical random vectors in the Gaussian and Student families are available from `Distributions.jl`:

    ```@example 3
    using Distributions
    Σ = [1 0.5
        0.5 1] # variance-covariance matrix.
    ν = 3 # number of degrees of freedom for the student.
    N = MvNormal(Σ)
    ```

    ```@example 3
    T = MvTDist(ν,Σ)
    ```



Elliptical copulas are simply copulas of elliptical distributions. This simplicity of definition is paid for in the expression of the copulas itself: the obtained function has usually no better expression than: 
```math
C = F \circ (F_1^{-1},...,F_d^{-1}),
```
where $F_i^{-1}$ denotes the almost-inverse of $F_i$, that is: 
```math
\forall u \in [0,1],\;F_i^{-1}(u) = \inf\left\{x :\, F_i(x) \ge u\right\},
```
and $F_i$ is usually hard to express from the elliptical assumptions.

Moreover, the form of dependence structures that can be reached inside this class is restricted. The elliptical copulas are parametrized by the corresponding univariate spherical generator and a correlation matrix, which is a very simple structure. See also [frahm2003,gomez2003,cote2019](@cite) for details on these copulas. 

On the other hand, there exist performant estimators of high-dimensional covariance matrices, and a large theory is built on the elliptical assumption of high dimensional random vectors, see e.g., [elidan2013,friedman2010,muller2019](@cite) among others. See also [derumigny2022](@cite) for a recent work on nonparametric estimation of the underlying univariate spherical distribution. 


!!! note "Note on internal implementation"
    If the exposition we just did on characteristic functions of Elliptical random vectors is fundamental to the definition of elliptical copulas, the package does not use this at all to function, and rather rely on the existence of multivariate and corresponding univariate families of distributions in `Distributions.jl`. 


You can obtain these elliptical copulas by the following code: 
```julia
using Copulas
Σ = [1 0.5
     0.5 1] # variance-covariance matrix.
ν = 3 # number of degrees of freedom for the student.
C_N = GaussianCopula(Σ)
C_T = TCopula(ν,Σ)
```

As already stated, the underlying code simply applies Sklar. In all generalities, you may define another elliptical copula by the following structure: 

```julia
struct MyElliptical{d,T} <: EllipticalCopula{d,T}
    θ:T
end
U(::Type{MyElliptical{d,T}}) where {d,T} # Distribution of the univaraite marginals, Normal() for the Gaussian case. 
N(::Type{MyElliptical{d,T}}) where {d,T} # Distribution of the mutlivariate random vector, MvNormal(\Sigma) for the Gaussian case. 
```

However, not much other cases than the Gaussian and Elliptical one are really used in the literature.

## Examples

To construct, e.g., a Student copula, you need to provide the Correlation matrix and the number of degree of freedom, as follows: 

```@example 4
using Copulas, Distributions
Σ = [1 0.5
    0.5 1] # variance-covariance matrix.
ν = 3 # number of degrees of freedom
C = TCopula(ν,Σ)
```

You can sample it and compute its density and distribution functions via the standard interface. We could try to fit a GaussianCopula on the sampled data, even if we already know that the tails will not be properly taken into account: 

```@example 4
u = rand(C,1000)
Ĉ = fit(GaussianCopula,u) # to fit on the sampled data. 
```

We see that the estimation we have on the correlation matrix is quite good, but rest assured that the tails of the distributions are not the same at all. To see that, let's plot the lower tail function (see [nelsen2006](@cite)) for both copulas: 

```@example 4
using Plots
chi(C,u) = 2 * log(1-u) / log(1 - 2u + cdf(C,[u,u])) -1
u = 0.5:0.03:0.99
plot(u,  chi.(Ref(C),u), label="True student copula")
plot!(u, chi.(Ref(Ĉ),u), label="Estimated Gaussian copula")
``` 

### Visual: Gaussian vs Student (same correlation)

```@example 4
using Plots
Σ = [1 0.7; 0.7 1]
ν = 4
CG = GaussianCopula(Σ)
CT = TCopula(ν, Σ)
UG = rand(CG, 3000)
UT = rand(CT, 3000)
plt = plot(layout=(1,2), size=(800, 350))
scatter!(plt[1], UG[1,:], UG[2,:]; ms=1.8, alpha=0.5, xlim=(0,1), ylim=(0,1), title="Gaussian copula", label=false)
scatter!(plt[2], UT[1,:], UT[2,:]; ms=1.8, alpha=0.5, xlim=(0,1), ylim=(0,1), title="Student copula (ν=4)", label=false)
plt
```

### Density heatmaps (cdf/pdf) on the unit square

```@example 4
grid = range(0.01, 0.99; length=100)
ZG = [pdf(CG, [u,v]) for u in grid, v in grid]
ZT = [pdf(CT, [u,v]) for u in grid, v in grid]
plot(heatmap(grid, grid, ZG'; title="Gaussian density", aspect_ratio=1, c=:viridis),
    heatmap(grid, grid, ZT'; title="Student density", aspect_ratio=1, c=:viridis),
    layout=(1,2), size=(800,330))
```

### Conditional on original scale via SklarDist

```@example 4
XG = SklarDist(CG, (Normal(), Normal()))
XT = SklarDist(CT, (Normal(), Normal()))
X1G = condition(XG, 2, 0.0)
X1T = condition(XT, 2, 0.0)
xgrid = range(quantile(X1T, 0.001), quantile(X1T, 0.999); length=401)
plot(xgrid, Distributions.cdf.(Ref(X1G), xgrid); label="Gaussian", xlabel="x", ylabel="cdf",
    title="F_{X1|X2=0}")
plot!(xgrid, Distributions.cdf.(Ref(X1T), xgrid); label="Student")
```

## Implementation

```@docs
EllipticalCopula
```


## See also

- Bestiary: [Implemented Elliptical copulas](@ref elliptical_cops)
- Manual: [Getting Started](@ref), [Sklar's Distribution](@ref)


```@bibliography
Pages = [@__FILE__]
Canonical = false
```
