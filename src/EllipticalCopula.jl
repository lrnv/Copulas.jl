"""
    EllipticalCopula{d,MT}

This is an abstract type. It implements an interface for all Elliptical copulas. We construct internally elliptical copulas using the sklar's theorem, by considering the copula ``C`` to be defined as : 

```math
C = F \\circ (F_1^{-1},...,F_d^{-1}),
```

where ``F`` and ``F_1,...,F_d`` are respectively the multivariate distribution function of some elliptical random vector and the univariate distribution function of its marginals.  For a type `MyCop <: EllipitcalCopula`, it is necessary to implement the following methods: 

- `N(::Type{MyCOp})`, returning the constructor of the elliptical random vector from its correlation matrix. For example, `N(GaussianCopula)` simply returns `MvNormal` from `Distributions.jl`.
- `U(::Type{MyCOp})`, returning the constructor for the univariate marginal, usually in standardized form. For example, `U(GaussianCopula)` returns `Normal` from `Distributions.jl`.

From these two functions, the abstract type provides a fully functional copula. 

# Details 

Recall the definition of spherical random vectors: 

> **Definition (Spherical and elliptical random vectors):** A random vector ``\\bm X`` is said to be spherical if for all orthogonal matrix ``\\bm A \\in O_d(\\mathbb R)``, ``\\bm A\\bm X \\sim \\bm X``. 
>
> For every matrix ``\\bm B`` and vector ``\\bm c``, the random vector ``\\bm B \\bm X + \\bm c`` is then said to be elliptical.


Recall that spherical random vectors are random vectors which characteristic functions (c.f.) only depend on the norm of their arguments. Indeed, for any ``\\bm A \\in O_d(\\mathbb R)``, 
```math
\\phi(\\bm t) = \\mathbb E\\left(e^{\\langle \\bm t, \\bm X \\rangle}\\right)= \\mathbb E\\left(e^{\\langle \\bm t, \\bm A\\bm X \\rangle}\\right) = \\mathbb E\\left(e^{\\langle \\bm A\\bm t, \\bm X \\rangle}\\right) = \\phi(\\bm A\\bm t).
```

We can therefore express this characteristic function as ``\\phi(\\bm t) = \\psi(\\lVert \\bm t \\rVert_2^2)``, where ``\\psi`` is a function that characterizes the spherical family, called the *generator* of the family. Any characteristic function that can be expressed as a function of the norm of its argument is the characteristic function of a spherical random vector, since ``\\lVert \\bm A \\bm t \\rVert_2 = \\lVert \\bm t \\rVert_2`` for any orthogonal matrix ``\\bm A``. 

However, note that this is not how the underlying code is working, we do not check for validity of the proposed generator (we dont even use it). You can construct such an elliptical family using simply Sklar: 

```julia
struct MyElliptical{d,T} <: EllipticalCopula{d,T}
    θ:T
end
U(::Type{MyElliptical{d,T}}) where {d,T} # Distribution of the univaraite marginals, Normal() for the Gaussian case. 
N(::Type{MyElliptical{d,T}}) where {d,T} # Distribution of the mutlivariate random vector, MvNormal(C.Σ) for the Gaussian case. 
```

These two functions are enough to implement the rest of the interface. 

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
abstract type EllipticalCopula{d,MT} <: Copula{d} end
Base.eltype(C::CT) where CT<:EllipticalCopula = Base.eltype(N(CT)(C.Σ))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT <: EllipticalCopula}
    Random.rand!(rng,N(CT)(C.Σ),x)
    x .= clamp.(Distributions.cdf.(U(CT),x),0,1)
    return x
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, A::DenseMatrix{T}) where {T<:Real, CT<:EllipticalCopula}
    # More efficient version that precomputes stuff:
    n = N(CT)(C.Σ)
    u = U(CT)
    Random.rand!(rng,n,A)
    A .= clamp.(Distributions.cdf.(u,A),0,1)
    return A
end
function Distributions._logpdf(C::CT, u) where {CT <: EllipticalCopula}
    x = StatsBase.quantile.(U(CT),u)
    return Distributions.logpdf(N(CT)(C.Σ),x) - sum(Distributions.logpdf.(U(CT),x))
end
function make_cor!(Σ)
    # Verify that Σ is a correlation matrix, otherwise make it so : 
    d = size(Σ,1)
    σ = [1/sqrt(Σ[i,i]) for i in 1:d]
    for i in 1:d
        for j in 1:d
            Σ[i,j] *= σ[i] .* σ[j]
        end
    end
end
