```@meta
CurrentModule = Copulas
```

# [General Discussion](@id archimedean_copulas_header)

An important parametric class of copulas is the class of Archimedean copulas. To define Archimedean copulas, we must take a look at their *generators*, which are unrelated to spherical generators, and must be $d$-monotone functions. 

## Generators and d-monotony

Archimedean generators can be defined as follows:
> **Definition (Archimedean generator):** A $d$-Archimedean generator is a $d$-monotone function 
>
>$\phi :\mathbb R_+ \to [0,1]$ such that $\phi(0) = 1$ and $\phi(+\infty) = 0$.

where the notion of $d$-monotone function can be defined as follows: 

> **Definition (d-monotony [mcneil2009](@cite)):** A function $\phi$ is said to be $d$-monotone if it has $d-2$ derivatives which satisfy 
>
> $(-1)^k \phi^{(k)} \ge 0 \;\forall k \in \{1,..,d-2\},$ and if $(-1)^{d-2}\phi^{(d-2)}$ is a non-increasing and convex function. 
>
>A function that is $d$-monotone for all $d$ is called **completely monotone**.


In this package, there is an abstract class [`Generator`](@ref) that contains those generators. Many Archimedean generators are already implemented for you ! See [the list of implemented archimedean generator](@ref available_archimedean_models) to get an overview. 

If you do not find the one you need, you may define it yourself by subtyping `Generator`. The API does not ask for much information, which is really convenient. Only the two following methods are required:

* The `ϕ(G::MyGenerator,t)` function returns the value of the archimedean generator itself. 
* The `max_monotony(G::MyGenerator)` returns its maximum monotony. 

Thus, a new generator implementation may simply look like:

```julia
struct MyGenerator{T} <: Generator
    θ::T
end
ϕ(G::MyGenerator,t) = exp(-G.θ * t) # can you recognise this one ?
max_monotony(G::MyGenerator) = Inf
```
!!! tip "Win-Win strategy"
    These two functions are enough to sample the corresponding Archimedean copula (see how in the [Inverse Williamson $d$-transforms](@ref w_trans_section) section of the documentation). However, if you know a bit more about your generator, implementing a few more simple methods can largely fasten the algorithms. You'll find more details on these methods in the [`Generator`](@ref) docstring.

```@docs
Generator
```

## Williamson d-transform

An easy way to construct new $d$-monotonous generators is the use of the Williamson $d$-transform.

> **Definition (Williamson d-transformation):** For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and an integer $d\ge 2$, the Williamson-d-transform of ``X`` is the real function supported on $[0,\infty[$ given by:
>
> $\phi(t) = 𝒲_{d}(X)(t)$
> $=\int_{t}^{\infty} \left(1 - \frac{t}{x}\right)^{d-1} dF(x)$
> $= \mathbb E\left( (1 - \frac{t}{X})^{d-1}_+\right) \mathbb 1_{t > 0} + \left(1 - F(0)\right)\mathbb 1_{t <0}$

In this package, we implemented it through the [`WilliamsonGenerator`](@ref) class. It can be used as follows: 

`WilliamsonGenerator(X::UnivariateRandomVariable, d)`.

This function computes the Williamson d-transform of the provided random variable $X$ using the [WilliamsonTransforms.jl](https://github.com/lrnv/WilliamsonTransforms.jl) package. See [williamson1955multiply, mcneil2009](@cite) for the literature. 

!!! warn "`max_monotony` of Williamson generators"
    The $d$-transform of a positive random variable is $d$-monotonous but not $k$-monotonous for any $k > d$. Its max monotony is therefore $d$. This has a few implications, one of the biggest one is that the $d$-variate Archimedean copula that corresponds has no density. 
    
    More genrally, if you want your Archimedean copula to have a density, you have to use a generator that is more-monotonous that the dimension of your model. 




```@docs
WilliamsonGenerator
```

## [Inverse Williamson d-transform](@id w_trans_section)


The Williamson d-transform is a bijective transformation[^1] from the set of positive random variables to the set of generators, and therefore has an inverse transformation (called, suprisingly, the inverse Williamson $d$-transform) that construct the a positive random variable *R* from a generator $\phi$.

[^1]:

    This bijection is to be taken carefully: the bijection is between random variables *with unit scales* and generators *with common value at 1*, sicne on both rescaling does not change the underlying copula. 

This transformation is implemented through one method in the Generator interface that is worth talking a bit about : `williamson_dist(G::Generator, d)`. This function computes the inverse Williamson d-transform of the d-monotone archimedean generator ϕ, still using the [WilliamsonTransforms.jl](https://github.com/lrnv/WilliamsonTransforms.jl) package. See [williamson1955multiply, mcneil2009](@cite)

To put it in a nutshell, for ``\phi`` a ``d``-monotone archimedean generator, the inverse Williamson-d-transform of ``\\phi`` is the cumulative distribution function ``F`` of a non-negative random variable ``R``, defined by : 

```math
F(x) = 𝒲_{d}^{-1}(\\phi)(x) = 1 - \\frac{(-x)^{d-1} \\phi_+^{(d-1)}(x)}{k!} - \\sum_{k=0}^{d-2} \\frac{(-x)^k \\phi^{(k)}(x)}{k!}
```

The [WilliamsonTransforms.jl](https://github.com/lrnv/WilliamsonTransforms.jl) package implements this transfromation (and its inverse, the Williamson d-transfrom) in all generality. It returns this cumulative distribution function in the form of the corresponding random variable `<:Distributions.ContinuousUnivariateDistribution` from `Distributions.jl`. You may then compute : 
* The cdf via `Distributions.cdf`
* The pdf via `Distributions.pdf` and the logpdf via `Distributions.logpdf`
* Samples from the distribution via `rand(X,n)`


## Archimedean copulas

Let's first define formally archimedean copulas: 

> **Definition (Archimedean copula):** If $\phi$ is a $d$-monotonous Archimedean generator, then the function 
>
>$$C(\bm u) = \phi\left(\sum\limits_{i=1}^d \phi^{-1}(u_i)\right)$$ is a copula. 

There are a few archimedean generators that are worth noting since they correspond to known archimedean copulas familiies: 
* [`IndependentGenerator`](@ref): $\phi(t) =e^{-t} \text{ generates } \Pi$.
* [`ClaytonGenerator`](@ref): $\phi_{\theta}(t) = \left(1+t\theta\right)^{-\theta^{-1}}$ generates the $\mathrm{Clayton}(\theta)$ copula.
* [`GumbelGenerator`](@ref): $\phi_{\theta}(t) = \exp\{-t^{\theta^{-1}}\}$ generates the $\mathrm{Gumbel}(\theta)$ copula.
* [`FrankGenerator`](@ref): $\phi_{\theta}(t) = -\theta^{-1}\ln\left(1+e^{-t-\theta}-e^{-t}\right)$ generates the $\mathrm{Franck}(\theta)$ copula.

There are a lot of others implemented in the package, see our [large list of implemented archimedean generator](@ref available_archimedean_models). 

Archimedean copulas have a nice decomposition, called the Radial-simplex decomposition: 

> **Property (Radial-simplex decomposition [mcneil2008,mcneil2009](@cite):** A $d$-variate random vector $\bm U$ following an Archimedean copula with generator $\phi$ can be decomposed into 
>
> $\bm U = \phi.(\bm S R),$
> where $\bm S$ is uniform on the $d$-variate simplex and $R$ is a non-negative random variable, independent form $\bm S$, defined as the inverse Williamson $d$-transform of $\phi$.  


This is why `williamson_dist(G::Generator,d)` is such an important function in the API: it allows to generator the radial part and sample the Archimedean copula. 

!!! note "Frailty decomposition for completely monotonous generators"
    It is well-known that completely monotone generators are Laplace transforms of non-negative random variables. This gives rise to another decomposition:

    > **Property (Frailty decomposition [hofert2013](@cite):** When $\phi$ is completely monotone, it is the Laplace transform of a non-negative random variable $W$ such that
    >
    >$$\bm U = \phi(\bm Y / W),$$  where $\bm Y$ is a vector of independent and identically distributed (i.i.d.) exponential distributions.

    The link between the distribution of $R$ and the distribution of $W$ can be explicited. We exploit this link and provide the `WilliamsonFromFrailty()` constructor that construct the distribution of $R$ from the distribution of $W$ and returns the corresponding  `WilliamsonGenerator` from the frailty distribution itself. The corresponding ϕ is simply the laplace transform of $W$. This is another potential way of constructing new archimedean copulas !  

```@docs
ArchimedeanCopula
```


```@bibliography
Pages = ["generalities.md"]
Canonical = false
```