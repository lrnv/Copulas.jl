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

!!! definition "Spherical and elliptical random vectors"
    A random vector ``\\boldsymbol X`` is said to be spherical if for all orthogonal matrix ``\\boldsymbol A \\in O_d(\\mathbb R)``, ``\\boldsymbol A\\boldsymbol X \\sim \\boldsymbol X``. 

    For every matrix ``\\boldsymbol B`` and vector ``\\boldsymbol c``, the random vector ``\\boldsymbol B \\boldsymbol X + \\boldsymbol c`` is then said to be elliptical.


Recall that spherical random vectors are random vectors which characteristic functions (c.f.) only depend on the norm of their arguments. Indeed, for any ``\\boldsymbol A \\in O_d(\\mathbb R)``, 
```math
\\phi(\\boldsymbol t) = \\mathbb E\\left(e^{\\langle \\boldsymbol t, \\boldsymbol X \\rangle}\\right)= \\mathbb E\\left(e^{\\langle \\boldsymbol t, \\boldsymbol A\\boldsymbol X \\rangle}\\right) = \\mathbb E\\left(e^{\\langle \\boldsymbol A\\boldsymbol t, \\boldsymbol X \\rangle}\\right) = \\phi(\\boldsymbol A\\boldsymbol t).
```

We can therefore express this characteristic function as ``\\phi(\\boldsymbol t) = \\psi(\\lVert \\boldsymbol t \\rVert_2^2)``, where ``\\psi`` is a function that characterizes the spherical family, called the *generator* of the family. Any characteristic function that can be expressed as a function of the norm of its argument is the characteristic function of a spherical random vector, since ``\\lVert \\boldsymbol A \\boldsymbol t \\rVert_2 = \\lVert \\boldsymbol t \\rVert_2`` for any orthogonal matrix ``\\boldsymbol A``. 

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
    d = length(C)
    (u==zeros(d) || u==ones(d)) && return Inf 
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

# ——————————————————————————————————————————————————————————
# Shared correlation-parameterization helpers (LKJ/partial corr)
# Map between correlation matrices and unconstrained vectors α ∈ ℝ^{d(d-1)/2}

# Gaussian / t (pareado por pares)

@inline function _unbound_corr_params(d::Int, Σ::AbstractMatrix)
    Lc = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ), check=true).L
    T = eltype(Σ)
    α = Vector{T}(undef, d*(d-1)÷2)
    k = 1
    @inbounds for i in 2:d
        denom = one(T)
        for j in 1:i-1
            z = Lc[i,j] / denom
            ϵ = sqrt(eps(T))
            z = clamp(z, -one(T) + ϵ, one(T) - ϵ)
            α[k] = atanh(z)
            k += 1
            denom *= sqrt(max(zero(T), one(T) - z*z))
        end
    end
    return α
end

@inline function _rebound_corr_params(d::Int, α::AbstractVector{T}) where {T}
    L = Matrix{T}(LinearAlgebra.I, d, d)
    @inbounds begin
        L[1,1] = one(T)
        k = 1
        for i in 2:d
            denom = one(T)
            for j in 1:i-1
                z = tanh(α[k]); k += 1
                L[i,j] = z * denom
                denom *= sqrt(max(zero(T), one(T) - z*z))
            end
            L[i,i] = denom
        end
    end
    Σ = L * L'
    Σ = (Σ + Σ')/2
    return Σ
end

# ===================== Φ y Φ^{-1} (erfc/erfcinv) =====================
@inline Φinv(p::T) where T = @fastmath sqrt(T(2)) * SpecialFunctions.erfcinv(T(2) * (T(1) - p))
@inline Φ(x::T) where T = @fastmath T(0.5) * SpecialFunctions.erfc(-x / sqrt(T(2)))
@inline φ(x::T) where T = @fastmath inv(sqrt(T(2π))) * exp(-x*x / T(2))

# Richtmyer Generators (shared)
@inline _δ(::Type{T}) where {T<:Real} = T(sqrt(eps(T)))
@inline richtmyer_roots(T, n) = sqrt.(T.(Float64.(Primes.primes(1, max(n-1, Int(floor(5n*log(n+1)/4)))))))[1:n-1]
function _chlrdr_orthant!(R::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    @boundscheck (size(R,1) == size(R,2) == length(b)) || throw(DimensionMismatch())
    n = length(b)
    c = R           # trabajamos in-place
    bp = b
    y = zeros(T, n)

    ϵ  = eps(T)
    ep = T(1e-10)

    @inbounds for k in 1:n
        im = k
        ckk = zero(T)
        dem = one(T)
        bm_tilde = zero(T)

        # --- elegir pivote: min Φ(b̃) ---
        for i in k:n
            cii = c[i,i]
            if cii > ϵ
                cii_sqrt = sqrt(cii)
                s = zero(T)
                if k > 1
                    @simd for j in 1:(k-1)
                        s += c[i,j]*y[j]
                    end
                end
                b_tilde = (bp[i] - s) / cii_sqrt
                de = Φ(b_tilde)
                if de <= dem
                    dem = de; ckk = cii_sqrt; im = i; bm_tilde = b_tilde
                end
            end
        end

        # --- permutar im ↔ k ---
        if im > k
            c[im,im], c[k,k] = c[k,k], c[im,im]
            bp[im],  bp[k]  = bp[k],  bp[im]
            if k > 1
                @simd for j in 1:(k-1)
                    c[im,j], c[k,j] = c[k,j], c[im,j]
                end
            end
            if im < n
                @simd for j in (im+1):n
                    c[j,im], c[j,k] = c[j,k], c[j,im]
                end
            end
            if k < im - 1
                for j in (k+1):(im-1)
                    c[j,k], c[im,j] = c[im,j], c[j,k]
                end
            end
        end

        # anular parte superior de la fila k
        if k < n
            @simd for j in (k+1):n
                c[k,j] = zero(T)
            end
        end

        # --- actualización y y[k] ---
        if ckk > k*ep
            c[k,k] = ckk
            inv_ckk = one(T)/ckk
            for i in (k+1):n
                c[i,k] *= inv_ckk
            end
            # actualización de rango-1 por doble bucle
            for i in (k+1):n
                cik = c[i,k]
                @simd for j in (k+1):i
                    c[i,j] -= cik * c[j,k]
                end
            end

            # --- razón de Mills estable ---
            if dem > ep  # dem ≈ Φ(b̃)
                if bm_tilde < -T(10)
                    # límite inferior: Φ(b̃) ~ 0 → usar aproximación de Mills
                    y[k] = bm_tilde  # E[Z | Z ≤ b̃] ≈ b̃
                else
                    # fórmula exacta y estable para todo b̃ ∈ [-10, +∞)
                    @fastmath y[k] = -φ(bm_tilde) / dem
                end
            else
                # caso degenerado (Φ(b̃) ≈ 0)
                y[k] = bm_tilde
            end
        else
            # columna singular
            @simd for i in k:n
                c[i,k] = zero(T)
            end
            y[k] = zero(T)
        end
    end

    return (c, bp)
end
# Generic kernel: assumes that rfill!(rvec) writes per-sample scales
function qmc_orthant_core!(ch::AbstractMatrix{T}, bs::AbstractVector{T}; m::Integer=10_000, r::Integer=12,
    rng::Random.AbstractRNG=Random.default_rng(), fill_w!::Function = (w::AbstractVector{T}, _j, _nv, _δ::T, _rng)->fill!(w, one(T))) where {T}

    n = length(bs)
    (size(ch,1) == size(ch,2) == n) || throw(DimensionMismatch())

    q  = richtmyer_roots(T, n)               # Richtmyer √primes (n-1)
    nv = max(div(m, r), 1)

    y  = zeros(T, nv, n-1)
    pv = Vector{T}(undef, nv)
    dc = Vector{T}(undef, nv)
    tv = Vector{T}(undef, nv)
    w  = Vector{T}(undef, nv)

    δ = _δ(T)
    p = zero(T); e = zero(T)

    @inbounds for j in 1:r
        # generate scales w[k] for this replicate (t: random; normal: w=1)
        fill_w!(w, j, nv, δ, rng)

        # first coordinate: P(Z1 ≤ w*b1)
        @simd for k in 1:nv
            d1k = Φ((bs[1]/w[k]) / ch[1,1])
            pv[k] = d1k
            dc[k] = d1k
        end

        # dimensions 2..n
        for i in 2:n
            qi = q[i-1]; xr = rand(rng, T)
            @inbounds @simd for k in 1:nv
                t = k*qi + xr; t -= floor(t)
                u = abs(2*t - one(T)) * dc[k]         # u ∈ (0,1)
                p_safe = clamp(u, δ, one(T)-δ)
                y[k, i-1] = Φinv(p_safe)
            end
            LinearAlgebra.mul!(tv, @view(y[:, 1:i-1]), @view(ch[i, 1:i-1]))
            ci_inv = inv(ch[i,i]); bi = bs[i]
            @inbounds @simd for k in 1:nv
                val = Φ( (bi/w[k] - tv[k]) * ci_inv )
                pv[k] *= val
                dc[k]  = val
            end
        end

        pj = sum(pv) / T(nv)
        dm = (pj - p) / T(j)
        p  += dm
        e   = (T(j) - 2)*e/T(j) + dm*dm
    end

    return (p, r==1 ? zero(T) : 3*sqrt(e))
end

