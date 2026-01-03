module HCubature

using LinearAlgebra: norm
import Combinatorics, DataStructures

export hcubature

# Minimal n-dimensional (n ≥ 2) Genz–Malik adaptive cubature using standard Arrays.
# Stripped-down version of https://github.com/JuliaMath/HCubature.jl, they get the credit (MIT licensed)


# Build direction vectors for the Genz–Malik rule (as ordinary vectors)
function _combos(n::Integer, k::Integer, λ::T) where {T<:Real}
    idxs = Combinatorics.combinations(1:n, k)
    pts = Vector{NTuple{n,T}}(undef, length(idxs))
    @inbounds for (i, c) in enumerate(idxs)
        v = fill(zero(T), n)
        for j in c
            v[j] = λ
        end
        pts[i] = Tuple(v)
    end
    return pts
end

function _signcombos(n::Integer, k::Integer, λ::T) where {T<:Real}
    idxs = Combinatorics.combinations(1:n, k)
    twoᵏ = 1 << k
    pts = Vector{NTuple{n,T}}(undef, length(idxs) * twoᵏ)
    out = 1
    @inbounds for c in idxs
        v = fill(zero(T), n)
        for j in c
            v[j] = λ
        end
        pts[out] = Tuple(v)
        # use gray code to flip one sign at a time
        gray = 0
        for s = 1:twoᵏ-1
            gray′ = s ⊻ (s >> 1)
            flip_idx = c[trailing_zeros(gray ⊻ gray′) + 1]
            gray = gray′
            v[flip_idx] = -v[flip_idx]
            pts[out + s] = Tuple(v)
        end
        out += twoᵏ
    end
    return pts
end

struct GenzMalik{n,T<:Real}
    p::NTuple{4,Vector{NTuple{n,T}}} # direction points as tuples
    w::NTuple{5,T}
    w′::NTuple{4,T}
end

const _gm_cache = Dict{Tuple{Int,DataType}, Any}()
const _gm_lock = ReentrantLock()

function _GenzMalik(n::Int, ::Type{T}=Float64) where {T<:Real}
    n < 2 && throw(ArgumentError("invalid dimension $n: Genz–Malik requires n ≥ 2"))

    λ₄ = sqrt(T(9)/T(10))
    λ₂ = sqrt(T(9)/T(70))
    λ₃ = λ₄
    λ₅ = sqrt(T(9)/T(19))

    twoⁿ = T(1) * (1 << n)
    w₁ = twoⁿ * ((T(12824) - T(9120)*n + T(400)*n^2) / T(19683))
    w₂ = twoⁿ * (T(980) / T(6561))
    w₃ = twoⁿ * ((T(1820) - T(400)*n) / T(19683))
    w₄ = twoⁿ * (T(200) / T(19683))
    w₅ = T(6859)/T(19683)
    w₄′ = twoⁿ * (T(25)/T(729))
    w₃′ = twoⁿ * ((T(265) - T(100)*n)/T(1458))
    w₂′ = twoⁿ * (T(245)/T(486))
    w₁′ = twoⁿ * ((T(729) - T(950)*n + T(50)*n^2)/T(729))

    p₂ = _combos(n, 1, λ₂)
    p₃ = _combos(n, 1, λ₃)
    p₄ = _signcombos(n, 2, λ₄)
    p₅ = _signcombos(n, n, λ₅)

    return GenzMalik{n,T}((p₂, p₃, p₄, p₅), (w₁, w₂, w₃, w₄, w₅), (w₁′, w₂′, w₃′, w₄′))
end

function get_rule(n::Int, ::Type{T}=Float64) where {T<:Real}
    lock(_gm_lock)
    try
        key = (n, T)
        haskey(_gm_cache, key) && return _gm_cache[key]::GenzMalik{n,T}
        g = _GenzMalik(n, T)
        _gm_cache[key] = g
        return g
    finally
        unlock(_gm_lock)
    end
end

countevals(::GenzMalik{n}) where {n} = 1 + 4n + 2n*(n-1) + (1 << n)

function _eval_rule(g::GenzMalik{n,Tg}, f, a::AbstractVector{Ta}, b::AbstractVector{Ta}, normfun) where {n,Tg<:Real,Ta<:Real}
    T = promote_type(Tg, Ta)
    c = (T.(a) .+ T.(b)) .* (T(0.5))
    Δ = (T.(b) .- T.(a)) .* (T(0.5))
    V = prod(Δ)

    f₁ = f(c)
    f₂ = zero(f₁)
    f₃ = zero(f₁)
    twelvef₁ = f₁ * T(12)
    maxdivdiff = zero(normfun(f₁))
    divdiff = Vector{typeof(maxdivdiff)}(undef, n)
    # scratch vectors to avoid allocations when evaluating f at shifted points
    cplus = similar(c)
    cminus = similar(c)
    @inbounds for i in 1:n
        # compute c ± Δ .* p₂
        p2i = g.p[1][i]
        for j in 1:n
            t = Δ[j] * p2i[j]
            cplus[j] = c[j] + t
            cminus[j] = c[j] - t
        end
        f₂ᵢ = f(cplus) + f(cminus)
        # compute c ± Δ .* p₃
        p3i = g.p[2][i]
        for j in 1:n
            t = Δ[j] * p3i[j]
            cplus[j] = c[j] + t
            cminus[j] = c[j] - t
        end
        f₃ᵢ = f(cplus) + f(cminus)
        f₂ += f₂ᵢ
        f₃ += f₃ᵢ
        dd = normfun(f₃ᵢ + twelvef₁ - (f₂ᵢ * T(7)))
        divdiff[i] = dd
        if dd > maxdivdiff
            maxdivdiff = dd
        end
    end

    f₄ = zero(f₁)
    @inbounds for p in g.p[3]
        for j in 1:n
            cplus[j] = c[j] + Δ[j] * p[j]
        end
        f₄ += f(cplus)
    end

    f₅ = zero(f₁)
    @inbounds for p in g.p[4]
        for j in 1:n
            cplus[j] = c[j] + Δ[j] * p[j]
        end
        f₅ += f(cplus)
    end

    I = V * (g.w[1]*f₁ + g.w[2]*f₂ + g.w[3]*f₃ + g.w[4]*f₄ + g.w[5]*f₅)
    I′ = V * (g.w′[1]*f₁ + g.w′[2]*f₂ + g.w′[3]*f₃ + g.w′[4]*f₄)
    E = normfun(I - I′)

    # choose axis
    kdivide = 1
    δf = E / (T(10)^n * V)
    for i in 1:n
        δ = divdiff[i] - maxdivdiff
        if δ > δf
            kdivide = i
            maxdivdiff = divdiff[i]
        elseif abs(δ) <= δf && abs(Δ[i]) > abs(Δ[kdivide])
            kdivide = i
        end
    end

    return I, E, kdivide
end

struct Box{T<:Real, TI<:Real}
    a::Vector{T}
    b::Vector{T}
    I::TI        # integral value (scalar)
    E::Float64   # error estimate as a real scalar
    kdiv::Int
end
Base.isless(i::Box, j::Box) = i.E < j.E

function _hcubature(f, a::Vector{T}, b::Vector{T};
                    norm::Function=norm, rtol::Real=0, atol::Real=0,
                    maxevals::Integer=typemax(Int), initdiv::Integer=1) where {T<:Real}
    length(a) == length(b) || throw(DimensionMismatch("endpoints must have same length"))
    n = length(a)
    n >= 2 || throw(ArgumentError("hcubature requires n ≥ 2; got n=$n"))
    F = float(T)
    g = get_rule(n, F)

    # Determine scalar integral type once by probing integrand at the midpoint
    mid = (F.(a) .+ F.(b)) .* F(0.5)
    TI = typeof(f(mid))

    # evaluation counter
    calls = Ref(0)
    fcount(x) = (calls[] += 1; f(x))

    # initial boxes: by default just one box; initdiv>1 splits uniformly
    split_points = [range(a[i], b[i], length=initdiv+1) for i in 1:n]
    heap = DataStructures.BinaryMaxHeap{Box{F,TI}}()

    # running totals
    Itot = zero(TI)
    Etot = 0.0

    # iterate over all subboxes
    function push_box(a₀, b₀)
        I, E, k = _eval_rule(g, fcount, a₀, b₀, norm)
        Ii = convert(TI, I)
        Ee = float(E)
        DataStructures.push!(heap, Box{F,TI}(a₀, b₀, Ii, Ee, k))
        return Ii, Ee
    end

    # generate Cartesian product of intervals
    function _build_boxes(i::Int, acc_a, acc_b)
        if i > n
            Ii, Ee = push_box(copy(acc_a), copy(acc_b))
            Itot += Ii
            Etot += Ee
            return
        end
        for j in 1:initdiv
            acc_a[i] = F(split_points[i][j])
            acc_b[i] = F(split_points[i][j+1])
            _build_boxes(i+1, acc_a, acc_b)
        end
    end
    _build_boxes(1, zeros(F, n), zeros(F, n))

    # default rtol if not specified
    if rtol == 0
        rtol = sqrt(eps(F))
    end

    # refine until tolerance or maxevals reached
    while Etot > max(atol, rtol*norm(Itot)) && calls[] + countevals(g) <= maxevals
        box = DataStructures.pop!(heap) # largest E
        # remove its contribution
        Itot -= box.I
        Etot -= box.E
        # split along kdiv
        k = box.kdiv
        mid = F(0.5) * (box.a[k] + box.b[k])
        a1 = copy(box.a); b1 = copy(box.b); b1[k] = mid
        a2 = copy(box.a); a2[k] = mid;      b2 = copy(box.b)
        I1, E1 = push_box(a1, b1)
        I2, E2 = push_box(a2, b2)
        # add contributions
        Itot += I1 + I2
        Etot += E1 + E2
    end

    return (Itot, Etot)
end

function hcubature(f, a, b; norm=norm, rtol::Real=0, atol::Real=0,
                   maxevals::Integer=typemax(Int), initdiv::Integer=1)
    F = float(promote_type(eltype(a), eltype(b)))
    return _hcubature(f, collect(F.(a)), collect(F.(b)); norm=norm, rtol=rtol, atol=atol, maxevals=maxevals, initdiv=initdiv)
end

end # module HCubature