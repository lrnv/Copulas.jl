###########################################################################
#####  Histogram / Checkerboard conditional distortion (uniform on bins)
###########################################################################

struct HistogramBinDistortion{T} <: Distortion
    m::Int                # number of bins along the axis
    probs::Vector{T}      # length m, normalized probabilities per bin
    cums::Vector{T}       # prefix sums, cums[k] = sum(probs[1:k])
    function HistogramBinDistortion(m::Integer, probs::AbstractVector{<:Real})
        m < 1 && throw(ArgumentError("m must be ≥ 1"))
        m==1 && return NoDistortion()
        length(probs) == m || throw(ArgumentError("probs length must be m"))
        v = float.(probs)
        s = sum(v)
        (s <= 0 || any(v .< 0)) && throw(ArgumentError("probabilities must be positive and have positive sum"))
        v ./= s
        c = cumsum(v)
        return new{eltype(v)}(Int(m), v, c)
    end
end

function Distributions.cdf(d::HistogramBinDistortion, u::Real)
    # piecewise linear within each bin
    m = d.m
    t = clamp(float(u), 0.0, 1.0)
    s = m * t
    k = min(max(floor(Int, s), 0), m - 1)   # 0-based bin index
    idx = k + 1                              # 1-based
    frac = s - k
    base = (idx > 1) ? d.cums[idx - 1] : zero(eltype(d.cums))
    return base + d.probs[idx] * frac
end

function Distributions.quantile(d::HistogramBinDistortion, α::Real)
    αf = clamp(float(α), 0.0, 1.0)
    if αf == 0.0; return 0.0; end
    if αf == 1.0; return 1.0; end
    # locate bin by prefix sums
    idx = searchsortedfirst(d.cums, αf)
    prev = (idx > 1) ? d.cums[idx - 1] : 0.0
    p = d.probs[idx]
    frac = (αf - prev) / max(p, eps(eltype(p)))
    # map back to [0,1] with m bins
    k = idx - 1
    return (k + frac) / d.m
end

function Distributions.logpdf(d::HistogramBinDistortion, u::Real)
    # Density is piecewise constant: on bin k, f(u) = m * probs[k]
    # Support is (0,1); return -Inf at boundaries for numerical safety.
    uf = float(u)
    (0 < uf < 1) || return -Inf
    m = d.m
    s = m * uf
    k = min(max(floor(Int, s), 0), m - 1)
    idx = k + 1
    p = d.probs[idx]
    (p <= 0) && return -Inf
    return log(float(m)) + log(float(p))
end
