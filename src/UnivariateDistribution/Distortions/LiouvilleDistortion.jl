struct LiouvilleDistortion{TM,TCM} <: Distortion
    margin::TM
    conditional_margin::TCM
end

function Distributions.cdf(D::LiouvilleDistortion, u::Real)
    u <= 0 && return zero(float(u))
    u >= 1 && return one(float(u))
    x = Distributions.quantile(D.margin, 1 - u)
    return Distributions.ccdf(D.conditional_margin, x)
end
function Distributions.quantile(D::LiouvilleDistortion, p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
    iszero(p) && return zero(float(p))
    isone(p) && return one(float(p))
    x = Distributions.quantile(D.conditional_margin, 1 - p)
    return Distributions.ccdf(D.margin, x)
end
function Distributions.logpdf(D::LiouvilleDistortion, u::Real)
    0 < u < 1 || return -Inf
    x = Distributions.quantile(D.margin, 1 - u)
    return log(Distributions.pdf(D.conditional_margin, x)) -
           log(Distributions.pdf(D.margin, x))
end
