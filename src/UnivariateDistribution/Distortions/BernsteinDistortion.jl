struct BernsteinDistortion{M} <: Distortion
    mixture::M
end
Distributions.cdf(d::BernsteinDistortion, u::Real) =
    u <= 0 ? zero(float(u)) : u >= 1 ? one(float(u)) : Distributions.cdf(d.mixture, u)
Distributions.logcdf(d::BernsteinDistortion, u::Real) = log(Distributions.cdf(d, u))
Distributions.pdf(d::BernsteinDistortion, u::Real) = Distributions.pdf(d.mixture, u)
Distributions.logpdf(d::BernsteinDistortion, u::Real) = log(Distributions.pdf(d, u))
Distributions.quantile(d::BernsteinDistortion, p::Real) = _quantile_from_cdf(d, p)
