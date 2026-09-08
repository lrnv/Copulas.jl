# Radial law of a lower-order margin. If ψ = W_D(F_R), Dirichlet
# aggregation gives W_d⁻¹(ψ) = Law(RB), B ~ Beta(d, D-d), independently.
struct WilliamsonBetaProduct{TX,TB,TO} <: Distributions.ContinuousUnivariateDistribution
    X::TX
    B::TB
    # Keep the exact originating order: reconstructing it as a + b from the
    # Beta parameters can lose structural identities through floating rounding.
    source_order::TO
end

_williamson_beta_source_order(B::Distributions.Beta) = sum(Distributions.params(B))
WilliamsonBetaProduct(X, B::Distributions.Beta) =
    WilliamsonBetaProduct(X, B, _williamson_beta_source_order(B))

function WilliamsonBetaProduct(X::WilliamsonFromFrailty, B::Distributions.Beta, source_order::Real)
    target_order = first(Distributions.params(B))
    source_order == X.order || return WilliamsonBetaProduct{typeof(X),typeof(B),typeof(source_order)}(X, B, source_order)
    return WilliamsonFromFrailty(X.frailty_dist, target_order)
end

function WilliamsonBetaProduct(X::WilliamsonBetaProduct, B::Distributions.Beta, source_order::Real)
    inner_target = first(Distributions.params(X.B))
    outer_target = first(Distributions.params(B))
    if source_order == inner_target
        merged_beta = Distributions.Beta(outer_target, X.source_order - outer_target)
        return WilliamsonBetaProduct(X.X, merged_beta, X.source_order)
    end
    return WilliamsonBetaProduct{typeof(X),typeof(B),typeof(source_order)}(X, B, source_order)
end

function Distributions.cdf(dist::WilliamsonBetaProduct, x::Real)
    x <= 0 && return zero(float(x))
    return Distributions.expectation(dist.X) do r
        r <= x ? one(float(x)) : Distributions.cdf(dist.B, x / r)
    end
end

function Distributions.pdf(dist::WilliamsonBetaProduct, x::Real)
    x <= 0 && return zero(float(x))
    return Distributions.expectation(dist.X) do r
        r <= x ? zero(float(x)) : Distributions.pdf(dist.B, x / r) / r
    end
end

# For continuous radials, conditioning on B integrates over its bounded support
# and reuses the radial distribution's specialized cdf/pdf implementations.
function Distributions.cdf(
    dist::WilliamsonBetaProduct{<:Distributions.ContinuousUnivariateDistribution},
    x::Real,
)
    x <= 0 && return zero(float(x))
    x >= Base.maximum(dist) && return one(float(x))
    return Distributions.expectation(b -> Distributions.cdf(dist.X, x / b), dist.B)
end

# The beta-density endpoint singularity makes the otherwise preferable
# E[F_X(x/B)] representation inaccurate in the upper tail for finite-support
# negative-Clayton radials. This radial is atomless, so its complementary-tail
# integral is exact and numerically stable. Keep the specialization narrow:
# generic Williamson inverses may carry boundary atoms despite their nominal
# ContinuousUnivariateDistribution type.
function Distributions.cdf(
    dist::WilliamsonBetaProduct{<:ClaytonWilliamsonDistribution},
    x::Real,
)
    x <= 0 && return zero(float(x))
    upper = Base.maximum(dist)
    x >= upper && return one(float(x))

    # Evaluate the upper tail directly. Besides avoiding cancellation near the
    # finite endpoint, this exposes `x` as an integration boundary instead of
    # hiding a discontinuity inside E[F_B(x / X)]. This matters for beta
    # reductions whose density is singular at an endpoint.
    lower = max(x, Base.minimum(dist.X))
    tail = QuadGK.quadgk(lower, Base.maximum(dist.X)) do r
        Distributions.pdf(dist.X, r) * Distributions.ccdf(dist.B, x / r)
    end
    return clamp(one(tail[1]) - tail[1], zero(tail[1]), one(tail[1]))
end

function Distributions.pdf(
    dist::WilliamsonBetaProduct{<:Distributions.ContinuousUnivariateDistribution},
    x::Real,
)
    x <= 0 && return zero(float(x))
    return Distributions.expectation(
        b -> iszero(b) ? zero(float(x)) : Distributions.pdf(dist.X, x / b) / b,
        dist.B,
    )
end

Distributions.logpdf(dist::WilliamsonBetaProduct, x::Real) = log(Distributions.pdf(dist, x))
Distributions.rand(rng::Distributions.AbstractRNG, dist::WilliamsonBetaProduct) =
    rand(rng, dist.X) * rand(rng, dist.B)
Base.minimum(dist::WilliamsonBetaProduct) = zero(float(Base.minimum(dist.X)))
Base.maximum(dist::WilliamsonBetaProduct) = Base.maximum(dist.X)

function Distributions.quantile(dist::WilliamsonBetaProduct, p::Real)
    return _quantile_from_cdf(dist, p)
end
