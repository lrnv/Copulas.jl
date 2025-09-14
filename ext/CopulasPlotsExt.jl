module CopulasPlotsExt

@static if !isdefined(Base, :get_extension)
    # Julia <1.9 fallback (optional, but we can just require 1.9+ for extensions).
end

using Copulas
using Distributions
using RecipesBase
using Plots
using Plots.PlotMeasures
using Random
using StatsBase

function _mk_Z(obj, xs, ys, what::Symbol)
    f = what === :pdf ? Distributions.pdf :
        what === :logpdf ? Distributions.logpdf : 
        what === :cdf ? Distributions.cdf : throw(ArgumentError("Unsupported quantity: $what. Choose among :pdf, :logpdf, :cdf"))
    Z = Matrix{Float64}(undef, length(ys), length(xs))
    @inbounds for (i,x) in enumerate(xs), (j,y) in enumerate(ys)
        Z[j,i] = f(obj, [x,y])
    end
    return Z
end
function _contour_grid(S::Copulas.SklarDist{CT,M}, n; what::Symbol=:pdf, scale::Symbol=:copula) where {CT<:Copulas.Copula{2}, M}
    if scale === :copula
        xs = range(0.001, 0.999; length=n)
        ys = range(0.001, 0.999; length=n)
        Z = _mk_Z(S.C, xs, ys, what)
        return xs, ys, Z
    elseif scale === :sklar
        qs = range(0.001, 0.999; length=n)
        xs = [quantile(S.m[1], q) for q in qs]
        ys = [quantile(S.m[2], q) for q in qs]
        Z = _mk_Z(S, xs, ys, what)
        return xs, ys, Z
    else
        throw(ArgumentError("scale must be :copula or :sklar"))
    end
end

@recipe function f(C::Copulas.Copula{d}, what=nothing) where {d}
    if d==2 
        plotattributes[:show_marginals] = false
    end
    # consume any user :scale to avoid backend axis-transform warnings
    if haskey(plotattributes, :scale)
        delete!(plotattributes, :scale)
        # ensure no axis scales are affected by the removed attribute
        plotattributes[:xscale] = :identity
        plotattributes[:yscale] = :identity
        plotattributes[:zscale] = :identity
    end
    # mark that this plot originated from a Copula, so guides can use u₁/u₂ semantics
    plotattributes[:_source_is_copula] = true
    S = Copulas.SklarDist(C, ntuple(_ -> Distributions.Uniform(), d))
    return S, what
end

@recipe function f(S::Copulas.SklarDist{CT,TplMargins}, what=nothing) where {d, CT<:Copulas.Copula{d}, TplMargins}
    # Determine if a surface was requested (bivariate only); used to pick default scale
    is_surface = get(plotattributes, :seriestype, :no) == :surface
    # Deal with main plots scale: default to :sklar for bivariate surfaces, else :copula
    if haskey(plotattributes, :scale)
        scale = plotattributes[:scale]
        delete!(plotattributes, :scale)
        # ensure backend doesn't interpret unknown axis scales
        plotattributes[:xscale] = :identity
        plotattributes[:yscale] = :identity
        plotattributes[:zscale] = :identity
        (scale === :copula || scale === :sklar) || throw(ArgumentError("scale must be :copula or :sklar"))
    else
        scale = (is_surface && d == 2) ? :sklar : :copula
    end

    n_scatter =      get(plotattributes, :n, 1500)
    overlay_n =      get(plotattributes, :overlay_n, 60)
    pts_alpha =      get(plotattributes, :pts_alpha, 0.25)
    show_axes =      get(plotattributes, :show_axes, true)
    show_marginals = get(plotattributes, :show_marginals, true)
    bins =           get(plotattributes, :bins, 40)
    show_corr =      get(plotattributes, :show_corr, true)
    marg_alpha =     get(plotattributes, :marg_alpha, 0.6)
    draw_contour =   (what === :pdf || what === :logpdf || what === :cdf)
    source_is_copula = get(plotattributes, :_source_is_copula, false)

    # Common overlay alpha (prefer bivariate keyword, fallback to pairwise)
    if d == 2
        # Marginal layout if requested (and not surface)
        if show_marginals && !is_surface
            layout := grid(2,2, heights=[0.25,0.75], widths=[0.75,0.25])
            legend := false
            colorbar := get(plotattributes, :colorbar, false)
            xs = ys = Z = nothing
            if draw_contour
                xs, ys, Z = _contour_grid(S, overlay_n; what=what, scale=scale)
            end
            pts = n_scatter > 0 ? rand(S, n_scatter) : nothing
            if pts !== nothing
                @series begin
                    subplot := 1
                    seriestype := :histogram
                    bins := bins
                    normalize := :pdf
                    alpha := marg_alpha
                    linecolor := :black
                    xticks := false
                    yguide := (show_axes ? "density" : nothing)
                    framestyle := :box
                    pts[1,:]
                end
            end
            @series begin
                subplot := 2
                seriestype := :scatter
                framestyle := :none
                xticks := false; yticks := false
                markersize := 0.0001
                [NaN], [NaN]
            end
            if draw_contour
                @series begin
                    subplot := 3
                    seriestype := :contour
                    if (scale === :copula) || source_is_copula
                        xguide --> (show_axes ? "u₁" : nothing)
                        yguide --> (show_axes ? "u₂" : nothing)
                    else
                        xguide --> (show_axes ? "x₁" : nothing)
                        yguide --> (show_axes ? "x₂" : nothing)
                    end
                    if !show_axes
                        framestyle := :none; ticks := false
                    else
                        framestyle := :box
                    end
                    if scale === :copula && length(S.m) == 2 && all(mi -> mi isa Distributions.Uniform && mi.a == 0 && mi.b == 1, S.m)
                        xlims := (0,1); ylims := (0,1)
                    end
                    xs, ys, Z
                end
            end
            if pts !== nothing
                @series begin
                    subplot := 3
                    seriestype := :scatter
                    alpha := pts_alpha
                    markersize := 2
                    markerstrokewidth := 0
                    markercolor := :black
                    if scale === :copula
                        u1 = cdf(S.m[1], pts[1,:])
                        u2 = cdf(S.m[2], pts[2,:])
                        u1, u2
                    else
                        pts[1,:], pts[2,:]
                    end
                end
            end
            if pts !== nothing
                @series begin
                    subplot := 4
                    seriestype := :histogram
                    orientation := :horizontal
                    bins := bins
                    normalize := :pdf
                    alpha := marg_alpha
                    linecolor := :black
                    yticks := false
                    xguide := (show_axes ? "density" : nothing)
                    framestyle := :box
                    pts[2,:]
                end
            end
        else
            # Center-only bivariate
            if (scale === :copula) || source_is_copula
                xguide --> (show_axes ? "u₁" : nothing) 
                yguide --> (show_axes ? "u₂" : nothing) 
            else
                xguide --> (show_axes ? "x₁" : nothing)
                yguide --> (show_axes ? "x₂" : nothing)
            end
            legend := false
            if !show_axes
                framestyle := :none
                ticks := false
            end
            colorbar := get(plotattributes, :colorbar, false)
            # Default camera for surface plots: rotate 30° to the left, keep elevation at 30° (user can override)
            if is_surface && !haskey(plotattributes, :camera)
                camera := (-30, 30)
            end
            if draw_contour
                xs, ys, Z = _contour_grid(S, overlay_n; what=what, scale=scale)
            end
            if scale === :copula && length(S.m) == 2 && all(mi -> mi isa Distributions.Uniform && mi.a == 0 && mi.b == 1, S.m)
                xlims --> (0,1); ylims --> (0,1)
            end
            if draw_contour
                @series begin
                    seriestype := get(plotattributes, :seriestype, :contour)
                    if is_surface && show_axes
                        zguide --> (scale === :copula ? "copula $(what)" : string(what))
                    end
                    xs, ys, Z
                end
            end
            if n_scatter > 0 && !is_surface
                @series begin
                    seriestype := :scatter
                    alpha := pts_alpha
                    markersize := 2
                    markerstrokewidth := 0
                    markercolor := :black
                    raw = rand(S, n_scatter)
                    if scale === :copula
                        u1 = cdf(S.m[1], raw[1,:])
                        u2 = cdf(S.m[2], raw[2,:])
                        u1, u2
                    else
                        raw[1,:], raw[2,:]
                    end
                end
            end
        end
        return
    else
        # Use scale already parsed above
        # Uniform (copula) sample for lower-triangle overlays & contours; full sample for marginals
        U = n_scatter > 0 ? rand(S.C, n_scatter) : nothing
        X = n_scatter > 0 ? rand(S, n_scatter) : nothing
        layout := (d, d)
        size --> (160 * d, 160 * d)
        legend := false
        colorbar := false
        plot_margin := 0mm
        left_margin := 0mm; right_margin := 0mm; top_margin := 0mm; bottom_margin := 0mm
        cols = d
        for i in 1:d
            for j in 1:d
                idx = (i-1) * cols + j
                show_x = (i == d)
                show_y = (j == 1)
                fs = :box
                xt = show_x ? :auto : false
                yt = show_y ? :auto : false
                if i == j
                    # Diagonal: always show marginal histograms (marginal scale) when sample is available
                    if X !== nothing
                        @series begin
                            subplot := idx
                            seriestype := :histogram
                            bins := bins
                            normalize := :pdf
                            linecolor := :black
                            alpha := 0.6
                            framestyle := fs
                            xticks := xt
                            yticks := yt
                            X[i,:]
                        end
                        xs_dense = range(minimum(X[i,:]), maximum(X[i,:]); length=200)
                        xi = X[i,:]; nloc = length(xi); σ = 1.06 * std(xi) * nloc^(-1/5)
                        if σ > 0 && isfinite(σ)
                            dens = [mean(@. exp(-0.5*((x - xi)/σ)^2)) / (σ*sqrt(2π)) for x in xs_dense]
                            @series begin
                                subplot := idx
                                seriestype := :line
                                framestyle := fs
                                xticks := xt
                                yticks := yt
                                xs_dense, dens
                            end
                        end
                    end
                elseif i > j
                    if draw_contour
                        @series begin
                            subplot := idx
                            seriestype := :contour
                            framestyle := fs
                            xticks := xt
                            yticks := yt
                            colorbar := false
                            Cij = Copulas.subsetdims(S.C, (j,i))
                            Sij = Copulas.SklarDist(Cij, (S.m[j], S.m[i]))
                            xs, ys, Z = _contour_grid(Sij, overlay_n; what=what, scale=scale)
                            xs, ys, Z
                        end
                    end
                    if (scale === :copula && U !== nothing) || (scale === :sklar && X !== nothing)
                        @series begin
                            subplot := idx
                            seriestype := :scatter
                            markersize := 1.6
                            markerstrokewidth := 0
                            markercolor := :black
                            alpha := pts_alpha
                            framestyle := fs
                            xticks := xt
                            yticks := yt
                            if scale === :copula
                                U[j,:], U[i,:]
                            else
                                X[j,:], X[i,:]
                            end
                        end
                    end
                else
                    if show_corr && ((scale === :copula && U !== nothing) || (scale === :sklar && X !== nothing))
                        if scale === :copula
                            x = U[j,:]; y = U[i,:]
                        else
                            x = X[j,:]; y = X[i,:]
                        end
                        rho = StatsBase.corspearman(x,y)
                        tau = StatsBase.corkendall(x,y)
                        trunc2(x) = trunc(x * 100) / 100
                        label_str = "τ=$(trunc2(tau))  ρ=$(trunc2(rho))"
                        @series begin
                            subplot := idx
                            seriestype := :scatter
                            framestyle := fs
                            xticks := false
                            yticks := false
                            markersize := 0.0001
                            markerstrokewidth := 0
                            annotation := (0.5, 0.5, text(label_str, :center, 8))
                            [NaN], [NaN]
                        end
                    else
                        @series begin
                            subplot := idx
                            seriestype := :scatter
                            framestyle := fs
                            xticks := false
                            yticks := false
                            markersize := 0.0001
                            markerstrokewidth := 0
                            [NaN], [NaN]
                        end
                    end
                end
            end
        end
    end
end
end # module
