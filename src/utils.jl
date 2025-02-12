"""
    pseudos(sample)

Compute the pseudo-observations of a multivariate sample. Note that the sample has to be given in wide format (d,n), where d is the dimension and n the number of observations.

Warning: the order used is ordinal ranking like https://en.wikipedia.org/wiki/Ranking#Ordinal_ranking_.28.221234.22_ranking.29, see `StatsBase.ordinalrank` for the ordering we use. If you want more flexibility, checkout `NormalizeQuantiles.sampleranks`.
"""
pseudos(sample::AbstractMatrix) = transpose(hcat([StatsBase.ordinalrank(sample[i, :]) ./ (size(sample, 2) + 1) for i in 1:size(sample, 1)]...))
