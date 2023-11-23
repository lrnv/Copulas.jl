"""
    pseudos(sample)

Compute the pseudo-observations of a multivariate sample. Note that the sample has to be given in wide format (d,n), where d is the dimension and n the number of observations.
"""
pseudos(sample) = transpose(hcat([StatsBase.ordinalrank(sample[i,:])./(size(sample,2)+1) for i in 1:size(sample,1)]...))
