pseudos(sample) = transpose(hcat([StatsBase.ordinalrank(sample[i,:])./(size(sample,2)+1) for i in 1:size(sample,1)]...))
