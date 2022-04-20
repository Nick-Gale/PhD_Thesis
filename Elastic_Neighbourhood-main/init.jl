using Distributed
n_procs = 1
addprocs(n_procs)

#plotting environment variable to fix axes labelling issues
ENV["GKSwstype"] = "100"

@everywhere using Revise
@everywhere includet("./src/module_elastic_neighbourhood.jl")
@everywhere using .ElasticNeighbourhood
