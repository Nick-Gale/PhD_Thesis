using GLPK
using TravelingSalesmanExact; set_default_optimizer!(GLPK.Optimizer)
using DelimitedFiles
using Dates
using Random
using Concorde

Random.seed!(1)
#warm up

dir = "./dataElasticNetComparison/"
label = datetime2unix(now()) #a time signature to uniquely label the tour
repeats = 25
n_vec = [[i, j] for i = 100:100:1000 for j = 1:repeats]

for i = 1:length(n_vec)
    n = n_vec[i][1]
    s = n_vec[i][2]
    x = 100000 * rand(n)
    y = 100000 * rand(n) 
    @time tour, cost = solve_tsp(x, y; dist="EUC_2D")  # tour, cost = get_optimal_tour(cities; verbose = false);#  tour = randperm(n) #
    append!(tour, -1)

    data = zeros(n, 3)
    for i = 1:n
        data[i, 1] = i
        data[i, 2] = x[i]
        data[i, 3] = y[i]
    end
    
    filename_tour = dir * "euc_rand_tour_$(n)_$(s).tsp"
    filename_solution = dir * "euc_rand_tour_$(n)_$(s).opt.tour" # use opt.fake.tour if you need a large tour without a solution
    
    name = "NAME : Euclidean Random Tour : n = $(n), sample = $(s) \n"
    comment = "COMMENT : Coordinates are generated uniformly sampled from [0, 100000] x [0, 100000] \n"
    dimension = "DIMENSION : $(n) \n"
    edge_weight = "EDGE_WEIGHT_TYPE : EUC_2D \n"
    type = "TYPE : TSP \n"

    #write the tour data 
    io = open(filename_tour, "a")  do io
        write(io, name)
        write(io, comment)
        write(io, dimension)
        write(io, type)
        write(io, edge_weight)
        write(io, "NODE_COORD_SECTION \n")
        writedlm(io, data)
        write(io, "EOF")
    end

    #write the optimal tour
    io = open(filename_solution, "a")  do io
        write(io, name)
        write(io, comment)
        write(io, dimension)
        write(io, edge_weight)
        write(io, "TOUR_SECTION \n")
        writedlm(io, tour)
    end

end