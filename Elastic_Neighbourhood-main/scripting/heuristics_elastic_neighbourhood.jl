#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose which tours you would like to solve and where you would like to save them
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dir = "./dataHeuristicComparisons/"
save_loc = "./dataHeuristicComparisons/results_gpu.txt"
@time file_names = readdir(dir)
@time opt_tours = filter(x->occursin(".tour", x), file_names)
@time tour_data = data_load(file_names, opt_tours, dir)
@time reshuffle = sortperm([length(tour_data[i][1]) for i in 1:length(tour_data)])
@time tour_data = tour_data[reshuffle]
test_indices = 1:100:length(tour_data)
ti = []
for i in test_indices
    push!(ti, collect(i:(i+30))...)
end
test_indices=ti 
println(test_indices)

plot_index = length(test_indices)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set parameters
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
L = (length(tour_data[test_indices][1][1]))
println("The length is $(L)")

params_dict = Dict(
    :initialisation_mode => "fractal",
    :x_init => 0.0,
    :y_init => 0.0
)

tours, opt_tours, computed_distances, opt_distances = @time batch_tours(params_dict, test_indices, tour_data; results_dir=save_loc, read_and_print=true)