#THE ELASTIC NEIGHBOURHOOD -- Author: Nicholas Gale
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose which tours you would like to solve and where you would like to save them
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dir = "./dataArt/"
save_loc = "./results/results_art.txt"
file_names = readdir(dir)
opt_tours = filter(x->occursin(".tour", x), file_names)
tour_data = data_load(file_names, opt_tours, dir)

reshuffle = sortperm([length(tour_data[i][1]) for i in 1:length(tour_data)])
tour_data = tour_data[reshuffle]
test_indices = 1:1
plot_index = length(test_indices)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set parameters
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
L = (length(tour_data[test_indices][1][1]))
println("The length is $(L)")

params_dict = Dict(
    :n_neighbourhoods => 1.5, 
    :n_inits => 1,
    :initialisation_mode => "fractal",
    :x_init => 0.0,
    :y_init => 0.0
)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
opt_times = [114.4, 179.1, 254.1, 409.4, 486.84, 676.0] .* 3600
opt_costs = [5757191, 6543609, 6810665, 7619953, 7888731, 8171677]

record_times = []
tour_costs = []
for i in test_indices
    t = @elapsed tours, opt_tours_i, computed_distances, opt_distances = batch_tours(params_dict, i, tour_data)
    l = compute_tour_length(tour_data[i][4], tour_data[i][5], Int.(tours[1][1]), euc_dist)
    append!(record_times, t)
    append!(tour_costs, l)
    plot_figs(tour_data[i][2], tour_data[i][3], Array{Int32,1}(tours[1][1]), tours[1], 500, save_name=string(i))
end
inds_rec = 1:length(record_times)
#Record 
using DelimitedFiles
data = hcat(collect(inds_rec), tour_costs, opt_costs[inds_rec], record_times, opt_times[inds_rec], tour_costs ./ opt_costs[inds_rec], record_times ./ opt_times[inds_rec])
writedlm("./dataArt/record_data", data)
