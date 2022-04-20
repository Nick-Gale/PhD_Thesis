#THE ELASTIC NEIGHBOURHOOD -- Author: Nicholas Gale
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose which tours you would like to solve and where you would like to save them
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dir = "./dataRandom/"
save_loc = "./tests.txt"
file_names = readdir(dir)
opt_tours_names = filter(x->occursin(".tour", x), file_names)
tour_data = data_load(file_names, opt_tours_names, dir)
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
    #:max_iterations => 20,
    :n_neighbourhoods => 2,
    :a => 1,
    :t_amp => 0.1,
    :initialisation_mode => "fractal",
    :n_inits => 1,
    :x_init => 0.0,
    :y_init => 0.0,
    :record => false,
    :threads => 16
)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tours, opt_tours, computed_distances, opt_distances = @time batch_tours(params_dict, test_indices, tour_data)
# #turn off for HPC

plot_analyse(plot_index, test_indices, tour_data, tours, opt_tours, computed_distances, opt_distances; DPI = 500, display_nodes=false)
println(computed_distances ./ opt_distances)
println(computed_distances ./ sqrt.(0.7124^2 .* length.(opt_tours)))

# plot_animation(tour_data[plot_index][1], tour_data[plot_index][2], tours[plot_index], "allo")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
println("Done.")