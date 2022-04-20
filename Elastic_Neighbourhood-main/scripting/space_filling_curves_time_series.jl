using Plots
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose which tours you would like to solve and where you would like to save them
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dir = "./dataRandom/"
save_loc = "./results_art.txt"
file_names = readdir(dir)
opt_tours_names = filter(x->occursin(".tour", x), file_names)
tour_data = data_load(file_names, opt_tours_names, dir)
reshuffle = sortperm([length(tour_data[i][1]) for i in 1:length(tour_data)])
tour_data = tour_data[reshuffle]
test_indices = 1:1
plot_index = 1

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set parameters
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
params_dict_static = Dict(
    :max_iterations => 1,
    :initialisation_mode => "grid",
    :t_amp => 0.05,
    :c => 0.0025,
    :init_radius => 0.1,
    :a => 0.0,
    :n_inits => 1,
    :x_init => 0.0,
    :y_init => 0.0,
)

params_dict_full = Dict(
    :initialisation_mode => "grid",
    :a => 0.0,
    :n_inits => 1,
    :x_init => 0.0,
    :y_init => 0.0,
)

params_time= Dict(
    :initialisation_mode => "grid",
    :record => true,
    :n_inits => 1,
    :x_init => 0.0,
    :y_init => 0.0,
)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tours_static, opt_tours, computed_distances, opt_distances = @time batch_tours(params_dict_static, test_indices, tour_data)
tours_full, opt_tours, computed_distances, opt_distances = @time batch_tours(params_dict_full, test_indices, tour_data)

plot_analyse(plot_index, test_indices, tour_data, tours_static, opt_tours, computed_distances, opt_distances; display_nodes=false, save_name="space_filling_static")
plot_analyse(plot_index, test_indices, tour_data, tours_full, opt_tours, computed_distances, opt_distances; display_nodes=false, save_name="space_filling_full_iterations")


dir = "./dataInitialisation/"
save_loc = "./results_art.txt"
file_names = readdir(dir)
opt_tours_names = filter(x->occursin(".tour", x), file_names)
tour_data = data_load(file_names, opt_tours_names, dir)
reshuffle = sortperm([length(tour_data[i][1]) for i in 1:length(tour_data)])
tour_data = tour_data[reshuffle]
test_indices = 1:1
plot_index = 1

tours_time, opt_tours, computed_distances, opt_distances = @time batch_tours(params_time, test_indices, tour_data)
pneig = plot_time_series(tour_data[test_indices][1][1], tour_data[test_indices][1][2], tours_time[1]; ms=1)
savefig(pneig, "./figures/fig_elastic_neighbourhood_time_series.png")
