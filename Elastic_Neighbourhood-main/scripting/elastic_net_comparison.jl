#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose which tours you would like to solve and where you would like to save them
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dir = "./dataElasticNetComparison/"
save_loc = "./dataElasticNetComparison/results_gpu.txt"
file_names = readdir(dir)
opt_tours = filter(x->occursin(".tour", x), file_names)
tour_data = data_load(file_names, opt_tours, dir)
reshuffle = sortperm([length(tour_data[i][1]) for i in 1:length(tour_data)])
tour_data = tour_data[reshuffle]
test_indices = 1:length(tour_data) # 1:100:length(tour_data) #1:1 # [1, 20, 80, 100, 120, 150, 200] # [length(tour_data) - 3] #  [1]#      collect(1:10) #  
plot_index = length(test_indices)
all_lengths = map(x -> length(x[1][:]), tour_data)# println(length.(tour_data[:,2]))
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set parameters
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
L = (length(tour_data[test_indices][1][1]))
println("The length is $(L)")

params_dict_neighbourhood = Dict(
    :max_iterations => 25,
    :n_neighbourhoods => 3, 
    :optimiser => "adam!",
    :x_init => 0.0,
    :y_init => 0.0,
    :initialisation_mode => "grid"
)

params_dict_net = Dict(
    :n_neighbourhoods => 2.5, 
    :c => 0,
    :optimiser => "gd!",
    :x_init => 0.0,
    :y_init => 0.0,
    :initialisation_mode => "elastic_net",
)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tours, opt_tours, computed_distances_neighbourhood, opt_distances  = @time batch_tours(params_dict_neighbourhood, test_indices, tour_data)

tours, opt_tours, computed_distances_net, opt_distances  = @time batch_tours(params_dict_net, test_indices, tour_data)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Make DataFrame and Plot
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
using DataFrames
using StatsPlots
df = DataFrame(opt=String[], L=Int64[], beta=Float64[])

for i = 1:length(test_indices)
    length_ind = test_indices[i]
    push!(df, ("Elastic Net", all_lengths[length_ind], computed_distances_net[i]/opt_distances[i] - 1))
    push!(df, ("Elastic Neighbourhood", all_lengths[length_ind], computed_distances_neighbourhood[i]/opt_distances[i] - 1))
end
inds = df[!, :beta] .< 1
df = df[inds,:]
a = @df df groupedboxplot(string.(:L), :beta, group=:opt, 
color=[RGBA(0, 0.4470, 0.7410) RGBA(0.8500,0.3250, 0.0980) RGBA(0.9290,0.6940, 0.1250) RGBA(0.4940,0.1840, 0.5560) RGBA(0.4660,0.6740, 0.1880)], 
linewidth=1,
xlabel="n(nodes)",
ylabel="Fractional Error",
title="Elastic Net Comparison: Error Against Optimal",
dpi=500,
ylim=(0,0.2))

savefig(a, "figures/fig_elastic_net_comparison.png")


tours, opt_tours, computed_distances_net, opt_distances = batch_tours(params_dict_net, [100], tour_data)
computed_distances_net ./ opt_distances