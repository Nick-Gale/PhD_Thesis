#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This script generates the initialisation domains and the scaling for a small dataset
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
using Plots, Random, CUDA, GLM, DataFrames, StatsBase
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose which tours you would like to solve and where you would like to save them
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dir = "./dataInitialisation/"
save_loc = "./results_art.txt"
file_names = readdir(dir)
opt_tours = filter(x->occursin(".tour", x), file_names)
tour_data = data_load(file_names, opt_tours, dir)
tour_lengths = [length(tour_data[i][1]) for i in 1:length(tour_data)]
unique_tour_lengths = unique(tour_lengths)
test_indices = [1,11]

grid_depth = 50
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set parameters for the clustering plots
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
params_dict = Dict(:n_inits => grid_depth^2, :initialisation_mode => "grid")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tours, opt_tours, computed_distances, opt_distances = batch_tours(params_dict, test_indices, tour_data)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot some clusters
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
clusters_uniform = vec(tours[1][4] ./ opt_distances[1])
clusters_skew = vec(tours[2][4] ./ opt_distances[2])
mz1 = 2
mz2 = 2.2
println(minimum(clusters_uniform))

p_empty = plot([0], axis=false, grid=false, legend=false, ticks=false, background_color=:transparent);
uniform_scatter_plt = plot(tour_data[1][1], tour_data[1][2], xlims=(-0.5, 0.5), ylims=(-0.5, 0.5), markersize=mz1, markerstrokewidth=0.0, st=:scatter, title="Uniformly Distributed Data", dpi=500, legend=false, aspect_ratio=1)
skew_scatter_plt = plot(tour_data[11][1], tour_data[11][2], xlims=(-0.5, 0.5), ylims=(-0.5, 0.5), markersize=mz1, markerstrokewidth=0.0, st=:scatter, title="Origin Skewed Data", dpi=500, legend=false, aspect_ratio=1)
cluster_uniform_plt = plot_clusters(clusters_uniform, "grid", params_dict[:n_inits]; DPI = 500, mz=mz2)
cluster_skew_plt = plot_clusters(clusters_skew, "grid", params_dict[:n_inits]; DPI = 500, mz=mz2)
ll = @layout [a{0.4w} b{0.1w} c{0.4w} d{0.1w}; e f]
plt_cluster = plot(uniform_scatter_plt, p_empty, skew_scatter_plt, p_empty, cluster_uniform_plt, cluster_skew_plt, layout=ll, dpi=500)
savefig(plt_cluster, "./figures/fig_initialisation_clusters_$(grid_depth).png")
println("Clustering Done")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Increase the number of n_inits and store the fractional improvements
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_indices = collect(1:10)
supN = 100
step_size = 1

dict_titles = Dict("simulated_annealing" => "Simulated Annealing", "nelder_mead" => "Nelder-Mead", "random" => "Random", "particle_swarm" => "Particle Swarm")
schemes = ["simulated_annealing", "nelder_mead", "random", "particle_swarm"]
for s in schemes
    params_dict_i = Dict(:n_inits => supN, :initialisation_mode => s, :x_init => 0.0, :y_init => 0, :init_radius => 0.01, :t0 => 1.0)
    @time tours_i, opt_tours_i, computed_distances_i, opt_distances_i  = batch_tours(params_dict_i, test_indices, tour_data)

    data = zeros(length(1:step_size:supN), length(test_indices))

    for i = 1:length(tours_i)
        for j = 1:supN
            ind = minimum([length(tours_i[i][4]), j])
            errors = tours_i[i][4][1:ind] ./ opt_distances_i[i] .- 1
            data[j, i] = minimum(errors)
        end
    end

    average = mean(data, dims = 2)
    logav = vec(log.(average))
    dat = vec((collect(1:step_size:supN)))

    d_mean_fit = lm(@formula(T ~ X), DataFrame(X=dat, T=logav))
    c2 = coef(d_mean_fit)[2]
    c1 = exp(coef(d_mean_fit)[1])
    mdl = c1 .* exp.(collect(1:step_size:supN) .* c2)

    plti = plot(data, 
            xlabel = "Number of random initialisation points", 
            ylabel = "Fractional Error",
            title = "Performance scaling: $(dict_titles[s])",
            ylim = (0, maximum(data)),
            legend = false,
            label = nothing,
            color=RGB(0, 0.4470, 0.7410),
            alpha=0.2,
            dpi=500)

    plot!(plti, average, ylim=(0, 0.08), label="Average", legend=true, color=RGB(0.6350, 0.0780, 0.1840), alpha=1.0, linestyle=:dash, dpi=500)
    savefig(plti, "./figures/fig_initialisation_npoints_$(s).png")
end

