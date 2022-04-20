using GLM, DataFrames, StatsBase, ProgressMeter, StatsPlots, StatsBase, Plots
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This script generates the initialisation domains and the scaling for a small dataset
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Choose which tours you would like to solve
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dir = "./dataRandom/" # "./RandomEuclidean/" # 
save_loc = "./results_art.txt"
file_names = readdir(dir)
opt_tours = filter(x->occursin(".tour", x), file_names)
tour_data = data_load(file_names, opt_tours, dir)
tour_lengths = [length(tour_data[i][1]) for i in 1:length(tour_data)]
sorting = sortperm(tour_lengths)
tour_lengths = tour_lengths[sorting]
tour_data = tour_data[sorting]
unique_tour_lengths = unique(tour_lengths)[1:10]

test_indices = map(x -> findfirst(x .== unique_tour_lengths), tour_lengths)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Wall time length dependence
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
params_dict = Dict(:n_inits => 10, :initialisation_mode => "random")
times10 = zeros(length(unique_tour_lengths))
for i = 1:length(times10)
    ind = findfirst(x -> unique_tour_lengths[i] == x, tour_lengths)
    times10[i] = @elapsed batch_tours(params_dict, test_indices[ind], tour_data)
end

params_dict = Dict(:n_inits => 1 , :initialisation_mode => "random")
times = zeros(length(unique_tour_lengths))
for i = 1:length(times)
    ind = findfirst(x -> unique_tour_lengths[i] == x, tour_lengths)
    times[i] = @elapsed batch_tours(params_dict, test_indices[ind], tour_data)
end

params_dict = Dict(:n_inits => 5, :initialisation_mode => "random")
times5 = zeros(length(unique_tour_lengths))
for i = 1:length(times5)
    ind = findfirst(x -> unique_tour_lengths[i] == x, tour_lengths)
    times5[i] = @elapsed batch_tours(params_dict, test_indices[ind], tour_data)
end

d_x = log.(unique_tour_lengths)
d_t = log.(times)
d_t5 = log.(times5)
d_t10 = log.(times10)
d_fit = lm(@formula(T ~ X), DataFrame(X=d_x, T=d_t))
d_fit5 = lm(@formula(T ~ X), DataFrame(X=d_x, T=d_t5))
d_fit10 = lm(@formula(T ~ X), DataFrame(X=d_x, T=d_t10))

runtime_plt = plot(dpi=500, xlabel="log(N) nodes", ylabel="log(WT) seconds", title="Wall-time Scaling")
plot!(runtime_plt, d_x, d_t, seriestype=:line, markershape=:rect, alpha=0.5, color=RGB(0, 0.4470, 0.7410), label="1 Initialisation")
plot!(runtime_plt, d_x, d_t5, seriestype=:line, markershape=:rect, alpha=0.5, color=RGB(0.6350, 0.0780, 0.1840), label="5 Initialisations")
plot!(runtime_plt, d_x, d_t10, seriestype=:line, markershape=:rect, alpha=0.5, color=RGB(0, 0, 0), label="10 Initialisations")
annotate!(runtime_plt, minimum(d_x) + 0.5, maximum(d_t10) - 0.0, text("1: Linear Fit: $(round(coef(d_fit)[2], digits=2))x +$(round(coef(d_fit)[1], digits=2)) \n R² = $(round(r2(d_fit), digits=3))", :black, 10))
annotate!(runtime_plt, minimum(d_x) + 0.5, maximum(d_t10) - 1 , text("5: Linear Fit: $(round(coef(d_fit5)[2], digits=2))x +$(round(coef(d_fit5)[1], digits=2)) \n R² = $(round(r2(d_fit5), digits=3))", :black, 10))
annotate!(runtime_plt, minimum(d_x) + 0.5, maximum(d_t10) - 2, text("10: Linear Fit: $(round(coef(d_fit10)[2], digits=2))x +$(round(coef(d_fit10)[1], digits=2)) \n R² = $(round(r2(d_fit10), digits=3))", :black, 10))
savefig(runtime_plt, "./figures/fig_scaling_runtime.png")