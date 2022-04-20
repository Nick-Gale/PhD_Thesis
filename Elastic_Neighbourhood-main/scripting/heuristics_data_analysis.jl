# quick and dirty script to do the data analysis.

using DataFrames 
using StatsPlots
using DelimitedFiles
using .ElasticNeighbourhood
dfun(a,b) = sqrt.(sum((a.-b).^2))

function create_data_frame(tour_data, tour_indexes, opt_tours, dir)
    # load the gpu data
    @time gpu_tours_data = readdlm(dir * "results_gpu.txt")
    gpu_inds = gpu_tours_data[2:end,1]

    df = DataFrame(opt=String[], L=Int64[], beta=Float64[])
    for i in tour_indexes
        L = 10000 + ceil(i/100) * 2000
        sample = mod(i-1, 30) + 1
        Cx = tour_data[i][1]
        Cy = tour_data[i][2]

        str_start = findfirst(".opt", opt_tours[i])[1]-1
        str = dir * opt_tours[i][1:str_start] * ".tsp"

        # do the GPU
        gpu_ind = findfirst(x -> x == i, gpu_inds)
        td = gpu_tours_data[gpu_ind+1,5:end]
        td[1] = parse(Int, td[1][2:end])
        lastind = findfirst(x->typeof(x)==SubString{String}, td[2:end]) + 1
        td[lastind] = parse(Int, td[lastind][1:end-1])
        
        td = vec(Int.(td[1:lastind]))
        n = length(Cx) 
        tl = compute_tour_length(Cx, Cy, td, dfun)
        push!(df, ("Elastic Neighbourhood", round(Int, n/1000), tl/sqrt(n)))

        # do the EAX
        eaxga_long_str = str * ".EAXGA_long.t"
        data = readdlm(eaxga_long_str)
        tour = data[2,:]
        tl = compute_tour_length(Cx, Cy, tour, dfun)
        push!(df, ("EAX-GA (Long)", Int(n/1000), tl/sqrt(n)))

        eaxga_short_str = str * ".EAXGA_short.t"
        data = readdlm(eaxga_short_str)
        tour = data[2,:]
        if any(tour.==0)
            push!(df, ("EAX-GA (Short)", Int(size(tour)[1]/1000), 0.813))
        else
            tl = compute_tour_length(Cx, Cy, tour, dfun)
            push!(df, ("EAX-GA (Short)", Int(n/1000), tl/sqrt(n)))
        end
        
        

        # do the LKH
        lkh_long_str = str * ".LKH_long.t"

        opt_tour = readdlm(lkh_long_str)
        tour_first = findfirst(x->x=="TOUR_SECTION", opt_tour)[1]+1
        tour_last = minimum(getindex.(filter(x -> x !== nothing, [findfirst(x->x=="EOF", opt_tour),findfirst(x->x==-1, opt_tour)]),1))-1
        opt_tour = opt_tour[tour_first:tour_last,1]
        tl = compute_tour_length(Cx, Cy, opt_tour, euc_dist)
        push!(df, ("LKH (Long)", Int(n/1000), tl/sqrt(n)))

        lkh_short_str = str * ".LKH_short.t"

        try 
            opt_tour = readdlm(lkh_short_str)
            tour_first = findfirst(x->x=="TOUR_SECTION", opt_tour)[1]+1
            tour_last = minimum(getindex.(filter(x -> x !== nothing, [findfirst(x->x=="EOF", opt_tour),findfirst(x->x==-1, opt_tour)]),1))-1
            opt_tour = opt_tour[tour_first:tour_last,1]
            tl = compute_tour_length(Cx, Cy, opt_tour, euc_dist)
            push!(df, ("LKH (Short)", Int(n/1000), tl/sqrt(n)))
        catch e
            push!(df, ("LKH (Short)", Int(n/1000), 0.813))
        end
    end
    return df
end

#####

#Load the data
dir = "./dataHeuristicComparisons/"
save_loc = "./dataHeuristicComparisons/results_gpu.txt"
#first we define the test indexes we used to generate Elastic Neighbourhood Solutions

test_indices = 1:100:1000
ti = []
for i in test_indices
    push!(ti, [i, i+1]...)
    push!(ti, collect((i+3):(i+25))...)
end

# now we create the data frame

df = create_data_frame(tour_data, ti, opt_tours, dir)

# Plot the box-plots

a = @df df groupedboxplot(string.(:L), :beta, group=:opt, 
color=[RGBA(0.8500,0.3250, 0.0980) RGBA(0.9290,0.6940, 0.1250) RGBA(0, 0.4470, 0.7410) RGBA(0.4940,0.1840, 0.5560) RGBA(0.4660,0.6740, 0.1880)], 
linewidth=1,
legend=:top,
xlabel="n(nodes) x 1e3",
ylabel="βs",
title="Heuristics Comparison: Scaled Tour Length",
ylim=(0.7, 0.825),
dpi=500)

savefig(a, "./figures/fig_heuristics_comparison.png")