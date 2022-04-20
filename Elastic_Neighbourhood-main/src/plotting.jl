#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting Functions.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function plot_figs(cx, cy, opt_tour, tours, DPI; display_nodes=false, save_name="")
    gr()
    fig = plot(xlims=(-0.5,0.5), ylims=(-0.5,0.5), axis_buffer=0.0, markeralpha=0.0, linealpha=0.5, size=(640,640), dpi=DPI, aspect_ratio=:equal, legend=false, show=false)
    plot!(fig, tours[2], tours[3], seriestype=:scatterpath, color=:blue,  markeralpha=0.0, linealpha=0.5);
    if display_nodes
        #plot!(fig, vcat(cx[Int.(tours[1])], cx[Int.(tours[1][1])]), vcat(cy[Int.(tours[1])],cy[Int.(tours[1][1])]), seriestype=:scatterpath, color=:red, markeralpha=0.0, linealpha=0.5);
        plot!(fig, cx, cy, seriestype=:scatter, color=:red,  markeralpha=0.5, linealpha=0.5);
    end
    
    savefig(fig, "./figures/fig_" * save_name * ".png");
end

function plot_analyse(plot_test, test_indices, tour_data, tours, opt_tours, computed_distances, opt_distances; DPI = 500, display_nodes=true, save_name="")
    cx = Array{Float64}(tour_data[test_indices[plot_test]][1])
    cy = Array{Float64}(tour_data[test_indices[plot_test]][2])
    opt_tour = Array{Int64}(tour_data[test_indices[plot_test]][3])
    tour = tours[plot_test]

    println("Average tour difference: $(1 - sum(opt_distances ./ computed_distances)/length(test_indices))")
    plot_figs(cx, cy, opt_tour, tour, DPI; display_nodes=display_nodes, save_name=save_name)
end

function plot_clusters(clusters, initialisation_mode, n_inits; DPI = 500, mz=8)
    color_map = map(x -> RGB(0.5, maximum([x - 1e-3, 0]), 0.5), log.(clusters) ./ maximum(log.(clusters)))
    if initialisation_mode == "grid"
        n_steps = round(Int, sqrt(n_inits))
        X = map(init_seed -> -0.5 + (mod(init_seed - 1, n_steps)) / (n_steps-1), 1:n_inits)
        Y = map(init_seed -> -0.5 + (ceil(init_seed / n_steps) - 1) / (n_steps-1), 1:n_inits)
    elseif initialisation_mode == "random"
        X = zeros(n_inits)
        Y = zeros(n_inits)
        for i = 1:n_inits 
            Random.seed!(i)
            X[i] = rand() - 0.5
            Y[i] = rand() - 0.5
        end
    end
    # fix the color bar
    append!(X, 0.5)
    append!(Y, 0.5)
    append!(clusters, 1.0)

    append!(X, 0.5)
    append!(Y, 0.5)
    append!(clusters, 1.1)
        
    plt = plot(X, Y, marker_z=clusters, xlims=(-0.5, 0.5), ylims=(-0.5, 0.5), markersize=mz, markerstrokewidth=0.0, st=:scatter, color=:thermal, legend=false, colorbar=:bottom)
    return plt
end

function plot_series(cx, cy, px, py, fraction, title_str, DPI; display_nodes=false, ms=0.01)
    px = px[1]
    append!(px, px[1])
    py = py[1]
    append!(py, py[1])
    gr()
    fig = plot(ticks=nothing, xlims=(-0.5, 0.5), ylims=(-0.5, 0.5), axis_buffer=0.0, markeralpha=0.0, linealpha=0.5, size=(640,640), dpi=DPI, aspect_ratio=:equal, legend=false, show=false)
    plot!(fig, px, py, seriestype=:scatterpath, color=:blue,  markeralpha=0.0, linealpha=0.5);
    if display_nodes
        plot!(fig, cx, cy, seriestype=:scatter, color=:red,  markeralpha=0.5, linealpha=0.0, markersize=ms);
    end

    return fig
end

function plot_animation(cx, cy, tour, savename=""; DPI = 500)
    ln = length(tour[1])
    animation = @animate for j = 1:(ceil(Int, ln/100)): 1.25 * ln
        i = Int(minimum([j, ln]))
        frac = 1/100;
        plot_series(cx, cy, tour[5][i], tour[6][i], frac, "Elastic Neighbourhood", 300)
    end
    gif(animation, "./figures/savename.gif", fps=24)
end

function plot_time_series(cx, cy, tour; DPI = 500, ms=0.01)
    ln = length(tour[5])
    plt_eneighbour = []
    step = round(Int, ln/9)
    for j = 1:9
        i = (j - 1) * step + 1
        frac = i/9
        push!(plt_eneighbour, plot_series(cx, cy, tour[5][i], tour[6][i], frac, "Elastic Neighbourhood", 500; display_nodes=true, ms=ms))
    end

    return plot(plt_eneighbour..., layout=(3,3), title = ["0.04 T" "0.16 T" "0.28 T" "0.4 T" "0.52 T" "0.64 T" "0.76 T" "0.88 T" "T"])
end