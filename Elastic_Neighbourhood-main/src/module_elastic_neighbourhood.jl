module ElasticNeighbourhood
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THE ELASTIC NEIGHBOURHOOD -- Author: Nicholas Gale
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
export adam!, adamw!, nadam!, momentum!, gd!, amsgrad!, nesterov!
export batch_tours, euc_dist, path_inj, find_crossing, intersection, compute_tour_length, compute_tour
export plot_figs, plot_analyse, plot_clusters, plot_animation, plot_time_series
export data_load, scan_file, write_tour
export EN
export elastic_neighbourhood

#functions
using Plots, Interpolations, DelimitedFiles, CUDA, Distributed, Random, Distributions, Optim, Revise

include("types.jl")
include("plotting.jl")
include("path_operations.jl")
include("workers.jl")
include("io.jl")
include("optimisers.jl")
include("mnpeano.jl")
include("elastic_neighbourhood_logic.jl")

# @everywhere includet("types.jl")
# @everywhere includet("plotting.jl")
# @everywhere includet("path_operations.jl")
# @everywhere includet("workers.jl")
# @everywhere includet("io.jl")
# @everywhere includet("optimisers.jl")
# @everywhere includet("mnpeano.jl")
# @everywhere includet("elastic_neighbourhood_logic.jl")

#
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Deployment
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

function _elastic_neighbourhood_vanilla(cx, cy, params)
    #construct several nets over the batch of multiple initialisation points
    n_inits = get(params, :n_inits, 1)
    function _wrapper(x, cx, cy, params)
        params[:init_seed]=x; 
        return EN(cx, cy; params...)
    end
    init_nets = map(x -> _wrapper(x, cx, cy, params), 1:n_inits)
    
    #distribute the work across multiple processes
    @time res = pmap(x -> _run_elastic_neighbourhood(x), init_nets)
    display("Above: time to run all GPU calculations")

    # batch the results
    Pxs = [res[i][1] for i = 1:n_inits]
    Pys = [res[i][2] for i = 1:n_inits]

    # if the evolution was recorded pull the data
    if params[:record] == true
        px_record = res[1][3]
        py_record = res[1][4]
    end

    # delete the references
    init_nets = nothing
    res = nothing
    
    # compute the actual tours and force garbage collection
    tour, px_return, py_return, trace = compute_tour_batch(cx, cy, Pxs, Pys, 1:n_inits) 
    pmap(x -> GC.gc(true), 1:n_inits)
    pmap(x -> CUDA.reclaim(), 1:n_inits)

    # if refinement is specified unwind all potential crossing
    if params[:refine] == true
        crossings = []
        tour = tour
        test = true
        while test
            crossings = find_crossings(cx, cy, Int.(tour))
            if length(crossings) >= 1
                tour = unwind_crossings(cx, cy, Int.(tour), crossings)
                test = true
            else
                test = false 
            end
        end
    end
    
    # return based on deployment parameters
    if params[:record] == true
        return tour, px_return, py_return, trace, px_record, py_record
    else
        return tour, px_return, py_return, trace
    end
end

function _elastic_neighbourhood_optim(cx, cy, params)
    """Deploying the Elastic Neighbourhood as a loss function to an optimisation routine with Elastic Neighbourhood hyper-parameters as the search space. Currently explores the intialisation location of a circular net. """
    function opt_fun(w)
        # Function operates on the last explored value. Need to change the 
        params[:x_init] = w[1]
        params[:y_init] = w[2]
        params[:init_seed] = 1
        net = EN(cx, cy; params...)
        px, py = _run_elastic_neighbourhood(net)
        tour_current = compute_tour(cx, cy, px, py)
        tour_length_current = compute_tour_length(cx, cy, tour_current, euc_dist)
        return tour_length_current
    end
    
    w0 = [params[:x_init], params[:y_init]]
    wupper = [0.5, 0.5]
    wlower = [-0.5, -0.5]
    
    if params[:initialisation_mode] == "grid"
        return _elastic_neighbourhood_vanilla(cx, cy, params)
    end

    if params[:initialisation_mode] == "random"
        return _elastic_neighbourhood_vanilla(cx, cy, params)
    end

    if params[:initialisation_mode]  == "simulated_annealing"
        res = Optim.optimize(opt_fun, wlower, wupper, w0, SAMIN(), Optim.Options(iterations=params[:n_inits], store_trace=true, show_trace=true))
    end

    if params[:initialisation_mode]  == "particle_swarm"
        res = Optim.optimize(opt_fun, w0, ParticleSwarm(lower=wlower, upper=wupper, n_particles=10), Optim.Options(iterations=params[:n_inits], store_trace=true, show_trace=true))
    end
    
    if params[:initialisation_mode]  == "nelder_mead"
        res = Optim.optimize(opt_fun, w0, NelderMead(), Optim.Options(iterations=params[:n_inits], store_trace=true, show_trace=true, g_tol=1e-100))
    end
    
    println(Optim.minimizer(res))
    println("The procedure converged: $(Optim.converged(res))")
    
    clusters = Optim.f_trace(res)
    wmin = Optim.minimizer(res)
    
    params[:x_init] = wmin[1]
    params[:y_init] = wmin[2]
    net = EN(cx, cy; params...)
    
    px_return, py_return = _run_elastic_neighbourhood(net)
    tour_return = compute_tour(cx, cy, px_return, py_return)
    return tour_return, px_return, py_return, clusters
end

function elastic_neighbourhood(cx, cy; params=Dict())
    """Universal function to execute the Elastic Neighbourhood optimisation method. 
    The first three return arguments are consistent: the tour, the x-positions of the final state of the net, and the y-positions of the final state of the net.
    The fourth return argument is the trace of all the examined tour costs (if using an optimisation routine or multiple initialisations), the tour cost   
    If there is no dictionary of parameters passed the backend will try its best. 
    The default is to use a fractal based initialisation mode without recording or refinement in Float32 precision. Inspect the EN constructors for more details."""
    # this method needs to be extended via Multiple Dispatch to allow for multiple possible TSP data types.
    if get(params, :refine, nothing) == nothing
        params[:refine] = false
    end

    if get(params, :record, nothing) == nothing
        params[:record] = false
    end

    if get(params, :time_status, nothing) == nothing
        params[:time_status] = false
    end

    if get(params, :initialisation_mode, nothing) == nothing
        params[:initialisation_mode] = "fractal"
    end 

    if get(params, :x_init, nothing) == nothing
        params[:x_init] = 0.0
    end

    if get(params, :y_init, nothing) == nothing
        params[:y_init] = 0.0
    end

    if get(params, :n_inits, nothing) == nothing
        params[:n_inits] = 1
    end

    if any(get(params, :initialisation_mode, nothing) .== ["grid", "random", "simulated_annealing", "nelder_mead", "particle_swarm"])
        return _elastic_neighbourhood_optim(cx, cy, params)
    else
        return _elastic_neighbourhood_vanilla(cx, cy, params)
    end
end

#
end