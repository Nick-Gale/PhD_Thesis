using LinearAlgebra
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Constructor Functions.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
mutable struct EN64
    #status
    time_status::Bool
    record::Bool
    #functions
    optimiser::String
    #parameters
    Lp::Int64
    Lc::Int64
    threads::Int64
    Bp::Int64
    Bc::Int64

    epsilon::Float64
    boundary::Float64

    max_iterations::Int64
    stability::Int64

    a::Float64
    b::Float64
    c::Float64
    t_amp::Float64
    t_scale::Float64
    init_radius::Float64
    init_seed::Int64

    alpha::Float64
    beta::Float64
    eta::Float64

    #arrays
    temperature_distribution::Array{Float64, 1}
    Cx::Array{Float64, 1}
    Cy::Array{Float64, 1}
    Px::Array{Float64, 1}
    Py::Array{Float64, 1}
    Arrays::Array{Array{Float64,1},1}

    #constructor function
    function EN64(cx, cy; init_seed=1, n_inits=1, init_radius=0.0,  max_iterations=25, stability=10, n_neighbourhoods=3, a=1, b=1, c=1, alpha=0.9, beta=0.999, eta=0.01, optimiser="adam!", t_amp=1.0, t_scale = 0.5, epsilon=10^-7, boundary = 0.6, threads=16, initialisation_mode="random", t0 = 0.5, x_init=0.0, y_init=0.0, precision=Float64, refine=false, time_status=true, record=false)

        Lp, Lc, Bp, Bc, a, b, c, t_rescaled, eta, t_amp_rescaled, max_iterations_rescaled, stability_rescaled, temperature_distribution, Cx, Cy, Px, Py, Arrays = set_parameters(cx, cy, n_neighbourhoods, n_inits, threads, a, b, c, t_scale, t_amp, max_iterations, stability, eta, initialisation_mode, init_radius, init_seed, x_init, y_init, precision)
        Arrays = Array{Float64, 1}.(Arrays)

        new(
        time_status,
        record,
        optimiser,
        Lp,
        Lc,
        threads,
        Bp,
        Bc,
        epsilon,
        boundary,
        max_iterations_rescaled,
        
        stability_rescaled,
        a,
        b,
        c,
        t_amp,
        t_scale,
        init_radius,
        init_seed,
        alpha,
        beta,
        eta,
        temperature_distribution,
        Cx,
        Cy,
        Px,
        Py,
        Arrays
        )
    end
end

mutable struct EN32
    #status
    time_status::Bool
    record::Bool
    #functions
    optimiser::String
    #parameters
    Lp::Int32
    Lc::Int32
    threads::Int32
    Bp::Int32
    Bc::Int32

    epsilon::Float32
    boundary::Float32

    max_iterations::Int32
    stability::Int32

    a::Float32
    b::Float32
    c::Float32
    t_amp::Float32
    t_scale::Float32
    init_radius::Float32
    init_seed::Int32

    alpha::Float32
    beta::Float32
    eta::Float32

    #arrays
    temperature_distribution::Array{Float32, 1}
    Cx::Array{Float32, 1}
    Cy::Array{Float32, 1}
    Px::Array{Float32, 1}
    Py::Array{Float32, 1}
    Arrays::Array{Array{Float32,1},1}

    #constructor function
    function EN32(cx, cy; init_seed=1, n_inits=1, init_radius=0.0,  max_iterations=25, stability=10, n_neighbourhoods=3, a=1, b=1, c=1, alpha=0.9, beta=0.999, eta=0.01, optimiser="adam!", t_amp=1.0, t_scale = 0.5, epsilon=10^-7, boundary = 0.6, threads=16, initialisation_mode="random", t0 = 0.5, x_init=0.0, y_init=0.0, precision=Float32, refine=false, time_status=false, record=false)

        Lp, Lc, Bp, Bc, a, b, c, t_rescaled, eta, t_amp_rescaled, max_iterations_rescaled, stability_rescaled, temperature_distribution, Cx, Cy, Px, Py, Arrays = set_parameters(cx, cy, n_neighbourhoods, n_inits, threads, a, b, c, t_scale, t_amp, max_iterations, stability, eta, initialisation_mode, init_radius, init_seed, x_init, y_init, precision)
        Arrays = Array{Float32, 1}.(Arrays)
        
        new(
        time_status,
        record,
        optimiser,
        Lp,
        Lc,
        threads,
        Bp,
        Bc,
        epsilon,
        boundary,
        max_iterations_rescaled,
        stability_rescaled,
        a,
        b,
        c,
        t_amp,
        t_scale,
        init_radius,
        init_seed,
        alpha,
        beta,
        eta,
        temperature_distribution,
        Cx,
        Cy,
        Px,
        Py,
        Arrays
        )
    end
end

function EN(cx, cy; init_seed=1, n_inits=1, init_radius=0.0, max_iterations=25, stability=10, n_neighbourhoods=3, a=1, b=1, c=1, alpha=0.9, beta=0.999, eta=0.01, optimiser="adam!", t_amp=0.1, t_scale = 0.99, epsilon=10^-7, boundary = 0.5125, threads=16, initialisation_mode="grid", optim_routine="simulated_annealing", t0=0.5, x_init=0.0, y_init=0.0, precision=Float32, refine=false, time_status=false, record=false)
    if precision == Float64
        e64 = EN64(cx, cy; init_seed=init_seed, n_inits=n_inits, init_radius=init_radius, max_iterations=max_iterations, stability=stability, n_neighbourhoods=n_neighbourhoods, a=a, b=b, c=c, alpha=alpha, beta=beta, eta=eta, optimiser=optimiser, t_amp=t_amp, t_scale=t_scale, epsilon=epsilon, boundary=boundary, threads=threads, initialisation_mode=initialisation_mode, t0=t0, x_init=x_init, y_init=y_init,  precision=precision, refine=refine, time_status=time_status, record=record)
        return e64
    end
    
    if precision == Float32
        e32 = EN32(cx, cy; init_seed=init_seed, n_inits=n_inits, init_radius=init_radius, max_iterations=max_iterations, stability=stability, n_neighbourhoods=n_neighbourhoods, a=a, b=b, c=c, alpha=alpha, beta=beta, eta=eta, optimiser=optimiser, t_amp=t_amp, t_scale=t_scale, epsilon=epsilon, boundary=boundary, threads=threads, initialisation_mode=initialisation_mode, t0=t0, x_init=x_init, y_init=y_init, precision=precision, refine=refine, time_status=time_status, record=record)
        return e32
    end
end

function set_parameters(cx, cy, n_neighbourhoods, n_inits, threads, a, b, c, t_scale, t_amp, max_iterations, stability, eta, initialisation_mode, init_radius, init_seed, x_init, y_init, precision)
    # set the dimensions of the problem and preallocate memory space
    Lp = round(Int, length(cx) * n_neighbourhoods)
    Lc = length(cx)
    Arrays = pre_allocate(Lp, Lc, precision)

    # set the launch context for CUDA
    Bp = ceil(Int, Lp / threads)
    Bc = ceil(Int, Lc / threads)

    # set the parameters specific to the elastic neighbourhood
    b = b
    a = a * 0.712 * sqrt(Lc) / Lp # sqrt(1/2) * sqrt(Lc) / Lp
    c = c
    # set the data derived parameters/hyper parameters
    eta_rescaled = eta # this currently is not changing but potentially can be scaled as a function of data in future hyper-parameter optimisation
    if init_radius == 0
        init_radius = 1/sqrt(4 * Lc)
    end
    stability_rescaled = round(Int32, 0.5/eta_rescaled)
    max_iterations_rescaled = max_iterations
    t_rescaled = 1/2*log(Lc)
    t_amp_rescaled = t_amp
    temperature_distribution = t_amp_rescaled .* exp.(-collect(0:(max_iterations-1)) ./ max_iterations .* t_rescaled)

    if initialisation_mode == "grid"
        x, y = initialisation_grid(init_seed, n_inits, cx, cy)
        Px, Py = path_init_circle(n_neighbourhoods, cx, cy, init_radius, x, y)
    elseif initialisation_mode == "random"
        s = 1/Lc
        n = MixtureModel(MvNormal[[MvNormal([cx[i], cy[i]], s*I(2)) for i in 1:Lc]...]);
        x, y = initialisation_random(init_seed, cx, cy; model = n)
        Px, Py = path_init_circle(n_neighbourhoods, cx, cy, init_radius, x, y)
    elseif initialisation_mode == "fractal"
        Px, Py = path_init_mpeano(Lp)
        stability_rescaled = round(Int, 5/eta)
        t_amp_rescaled = 100 * t_amp * 0.712/(sqrt(Lc) * n_neighbourhoods) #10 * 0.712/(sqrt(Lc) * n_neighbourhoods)
        temperature_distribution = t_amp_rescaled .* (0.5) .^ (1:5)
        max_iterations_rescaled = length(temperature_distribution)
    elseif initialisation_mode == "elastic_net"
        x, y = initialisation_grid(init_seed, n_inits, cx, cy)
        Px, Py = path_init_circle(n_neighbourhoods, cx, cy, 0.2, x, y)

        stability_rescaled = 25
        a = 0.2 * sqrt(Lc/2) / Lp
        t_amp_rescaled = 0.2
        max_iterations_rescaled = round(Int, 60 * log(Lc)) 
        t_rescaled = 0.5 * log(Lc) #- log(0.2)
        temperature_distribution = t_amp_rescaled .* exp.(-((0:(max_iterations_rescaled-1)) ./ max_iterations_rescaled) .* t_rescaled)
    else
        # assume the user has provided a starting location most likely on the basis of a starting routine
        
        Px, Py = path_init_circle(n_neighbourhoods, cx, cy, init_radius, x_init, y_init)
    end
    return Lp, Lc, Bp, Bc, a, b, c, t_rescaled, eta_rescaled, t_amp_rescaled, max_iterations_rescaled, stability_rescaled, temperature_distribution, cx, cy, Px, Py, Arrays
end

function pre_allocate(L_P, L_C, precision)
    """A convenience function to assign data locations for all the various arrays used in the GPU calculations."""

    #gradient calculation matrices
    normaliser_nodes_x = zeros(precision, L_C)
    normaliser_nodes_y = zeros(precision, L_C)
    normaliser_net_x =zeros(precision, L_P)
    normaliser_net_y = zeros(precision, L_P)
    dPnodes_x = zeros(precision, L_P)
    dPnodes_y = zeros(precision, L_P)
    dPnet_x = zeros(precision, L_P)
    dPnet_y = zeros(precision, L_P)
    laplacian_x = zeros(precision, L_P)
    laplacian_y = zeros(precision, L_P)

    #optimiser matrices
    update_x = zeros(precision, L_P)
    m_x = zeros(precision, L_P)
    v_x = zeros(precision, L_P)
    M_x = zeros(precision, L_P)

    update_y = zeros(precision, L_P)
    m_y = zeros(precision, L_P)
    v_y = zeros(precision, L_P)
    M_y = zeros(precision, L_P)

    #gradient
    grad_x = zeros(precision, L_P)
    grad_y = zeros(precision, L_P)
    return [normaliser_nodes_x, normaliser_nodes_y, normaliser_net_x, normaliser_net_y, dPnodes_x, dPnodes_y, dPnet_x, dPnet_y, laplacian_x, laplacian_y, update_x, m_x, v_x, M_x, grad_x, update_y, m_y, v_y, M_y, grad_y]
end
