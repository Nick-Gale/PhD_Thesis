#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Elastic Neighbourhood Functions
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
using CUDA
using Random
CUDA.seed!(1)

function laplacian!(laplacian_vector, P, Lp)
    """GPU compute the laplacian of the net."""
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Lp
        @inbounds laplacian_vector[i] = P[Int(mod(i-2, Lp) + 1)] - 2 * P[Int(mod(i-1, Lp) + 1)] + P[Int(mod(i, Lp) + 1)]
    end
    return nothing
end

function normaliser!((K, normaliser, Cprimary, Cauxillary, Pprimary, Pauxillary, Lp, Lc, B, T))
    """GPU compute the normaliser for the nodes(city)/net weights on the basis of data given by Cprimary, Cauxillay"""
    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    test = (indy <= Lp) && (indx <= Lc)
    if test
        @inbounds @atomic normaliser[indx] += exp(((Cprimary[indx] - Pprimary[indy])^2 + (Cauxillary[indx] - Pauxillary[indy])^2) / (-2 * K^2))  
    end
    return nothing
end

function dw!((K, dP, normaliser, Cprimary, Cauxillary, Pprimary, Pauxillary, epsilon, Lp, Lc, B, T))
    """GPU compute of gradient contribution from interatctions with nodes or net along primary axis (auxillary indicates the axis locations needed for calculation, but not output)"""
    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    shared = @cuDynamicSharedMem(eltype(Cprimary), (T, T))

    test = (indy <= Lc) && (indx <= Lp)
    if test
        @inbounds shared[threadIdx().x, threadIdx().y] = exp(((Cprimary[indy] - Pprimary[indx])^2 + (Cauxillary[indy] - Pauxillary[indx])^2) / (-2 * K^2)) / (normaliser[indy] + epsilon) * (Cprimary[indy] - Pprimary[indx])
    end
    sync_threads()
    if test
        @inbounds @atomic dP[indx] += shared[threadIdx().x, threadIdx().y] # exp(((Cprimary[indy] - Pprimary[indx])^2 + (Cauxillary[indy] - Pauxillary[indx])^2) / (-2 * K^2)) / (normaliser[indy] + epsilon) * (Cprimary[indy] - Pprimary[indx]) #
    end
    return nothing
end

function grad!((K, a, b, c, grad, Cprimary, Cauxillary, Pprimary, Pauxillary, dPnodes, dPnet, dPlaplacian, normaliser_nodes, normaliser_net, epsilon, Lp, Lc, Bp, Bc, T))
    #compute the gradient components along the primary direction
    s1 = CUDA.CuStream()
    s2 = CUDA.CuStream()
    s3 = CUDA.CuStream()
    smem_size = (sizeof(eltype(Cprimary)) * (T+1) * (T+1))

    @cuda blocks=Bp threads=T stream=s3 laplacian!(dPlaplacian, Pprimary, Lp) 
    @cuda blocks=(Bc, Bp) threads=(T, T) stream=s1 shmem = smem_size normaliser!((K, normaliser_nodes, Cprimary, Cauxillary, Pprimary, Pauxillary, Lp, Lc, Bp, T)) 
    @cuda blocks=(Bp, Bp) threads=(T, T) stream=s2 shmem = smem_size normaliser!((K, normaliser_net, Pprimary, Pauxillary, Pprimary, Pauxillary, Lp, Lp, Bp, T))
    
    synchronize(s1); synchronize(s2); synchronize(s3);
    
    @cuda blocks=(Bp, Bc) threads=(T, T) stream=s1 shmem = smem_size dw!((K, dPnodes, normaliser_nodes, Cprimary, Cauxillary, Pprimary, Pauxillary, epsilon, Lp, Lc, Bc, T))
    @cuda blocks=(Bp, Bp) threads=(T, T) stream=s2 shmem = smem_size dw!((K, dPnet, normaliser_net, Pprimary, Pauxillary, Pprimary, Pauxillary, epsilon, Lp, Lp, Bp, T))
    synchronize(s1); synchronize(s2); synchronize(s3);
    
    #compute the gradient with scaling
    grad .= (c .* dPnet .- a .* dPnodes .- (b .* K) .* dPlaplacian)
    dPnodes .= 0; dPnet .= 0; normaliser_nodes .= 0; normaliser_net .= 0;
    return nothing
end

function neighbourhood_update!(t_long, t_short, K, a, b, c, alpha, beta, eta, boundary, stability, optimiser, 
    Cx, Cy, Px, Py, normaliser_nodes_x, normaliser_nodes_y, normaliser_net_x, normaliser_net_y, dPnodes_x, dPnodes_y, dPnet_x, dPnet_y, dPlaplacian_x, dPlaplacian_y, update_x, m_x, v_x, M_x, grad_x, update_y, m_y, v_y, M_y, grad_y,
    epsilon, Lp, Lc, Bp, Bc, T)
    """Runs the neighbourhood update function with a given set of parameters: K, a, b, c. Can be augmented with an optional set of a parameters related to the optimiser being used: alpha, beta, eta. Finally, a boundary parameter can be specified which defaults to 0.6 as the data is typically scaled to be in the domain [-0.5,0.5]x[-0.5,0.5].
        The function runs by computing the gradient according to the network of data and then applying an optimiser over this gradient. The optimiser is most typically ADAM.
    """

    #update x
    grad!((K, a, b, c, grad_x, Cx, Cy, Px, Py, dPnodes_x, dPnet_x, dPlaplacian_x, normaliser_nodes_x, normaliser_net_x, epsilon, Lp, Lc, Bp, Bc, T))
    optimiser(alpha, beta, eta, (t_long-1) * stability + t_short, Px, update_x, m_x, v_x, M_x, grad_x, epsilon)
    Px .= Px .- update_x # (abs.(Px .- update_x) .< boundary) .* (Px .- update_x) .+ (abs.(Px .- update_x) .> boundary) .* boundary .* sign.(Px .- update_x) # 
    
    #update y
    grad!((K, a, b, c, grad_y, Cy, Cx, Py, Px, dPnodes_y, dPnet_y, dPlaplacian_y, normaliser_nodes_y, normaliser_net_y, epsilon, Lp, Lc, Bp, Bc, T))
    optimiser(alpha, beta, eta, (t_long-1) * stability + t_short, Py, update_y, m_y, v_y, M_y, grad_y, epsilon)
    Py .= Py .- update_y # (abs.(Py .- update_y) .< boundary) .* (Py .- update_y) .+ (abs.(Py .- update_y) .> boundary) .* boundary .* sign.(Py .- update_y) #  
    return nothing 
end

function hot_dw!((K, normaliser, dPx, dPy, Cx, Cy, Px, Py, epsilon, Lp, Lc, B, T))
    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    test = (indy <= Lc) && (indx <= Lp)
    if test
        res = exp(((Cx[indy] - Px[indx])^2 + (Cy[indy] - Py[indx])^2) / (-2 * K^2)) / (normaliser[indy] + epsilon)
        @inbounds @atomic dPx[indx] += res * (Cx[indy] - Px[indx])
        @inbounds @atomic dPy[indx] += res * (Cy[indy] - Py[indx])
    end
    return nothing
end

function hot_laplacian!(laplacian_vectorx, laplacian_vectory, Px, Py, Lp)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Lp
        @inbounds laplacian_vectory[i] = Py[Int(mod(i-2, Lp) + 1)] - 2 * Py[Int(mod(i-1, Lp) + 1)] + Py[Int(mod(i, Lp) + 1)]
        @inbounds laplacian_vectorx[i] = Px[Int(mod(i-2, Lp) + 1)] - 2 * Px[Int(mod(i-1, Lp) + 1)] + Px[Int(mod(i, Lp) + 1)]
    end
    return nothing
end

function hot_grad!((K, a, b, c, gradx, grady, Cx, Cy, Px, Py, 
    dPnodesx, dPnetx, dPlaplacianx, dPnodesy, dPnety, dPlaplaciany,
    normaliser_nodes, normaliser_net, epsilon, Lp, Lc, Bp, Bc, T))
    # A less modular but slightly more efficient implementation of grad, it only computes necessary values for dw once rather than twice.

    s1 = CUDA.CuStream()
    s2 = CUDA.CuStream()
    s3 = CUDA.CuStream()
    smem_size = (sizeof(eltype(Cx)) * T * T)

    CUDA.@sync @cuda blocks=Bp threads=T stream=s1 hot_laplacian!(dPlaplacianx, dPlaplaciany, Px, Py, Lp) 
    CUDA.@sync @cuda blocks=(Bc, Bp) threads=(T, T) stream=s2 shmem=smem_size normaliser!((K, normaliser_nodes, Cx, Cy, Px, Py, Lp, Lc, Bp, T))
    CUDA.@sync @cuda blocks=(Bp, Bp) threads=(T, T) stream=s3 shmem=smem_size normaliser!((K, normaliser_net, Py, Px, Py, Px, Lp, Lp, Bp, T))
    synchronize(s1); synchronize(s2); synchronize(s3);
    
    CUDA.@sync @cuda blocks=(Bp, Bc) threads=(T, T) stream=s1 shmem=smem_size hot_dw!((K, normaliser_nodes, dPnodesx, dPnodesy, Cx, Cy, Px, Py, epsilon, Lp, Lc, Bc, T))
    CUDA.@sync @cuda blocks=(Bp, Bp) threads=(T, T) stream=s2 shmem=smem_size hot_dw!((K, normaliser_net, dPnetx, dPnety, Px, Py, Px, Py, epsilon, Lp, Lp, Bp, T))
    synchronize(s1); synchronize(s2); synchronize(s3);
    
    #compute the gradient with scaling
    grady .= (c .* dPnety .- a .* dPnodesy .- (b .* K) .* dPlaplaciany)
    gradx .= (c .* dPnetx .- a .* dPnodesx .- (b .* K) .* dPlaplacianx)
    return nothing
end

function hot_neighbourhood_update!(t_long, t_short, K, a, b, c, alpha, beta, eta, boundary, stability, optimiser, 
    Cx, Cy, Px, Py, 
    normaliser_nodes_x, normaliser_nodes_y, normaliser_net_x, normaliser_net_y, 
    dPnodes_x, dPnodes_y, dPnet_x, dPnet_y, dPlaplacian_x, dPlaplacian_y, 
    update_x, m_x, v_x, M_x, grad_x, 
    update_y, m_y, v_y, M_y, grad_y,
    epsilon, Lp, Lc, Bp, Bc, T)
    """Performs the same calculations as neighbourhood_update but does both directions simultaneously. This is performed by calling the standard functions with hot_* prefixes. This method is not intuitively extensible to dimensionality other the two."""

    # compute gradients
    hot_grad!((K, a, b, c, grad_x, grad_y, Cx, Cy, Px, Py, 
    dPnodes_x, dPnet_x, dPlaplacian_x, 
    dPnodes_y, dPnet_y, dPlaplacian_y,
    normaliser_nodes_y, normaliser_net_y, epsilon, Lp, Lc, Bp, Bc, T))

    #update x
    optimiser(alpha, beta, eta, (t_long-1) * stability + t_short, Px, update_x, m_x, v_x, M_x, grad_x, epsilon)
    Px .= Px .- update_x
    #update y
    optimiser(alpha, beta, eta, (t_long-1) * stability + t_short, Py, update_y, m_y, v_y, M_y, grad_y, epsilon)
    Py .= Py .- update_y
    
    dPnodes_x .= 0; dPnet_x .= 0; 
    dPnodes_y .= 0; dPnet_y .= 0; 
    normaliser_nodes_y .= 0; normaliser_net_y .= 0;
    synchronize()
    return nothing 
end

function _run_elastic_neighbourhood(elastic_net)
    """Perform elastic neighbourhood optimisation on a net initialised with an elastic net constructor. 
    Returns either the final net or the full evolution of the net through the optimisation. These are given in the form [Px, Py] and [Px, Py, Px_series, Py_series].
    The returned nets can then be used in various path operations to find optimal tours or passsed on for plotting.
    """
    #set the GPU device on the basis of available GPU's if GPU's available
    n_devices = length(collect(CUDA.devices()))
    if n_devices >= 1
        device!(mod(elastic_net.init_seed - 1, n_devices))
        CUDA.context()
        GPU_Cx, GPU_Cy, GPU_Px, GPU_Py = CuArray.([elastic_net.Cx, elastic_net.Cy, elastic_net.Px, elastic_net.Py])
        GPU_Arrays = CuArray.(elastic_net.Arrays)
        CUDA.allowscalar(false)
    else
        println("A GPU is currently needed; CPU functionality needs to be added.")
    end
    if elastic_net.record
        px_record = []
        py_record = []
    end

    n = elastic_net.Lp / elastic_net.Lc
    beta = 0.712
    dx = beta/n/sqrt(elastic_net.Lc)
    modifyc = length(elastic_net.temperature_distribution) > 5 # elastic_net.c > 0.9 # hack
    @inbounds for t_long = 1:elastic_net.max_iterations
        
        # set the dynamic parameters
        K = elastic_net.temperature_distribution[t_long]
        if modifyc
            elastic_net.c = dx * K * exp(-dx^2 / (2 * K ^ 2))
        end
        
        # perform the optimisations
        for t_short = 1:elastic_net.stability
            hot_neighbourhood_update!(
            Int(t_long), Int(t_short), elastic_net.temperature_distribution[t_long], elastic_net.a, elastic_net.b, elastic_net.c, elastic_net.alpha, elastic_net.beta, elastic_net.eta, elastic_net.boundary, elastic_net.stability, getfield(Main, Symbol(elastic_net.optimiser)), 
            GPU_Cx, GPU_Cy, GPU_Px, GPU_Py, GPU_Arrays..., 
            elastic_net.epsilon, elastic_net.Lp, elastic_net.Lc, elastic_net.Bp, elastic_net.Bc, elastic_net.threads)

            # record if necessary
            if elastic_net.record
                push!(px_record, [Array(GPU_Px)])
                push!(py_record, [Array(GPU_Py)])
            end
        end

        if elastic_net.time_status
            if t_long == 2
                t2_time = time()
            end
            
            account = 3
            if t_long == account
                t3_time = time()
                t_delta = (t3_time - t2_time) / (account - 2)
                sec_est = round(t_delta * (elastic_net.max_iterations - account) / 1, digits=3)
                mins_est = round(t_delta * (elastic_net.max_iterations - account) / 60, digits=3)
                hours_est = round(t_delta * (elastic_net.max_iterations - account) / 3600, digits=3)
                println("At $(Time(now())) there are an estimated $(hours_est) hours OR $(mins_est) minutes OR $(sec_est) seconds remaining.")
                println("Estimated finish time: $(Time(now() + Second(round(Int, sec_est))))")
            end

            if t_long == elastic_net.max_iterations
                println("Finish time: $(Time(now())).")
            end
        end
    end

    # Unload data and derefence the elastic net object and memory allocations
    px, py = [Array(GPU_Px), Array(GPU_Py)]
    CUDA.unsafe_free!.([GPU_Cx, GPU_Cy, GPU_Px, GPU_Py, GPU_Arrays...])

    if elastic_net.record
        elastic_net = nothing
        return [px, py, px_record, py_record]
    else elastic_net.record == false
        elastic_net = nothing
        return [px, py]
    end
end