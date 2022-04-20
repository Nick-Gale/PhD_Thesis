using Plots
using CUDA
using ProgressMeter
using Random 

function adam!(alpha, beta, eta, t, weights, update, m, v, M, grad, epsilon)
    #a fast inplace memory computation of Adam
    m .= alpha .* m .+ (1 - alpha) .* grad
    v .= beta .* v .+ (1 - beta) .* (grad .^ 2)
    update .= eta .* m ./ (1 - alpha^t) ./ (sqrt.(v ./ (1 - beta^t)) .+ epsilon)
    return nothing
end

function gd!(alpha, beta, eta, t, weights, update, m, v, M, grad, epsilon)
    update .= eta .* grad
    return nothing
end

function lap!((L, P1, Lp, Lc, T))
    #GPU compute the normaliser for the nodes(city)/net weights
    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    cortxi = mod(indx - 1, sqrt(Lp)) + 1
    cortxj = ceil(indx/sqrt(Lp))

    cortyi = mod(indy - 1, sqrt(Lp)) + 1
    cortyj = ceil(indy/sqrt(Lp))

    testneighbour = (abs(cortxi - cortyi) <= 1.2) && (abs(cortxj - cortyj) <= 1.2)

    shared = @cuDynamicSharedMem(eltype(L), (T, T))
    test = (indy <= Lp) && (indx <= Lc) && testneighbour
    if test
       @inbounds shared[threadIdx().x, threadIdx().y] = -P1[indx] + P1[indy]
    end
    
    if test
        @inbounds @atomic L[indx] += shared[threadIdx().x, threadIdx().y] # exp(((Cprimary[indx] - Pprimary[indy])^2 + (Cauxillary[indx] - Pauxillary[indy])^2) / (-2 * K^2)) #
    end
    return nothing
end

function normaliser!((K, normaliser, C1, C2, C3, P1, P2, P3, Lp, Lc, B, T))
    #GPU compute the normaliser for the nodes(city)/net weights
    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    shared = @cuDynamicSharedMem(eltype(C1), (T, T))
    test = (indy <= Lp) && (indx <= Lc)
    if test
       @inbounds shared[threadIdx().x, threadIdx().y] = exp( ( (C1[indx] - P1[indy])^2 + (C2[indx] - P2[indy])^2 + (C3[indx] - P3[indy])^2) / (-2 * K^2)) # exp(((mod(C1[indx] - P1[indy], 1))^2 + (mod(C2[indx] - P2[indy], 1) )^2 + (mod(C3[indx] - P3[indy], 1))^2) / (-2 * K^2)) # exp(((mod(C1[indx] - P1[indy] + 0.5, 1) - 0.5)^2 + (mod(C2[indx] - P2[indy] + 0.5, 1) - 0.5)^2 + (mod(C3[indx] - P3[indy] + 0.5, 1) - 0.5)^2) / (-2 * K^2))
    end
    
    if test
        @inbounds @atomic normaliser[indx] += shared[threadIdx().x, threadIdx().y] # exp(((Cprimary[indx] - Pprimary[indy])^2 + (Cauxillary[indx] - Pauxillary[indy])^2) / (-2 * K^2)) #
    end
    return nothing
end

function dw!((K, dP, normaliser, C1, C2, C3, P1, P2, P3, epsilon, Lp, Lc, B, T))
    #GPU compute of gradient contribution from interatctions with nodes or net along primary axis (auxillary indicates the axis locations needed for calculation, but not output)
    indx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    indy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    shared = @cuDynamicSharedMem(eltype(C1), (T, T))

    test = (indy <= Lc) && (indx <= Lp)
    if test
        @inbounds shared[threadIdx().x, threadIdx().y] = exp(( (C1[indy]-P1[indx])^2 + (C2[indy]-P2[indx])^2 + (C3[indy]-P3[indx])^2  ) / (-2 * K^2)) / (normaliser[indy] + epsilon) * (C1[indy] - P1[indx]) # exp(((mod(C1[indy] - P1[indx], 1))^2 + (C2[indy] - P2[indx])^2 + (C3[indy] - P3[indx])^2) / (-2 * K^2)) / (normaliser[indy] + epsilon) * (C1[indy] - P1[indx])  # exp(((mod(C1[indy] - P1[indx] + 0.5, 1)-0.5)^2 + (mod(C2[indy] - P2[indx] + 0.5, 1)-0.5)^2 + (mod(C3[indy] - P3[indx] + 0.5, 1)-0.5)^2) / (-2 * K^2)) / (normaliser[indy] + epsilon) * (C1[indy] - P1[indx])
    end

    if test
        @inbounds @atomic dP[indx] += shared[threadIdx().x, threadIdx().y] # exp(((Cprimary[indy] - Pprimary[indx])^2 + (Cauxillary[indy] - Pauxillary[indx])^2) / (-2 * K^2)) / (normaliser[indy] + epsilon) * (Cprimary[indy] - Pprimary[indx]) #
    end
    return nothing
end

function grad!((K, a, b, c, grad, C1, C2, C3, P1, P2, P3, dPnodes, dPnet, dPlaplacian, normaliser_nodes, normaliser_net, epsilon, Lp, Lc, Bp, Bc, T))
    #compute the gradient components along the primary direction
    s1 = CUDA.CuStream()
    s2 = CUDA.CuStream()
    s3 = CUDA.CuStream()
    smem_size = (sizeof(eltype(P1)) * (T + 1) * (T + 1))

    @cuda blocks=(Bp, Bp) threads=(T, T) stream=s3 shmem = smem_size lap!((dPlaplacian, P1, Lp, Lp, T))
    @cuda blocks=(Bc, Bp) threads=(T, T) stream=s1 shmem = smem_size normaliser!((K, normaliser_nodes, C1, C2, C3, P1, P2, P3, Lp, Lc, Bp, T))
    @cuda blocks=(Bp, Bp) threads=(T, T) stream=s2 shmem = smem_size normaliser!((K, normaliser_net, P1, P2, P3, P1, P2, P3, Lp, Lp, Bp, T))
    synchronize(s1); synchronize(s2); synchronize(s3);
    @cuda blocks=(Bp, Bc) threads=(T, T) stream=s2 shmem = smem_size dw!((K, dPnodes, normaliser_nodes, C1, C2, C3, P1, P2, P3, epsilon, Lp, Lc, Bc, T))
    @cuda blocks=(Bp, Bp) threads=(T, T) stream=s1 shmem = smem_size dw!((K, dPnet, normaliser_net, P1, P2, P3, P1, P2, P3, epsilon, Lp, Lp, Bp, T))
    synchronize(s1); synchronize(s2); synchronize(s3);
    #compute the gradient with scaling
    grad .= c .* dPnet .- a .* dPnodes .- (b .* K) .* dPlaplacian #  grad .+ (c .* dPnet .- a .* dPnodes .- (b .* K) .* dPlaplacian)
    dPnodes .= 0; dPnet .= 0; normaliser_nodes .= 0; normaliser_net .= 0; dPlaplacian .= 0;
    return nothing
end

function neighbourhood_update!(t_long, t_short, K, a, b, c, alpha, beta, eta, boundary, stability, optimiser, 
    Cx, Cy, Co, Px, Py, Po, 
    normaliser_nodes_x, normaliser_nodes_y, normaliser_nodes_o, normaliser_net_x, normaliser_net_y, normaliser_net_o, 
    dPnodes_x, dPnodes_y, dPnodes_o, dPnet_x, dPnet_y, dPnet_o, dPlaplacian_x, dPlaplacian_y, dPlaplacian_o, 
    update_x, m_x, v_x, M_x, grad_x, 
    update_y, m_y, v_y, M_y, grad_y,
    update_o, m_o, v_o, M_o, grad_o,
    epsilon, Lp, Lc, Bp, Bc, T)
    #update x
    grad!((K, a, b, c, grad_x, Cx, Cy, Co, Px, Py, Po, dPnodes_x, dPnet_x, dPlaplacian_x, normaliser_nodes_x, normaliser_net_x, epsilon, Lp, Lc, Bp, Bc, T))
    optimiser(alpha, beta, eta, (t_long-1) * stability + t_short, Px, update_x, m_x, v_x, M_x, grad_x, epsilon)
    Px .=  Px .- update_x #  (abs.(Px .- update_x) .< boundary) .* (Px .- update_x) .+ (abs.(Px .- update_x) .> boundary) .* boundary .* sign.(Px .- update_x) #  
    #update y
    grad!((K, a, b, c, grad_y, Cy, Cx, Co, Py, Px, Po, dPnodes_y, dPnet_y, dPlaplacian_y, normaliser_nodes_y, normaliser_net_y, epsilon, Lp, Lc, Bp, Bc, T))
    optimiser(alpha, beta, eta, (t_long-1) * stability + t_short, Py, update_y, m_y, v_y, M_y, grad_y, epsilon)
    Py .=  Py .- update_y # (abs.(Py .- update_y) .< boundary) .* (Py .- update_y) .+ (abs.(Py .- update_y) .> boundary) .* boundary .* sign.(Py .- update_y) # 

    #update o
    grad!((K, a, b, c, grad_o, Co, Cy, Cx, Po, Py, Px, dPnodes_o, dPnet_o, dPlaplacian_o, normaliser_nodes_o, normaliser_net_o, epsilon, Lp, Lc, Bp, Bc, T))
    optimiser(alpha, beta, eta, (t_long-1) * stability + t_short, Po, update_o, m_o, v_o, M_o, grad_o, epsilon)
    Po .= Po .- update_o #  (abs.(Po .- update_o) .< boundary) .* (Po .- update_o) .+ (abs.(Po .- update_o) .> boundary) .* boundary .* sign.(Po .- update_o) #  
    return nothing 
end

function net(N, bs; batches=1, time=100, K=0.05, a=1, b=1, c=0.5, r=0.1)
    Nb = round(Int, N / bs)
    s = sqrt(N)
    px = rand(N) #.- 0.5
    py = rand(N) #.- 0.5
    po = r .* CUDA.rand(N)# cos.(CUDA.rand(N) .* pi/2) #.- 0.5
    for i = 1:N
        x = mod(i-1,s)/s + 1f-6
        y = ceil(i/s)/s + 1f-6
        px[i] = x #- 0.5
        py[i] = y #- 0.5
    end
    Px = CuArray(px)
    Py = CuArray(py)
    Po = CuArray(po)
    # pre-allocate
    Cx = CUDA.rand(Nb) #  .- 0.5
    Cy = CUDA.rand(Nb) #.- 0.5
    Co = r .* CUDA.rand(Nb) # cos.(CUDA.rand(Nb) .* pi/2)# .- 0.5

    normaliser_nodes_x = CUDA.zeros(Nb) 
    normaliser_nodes_y = CUDA.zeros(Nb) 
    normaliser_nodes_o = CUDA.zeros(Nb) 

    normaliser_net_x = CUDA.zeros(N) 
    normaliser_net_y = CUDA.zeros(N) 
    normaliser_net_o = CUDA.zeros(N) 

    dPnodes_x = CUDA.zeros(N) 
    dPnodes_y = CUDA.zeros(N) 
    dPnodes_o = CUDA.zeros(N) 

    dPnet_x = CUDA.zeros(N) 
    dPnet_y = CUDA.zeros(N) 
    dPnet_o = CUDA.zeros(N) 

    dPlaplacian_x = CUDA.zeros(N) 
    dPlaplacian_y = CUDA.zeros(N) 
    dPlaplacian_o = CUDA.zeros(N) 
    update_x = CUDA.zeros(N) 

    m_x = CUDA.zeros(N) 
    v_x = CUDA.zeros(N) 
    M_x = CUDA.zeros(N) 
    grad_x = CUDA.zeros(N) 

    update_y = CUDA.zeros(N) 
    m_y = CUDA.zeros(N) 
    v_y = CUDA.zeros(N) 
    M_y = CUDA.zeros(N) 
    grad_y = CUDA.zeros(N) 

    update_o = CUDA.zeros(N) 
    m_o = CUDA.zeros(N) 
    v_o = CUDA.zeros(N) 
    M_o = CUDA.zeros(N) 
    grad_o = CUDA.zeros(N) 

    #launch parameters
    Lp = N
    Lc = Nb
    T = 16 
    Bp = round(Int, N/T) 
    Bc = round(Int, Nb/T) 

    # parameters
    alpha = 0.9
    beta = 0.999
    eta = 0.01

    K = K
    a = a
    b = b# /N^2
    c = c

    S = time
    L = batches

    boundary = 2.5
    epsilon = 1f-6
    Cx = CUDA.rand(Nb) #.- 0.5
    Cy = CUDA.rand(Nb) #.- 0.5
    Co = r .* CUDA.rand(Nb) # cos.(CUDA.rand(Nb) .* pi/2) #.- 0.5

    for tL = 1:L
        Cx = CUDA.rand(Nb) #.- 0.5
        Cy = CUDA.rand(Nb) #.- 0.5
        Co = r .* CUDA.rand(Nb) # cos.(CUDA.rand(Nb) .* pi/2) #.- 0.5
    
        for tS = 1:S
            neighbourhood_update!(tL, tS, K, a, b, c, alpha, beta, eta, boundary, S, getfield(Main, Symbol("adam!")), 
                Cx, Cy, Co, Px, Py, Po, 
                normaliser_nodes_x, normaliser_nodes_y, normaliser_nodes_o, normaliser_net_x, normaliser_net_y, normaliser_net_o, 
                dPnodes_x, dPnodes_y, dPnodes_o, dPnet_x, dPnet_y, dPnet_o, dPlaplacian_x, dPlaplacian_y, dPlaplacian_o, 
                update_x, m_x, v_x, M_x, grad_x, 
                update_y, m_y, v_y, M_y, grad_y,
                update_o, m_o, v_o, M_o, grad_o,
                epsilon, Lp, Lc, Bp, Bc, T)
        end
    end

    return Array(Px), Array(Py), Array(Po), px, py
end

function chi2(px, py, N)
    # partition the grid into NxN smaller grids. Each of these are expected to have 1/N^2 * length(Px) points under a uniform distribution.
    # make N2 observations of these grids and calculate chi2 statistics.

    obs = zeros(N^2)
    expectation = length(px)/N^2

    for i = 1:N
        for j = 1:N
            nij = filter(x -> ((i-1)/N < px[x] < i/N) && ((j-1)/N < py[x] < j/N), 1:length(px))
            obs[(i-1)*N + j] = length(nij)
        end
    end

    chi2 = sum((obs .- expectation).^2 ./ expectation )
    return chi2
end

function plot_c(px, py, s, str)
    c = plot()
    Px = reshape(px, s, s)
    Py = reshape(py, s, s)
    for i = 1:s
        for j = 1:s
            i1 = i > 1 ? i - 1 : i#mod(i - 2, s) + 1
            i2 = i < s ? i + 1 : i#mod(i, s) + 1
            j1 = j > 1 ? j - 1 : j#mod(j - 2, s) + 1
            j2 = j < s ? j + 1 : j#mod(j, s) + 1
            p0 = [Px[i,j], Py[i,j]]
            px1 = [Px[i1,j], Px[i,j]]
            px2 = [Px[i2,j], Px[i,j]]
            px3 = [Px[i,j1], Px[i,j]]
            px4 = [Px[i,j2], Px[i,j]]
            py1 = [Py[i1,j], Py[i,j]]
            py2 = [Py[i2,j], Py[i,j]]
            py3 = [Py[i,j1], Py[i,j]]
            py4 = [Py[i,j2], Py[i,j]]
            plot!(c, px1, py1, c=:blue, legends=:false, title=str * ": Retinotopy", size=(500,500), dpi=500, aspect_ratio = 1, xlim=(0,1), ylim=(0,1))
            plot!(c, px2, py2, c=:blue)
            plot!(c, px3, py3, c=:blue)
            plot!(c, px4, py4, c=:blue)
        end
    end
    return c
end

function plot_chi2(px, py, px_net, py_net)
    maxN = floor(Int, sqrt(length(px)))

    px_uni = rand(length(px))
    py_uni = rand(length(px))

    cNet = zeros(maxN)
    cNeigh = zeros(maxN)
    cUniform = zeros(maxN)
    for i = 1:maxN
        cNet[i] = chi2(px_net, py_net, i)
        cNeigh[i] = chi2(px, py, i)
        cUniform[i] = chi2(px_uni, py_uni, i)
    end

    plt = plot()
    plot!(plt, 1:maxN, hcat(cNeigh, cNet, cUniform), ylabel="χ²", xlabel="Grid Size", title="Retinotopic Uniformity", label=["Elastic Neighbourhood" "Elastic Net" "Random Uniform Sample"], DPI = 500)
end

CUDA.seed!(1)
# parameters
N = 1600
batches = 5
time = 100
K = 0.05
a = 1 # 0.5
b = 1 #0.0001
r = 0.05 # 0.05
s = round(Int, sqrt(N))

## Neighbourhood
str = "Elastic Neighbourhood"
c = 0.1
px, py, po, pxi, pyi = net(N, 8; batches=batches, time=time, K=K, a=a, b=b, c=c, r=r)

A = plot(px, py, st=:scatter)
B = plot(unique(pxi), unique(pyi), reshape((po ./ r), s, s), xlim=(0,1), ylim=(0,1), st=:heatmap, c=:rainbow, fill=true, title= str * ": Feature Selectivity", dpi=500,  size=(500,500), aspect_ratio = 1)
C = plot_c(px, py, s, str)
savefig(B, "./figures/fig_elastic_neighbourhood_feature.png")
savefig(C, "./figures/fig_elastic_neighbourhood_retinotopy.png")

## Net 
str = "Elastic Net"
c = 0.0
px_net, py_net, po, pxi, pyi = net(N, 8; batches=batches, time=time, K=K, a=a, b=b, c=c, r=r)

A = plot(px_net, py_net, st=:scatter)
B = plot(unique(pxi), unique(pyi), reshape((po ./ r), s, s), xlim=(0,1), ylim=(0,1), st=:heatmap, c=:rainbow, fill=true, title= str * ": Feature Selectivity", dpi=500, size=(500,500), aspect_ratio = 1)
C = plot_c(px_net, py_net, s, str)
savefig(B, "./figures/fig_elastic_net_feature.png")
savefig(C, "./figures/fig_elastic_net_retinotopy.png")

D = plot_chi2(Array(px), Array(py), Array(px_net), Array(py_net))
savefig(D, "./figures/fig_chi2_statistics_uniform.png")
