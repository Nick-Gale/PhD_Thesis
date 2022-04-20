using Revise
using CUDA
CUDA.seed!(1)
includet("../src/elastic_neighbourhood_logic.jl")
Lc = 10000
Lp = 2 * Lc
T = 16
Bp = ceil(Int, Lp / T)
Bc = ceil(Int, Lc / T)
K=0.01
a=0.007
b=1
c=1 
gradx = CUDA.zeros(Lp)
grady = CUDA.zeros(Lp)
Cx = CUDA.rand(Lc)
Cy = CUDA.rand(Lc)
Px = CUDA.rand(Lp)
Py = CUDA.rand(Lp)
dPnodesx = CUDA.zeros(Lp)
dPnetx = CUDA.zeros(Lp)
dPlaplacianx = CUDA.zeros(Lp)
dPnodesy = CUDA.zeros(Lp)
dPnety = CUDA.zeros(Lp)
dPlaplaciany = CUDA.zeros(Lp)
normaliser_nodes = CUDA.zeros(Lc)
normaliser_net = CUDA.zeros(Lp)
epsilon = 1f-7

dPnodesx = CUDA.zeros(Lp)
dPnetx = CUDA.zeros(Lp)
dPlaplacianx = CUDA.zeros(Lp)

hot_grad!((K, a, b, c, gradx, grady, Cx, Cy, Px, Py, 
    dPnodesx, dPnetx, dPlaplacianx, dPnodesy, dPnety, dPlaplaciany,
    normaliser_nodes, normaliser_net, epsilon, Lp, Lc, Bp, Bc, T))

normaliser_c = sum(exp.(((Cx' .- Px).^2 + (Cy' .- Py).^2) ./-2 ./K^2), dims=1)
normaliser_p = sum(exp.(((Px' .- Px).^2 + (Py' .- Py).^2) ./-2 ./K^2), dims=1)

for i = 1:Lp
    dPlaplacianx[i] = Px[Int(mod(i-2, Lp) + 1)] - 2 * Px[Int(mod(i-1, Lp) + 1)] + Px[Int(mod(i, Lp) + 1)]
end

dPnodesx = sum(exp.(((Cx' .- Px).^2 + (Cy' .- Py).^2) ./-2 ./ K^2) ./ (normaliser_c .+ epsilon) .* (Cx' .- Px), dims=2)
dPnetx = sum(exp.(((Px' .- Px).^2 + (Py' .- Py).^2) ./-2 ./ K^2) ./ (normaliser_p .+ epsilon) .* (Px' .- Px), dims=2)

grad_true_x = (c .* dPnetx .- a .* dPnodesx .- (b .* K) .* dPlaplacianx) 

println(all((gradx .- grad_true_x) .< 1f-7))
