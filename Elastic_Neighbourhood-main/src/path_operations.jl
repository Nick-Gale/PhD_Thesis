#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initialisation Functions.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

function initialisation_grid(init_seed, n_inits, cx, cy)
    n_steps = round(Int, sqrt(n_inits))
    if n_inits == 1
        x = sum(cx)/length(cx)
        y = sum(cy)/length(cy)
    else
        x = -0.5 + (mod(init_seed - 1, n_steps)) / (n_steps-1)
        y = -0.5 + (ceil(init_seed / n_steps) - 1) / (n_steps-1)
    end
    return x, y
end

function initialisation_random(init_seed, cx, cy; model=Product([Uniform([-0.5, 0.5]), Uniform(-0.5, 0.5)]))
    Random.seed!(init_seed)
    test = true
    x, y = rand(model)
    while test 
        if abs(x) > 0.5 || abs(y) > 0.5
            test = true 
        else
            test = false
        end
        if init_seed == 1
            x = sum(cx)/length(cx)
            y = sum(cy)/length(cy)
        else
            x, y = rand(model)
        end
    end
    #println([x, y])
    return x, y
end

function path_init_mpeano(lp)
    n = pred(lp)
    nx, ny = mpeano(n)
    
    dn = 1/length(nx)
    dl = 1/lp

    itn = collect(0:dn:(1-dn))
    push!(itn, 1)
    itl = 0:dl:(1-dl)

    data = [[nx[i], ny[i]] for i in 1:(length(itn)-1)]
    push!(data, [nx[1], ny[1]])
    interp = LinearInterpolation(itn, data)

    px = []
    py = []
    for t in itl
        x, y = interp(t)
        push!(px, x)
        push!(py, y)
    end

    px .= px .- minimum(px)
    py .= py .- minimum(py)
    sup = maximum([maximum(px), maximum(py)])
    px = (px ./ sup .- 0.5)
    py = (py ./ sup .- 0.5)

    return px, py 
end

function path_init_circle(density_neighbourhoods, Cx, Cy, radius, Px_center, Py_center)
    Px = zeros(round(Int, density_neighbourhoods * length(Cx)))
    Py = zeros(round(Int, density_neighbourhoods * length(Cy)))
    
    Px .= Px_center .+ radius * cos.(2 * pi * (1:length(Px))/length(Px))
    Py .= Py_center .+ radius * sin.(2 * pi * (1:length(Py))/length(Py))
    return Px, Py
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Refinement Function
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function counter_clockwise(Ax, Ay, Bx, By, Cx, Cy)
    return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
end

function intersection(Ux, Uy, Sx, Sy, Tx, Ty, Vx, Vy)
    test1 = counter_clockwise(Ux, Uy, Tx, Ty, Vx, Vy) != counter_clockwise(Sx, Sy, Tx, Ty, Vx, Vy)
    test2 = counter_clockwise(Ux, Uy, Sx, Sy, Vx, Vy) != counter_clockwise(Ux, Uy, Sx, Sy, Tx, Ty)
    return test1 && test2
end

function path_intersection(Cx, Cy, U, S, T, V)
    # see if line sections US and TV interesct

    # get coordinates
    Ux = Cx[U]
    Uy = Cy[U]

    Sx = Cx[S]
    Sy = Cy[S]

    Tx = Cx[T]
    Ty = Cy[T]

    Vx = Cx[V]
    Vy = Cy[V]

    return intersection(Ux, Uy, Sx, Sy, Tx, Ty, Vx, Vy)
end

function find_crossings(Cx, Cy, tour)
    L = length(tour)
    list_of_crossings = []
    for i = 1:(L-1)
        U = tour[i]
        S = tour[i+1]
        for j = (i+2):(L-1)
            T = tour[j]
            V = tour[j + 1]
            does_intersect = path_intersection(Cx, Cy, U, S, T, V)
            if does_intersect
                push!(list_of_crossings, [i, i+1, j, j+1])
            end
        end
    end
    return list_of_crossings
end 

function unwind_crossings(Cx, Cy, tour, crossings)
    u, s, t, v = crossings[1]
    return reverse(tour, s, t)
end

function two_opt(Cx, Cy, tour)
    err = compute_tour_length(Cx, Cy, tour, euc_dist)
    test = true
    need_to_loop = false
    L = length(Cx)
    while test
        for i = 1:(L-1)
            for j = (i+1):(L)
                cost = compute_tour_length(Cx, Cy, reverse(tour, i, j), euc_dist)
                cost_change = cost - err
                if cost_change < 1e-14
                    println(cost_change)
                    reverse!(tour, i, j)
                    need_to_loop = true
                    err = cost 
                end
            end
        end

        if need_to_loop == true
            test = true
        else
            test = false
        end
        need_to_loop = false
    end
    return tour
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Path Specific Functions.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function path_pad(Px, Py, depth)
    #pad out the current path by inserting $depth beads in a straight line in-between each of the Px/Py's
    L = length(Px)
    Px_new = zeros(L * depth)
    Py_new = zeros(L * depth)
    for i = 1:(L - 1)
        for j = 1:depth
            Px_new[(i - 1) * depth + j] = Px[i] + j/depth * (Px[i + 1] - Px[i])
            Py_new[(i - 1) * depth + j] = Py[i] + j/depth * (Py[i + 1] - Py[i])
        end
    end
    for j = 1:depth
        Px_new[(L-1) * depth + j] = Px[L] + j/depth * (Px[1] - Px[L])
        Py_new[(L-1) * depth + j] = Py[L] + j/depth * (Py[1] - Py[L])
    end
    return Px_new, Py_new
end

function path_inj(Cx, Cy, Px, Py, depth)
    #find the closest point on the (near) continuous path to each city. As the tour nears K=0 a continous path should give uniqueness but a large number of points is needed. To achieve this the given net is padded out with points inbetween each bead.
    X, Y = path_pad(Px, Py, depth)
    path = zeros(Int64, length(Cx))
    for i = 1:length(Cx)
        path[i] = argmin(sqrt.((X .- Cx[i]).^2 .+ (Y .- Cy[i]).^2))
        # remove this point from being accessible
        X[path[i]] = Inf
        Y[path[i]] = Inf
    end
    return path
end

function compute_tour(Cx, Cy, Px, Py)
    path_inds = path_inj(Cx, Cy, Px, Py, 2)
    tour = sortperm(path_inds)
    return tour
end

function compute_tour_length(Cx, Cy, tour, dist_fun)
    tour_length = 0
    L = length(tour)
    for j = 2:L
        tour_length += dist_fun([Cx[tour[j]],Cy[tour[j]]], [Cx[tour[j-1]], Cy[tour[j-1]]])
    end
    tour_length += dist_fun([Cx[tour[L]],Cy[tour[L]]], [Cx[tour[1]], Cy[tour[1]]])
    return tour_length
end

#Construct all tours in the batch and computer the best.
function compute_tour_batch(cx, cy, Pxs, Pys, tour_index_vec)
    tours = []
    lengths = []
    for i = 1:length(tour_index_vec)
        push!(tours, [compute_tour(cx, cy, Pxs[i], Pys[i]), Pxs[i], Pys[i]])
        push!(lengths, compute_tour_length(cx, cy, Int.(tours[i][1]), euc_dist))
    end
    ind = argmin(lengths)
    degenerate_lengths = unique(lengths)
    clusters = map(x -> findfirst(x .== degenerate_lengths), lengths)
    return [tours[ind]..., lengths]
end
