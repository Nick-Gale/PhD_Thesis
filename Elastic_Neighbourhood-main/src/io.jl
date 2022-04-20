#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IO Functions. #these can probably be improved.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function data_load(file_names, opt_tours, dir)
    tour_data = []
    for i = 1:length(opt_tours)
        str_start = findfirst(".opt", opt_tours[i])[1]-1
        str = dir * opt_tours[i][1:str_start] * ".tsp"
        city_data = readdlm(str)
        if (findfirst(x->x=="NODE_COORD_SECTION", city_data)!==nothing) & (findfirst(x->x=="EDGE_WEIGHT_TYPE", city_data)!==nothing)
            if (any("EUC_2D" .== city_data[findfirst(x->x=="EDGE_WEIGHT_TYPE", city_data)[1], :]))
                city_first = findfirst(x->x=="NODE_COORD_SECTION", city_data)[1]+1
                if findfirst(x->x=="EOF", city_data)!==nothing
                    city_last = findfirst(x->x=="EOF", city_data)[1]-1
                else
                    city_last = size(city_data)[1]
                end
                cx_unscaled = city_data[city_first:city_last, 2]
                cy_unscaled = city_data[city_first:city_last, 3]

                c_max = maximum(vcat(cx_unscaled, cy_unscaled))
                cx = cx_unscaled ./ c_max .- 0.5
                cy = cy_unscaled ./ c_max .- 0.5

                if dir == "./dataArt/"
                    opt_tour = randperm(length(cx))
                    push!(tour_data, [cx, cy, opt_tour, cx_unscaled, cy_unscaled])
                else
                    opt_tour = readdlm(dir * opt_tours[i])
                    tour_first = findfirst(x->x=="TOUR_SECTION", opt_tour)[1]+1
                    tour_last = minimum(getindex.(filter(x -> x !== nothing, [findfirst(x->x=="EOF", opt_tour),findfirst(x->x==-1, opt_tour)]),1))-1
                    opt_tour = opt_tour[tour_first:tour_last,1]
                    push!(tour_data, [cx, cy, opt_tour, cx_unscaled, cy_unscaled])
                end
            end
        end
    end
    return tour_data
end

function scan_file(tours_index_vec, dir)
    data = readdlm(dir)
    return setdiff(tours_index_vec, data[2:end, 1])
end

function touch_results_file(dir)
    io = open(dir, "a")  do io
        writedlm(io, ["Index" "Length" "Optimal Length" "Time" "Tour"])
    end
end

function write_tour(tour_index, tour_length, opt_length, tour, time, dir)
    io = open(dir, "a")  do io
        writedlm(io, [tour_index, tour_length, opt_length, time, Int.(tour)]')
    end
end