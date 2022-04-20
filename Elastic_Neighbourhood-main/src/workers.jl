#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Worker Functions.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
euc_dist(a,b) = sqrt.(sum((a.-b).^2))
fake_euc_dist(a,b) = round(Int, sqrt.(sum((a.-b).^2)))
vecdot(a,b) = sum(vec(a) .* vec(b))

function batch_tours(param_dict, tours_index_vec, tour_data; results_dir="./results/", max_time=Inf, read_and_print=false)
    #check results file
    if read_and_print
        if isfile(results_dir)
            tours_index_vec = scan_file(tours_index_vec, results_dir)
        else
            touch_results_file(results_dir)
        end
    end
    
    tours = []
    times = []
    inds = []
    opt_tours = []
    tour_lengths = []
    optimal_tour_lengths = []
    t0 = time()
    for x in tours_index_vec
        time_begin = time()
        if time_begin - t0 > max_time
            # do as much of the work as possible
            break
        else
            append!(tours, [elastic_neighbourhood(tour_data[x][1], tour_data[x][2]; params=param_dict)])
            append!(opt_tours, [tour_data[x][3]])
            
            time_end = time()
            append!(times, time_end - time_begin)
            append!(tour_lengths, compute_tour_length(tour_data[x][1], tour_data[x][2], Int.(tours[end][1]), euc_dist))
            append!(optimal_tour_lengths, compute_tour_length(tour_data[x][1], tour_data[x][2], Int.(opt_tours[end]), euc_dist))

            if read_and_print
                write_tour(x, tour_lengths[end], optimal_tour_lengths[end], tours[end][1], times[end], results_dir)
            end
        end
    end
    return tours, opt_tours, tour_lengths, optimal_tour_lengths
end 