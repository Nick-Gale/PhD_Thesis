# A simple implementation of an L-System with rules from paper describing the MNPeano curve: "The Euclidean Traveling Salesman Problem and a Space-Filling Curve". DOI https://doi.org/10.1016/0960-0779(95)80046-J.
# Some code has been frankensteined from Luxor.jl for learning purposes. 


using Interpolations
mutable struct LSystem
    rules::Dict{String, String}
    state::Array{Int64, 1}
    initial_state::Array{Int64, 1}
    function LSystem(rules, state_as_string)
        newlsystem = new(rules, string_to_array(state_as_string), string_to_array(state_as_string))
        return newlsystem
    end
end

function string_to_array(str::String)
    return map(x -> Int(Char(x)), collect(str))
end

function array_to_string(arr::Array)
    return join(string.(Char.(collect(arr))))
end

function evaluate(ls::LSystem, iterations=1)
    next_state = Array{Int64, 1}()
    for i in 1:iterations
        @debug println("iteration $i")
        for j in 1:length(ls.state) # each character in state
            s = string(Char(ls.state[j]))
            if haskey(ls.rules, s)
                #  replace it using the rule
                value = ls.rules[s]
                varr = string_to_array(value)
                if ! isempty(value)
                    push!(next_state, varr...)
                end
            else # keep it in
                push!(next_state, ls.state[j])
            end
        end
        @debug array_to_string(ls.state)
        ls.state = next_state
        next_state = Array{Int64, 1}()
    end
end

function move_turtle(turtle, distance)
    turtle["x"] = turtle["x"] + distance * cos(turtle["angle"])
    turtle["y"] = turtle["y"] - distance * sin(turtle["angle"])
    return turtle 
end


function place_points(ls, its, turndegrees, distance, initx, inity)
    evaluate(ls, its)
    states = Char.(ls.state)
    repeat = 1
    delta = pi/180 * turndegrees

    turtle = Dict("x" => initx, "y" => inity, "angle" => delta)
    x_list = []
    y_list = []
    for val in states
        if val == 'F'
            turtle = move_turtle(turtle, distance)
            append!(x_list, turtle["x"])
            append!(y_list, turtle["y"])
        elseif val == '+'
            turtle["angle"] += delta * repeat
            repeat = 1
        elseif val == '-'
            turtle["angle"] -= delta * repeat
            repeat = 1
        elseif val == 'I'
            distance *= 1/sqrt(2)
        elseif val == 'Q'
            distance *= sqrt(2)
        elseif val == 'H'
            distance *= 2
        elseif val == 'J'
            distance *= 0.5
        elseif val == '!'
            delta = -delta
        elseif (val == '2')
            repeat = 2
        else
            nothing
        end
    end
    return x_list, y_list
end


function mpeano(its)
    mpeano = LSystem(
        Dict("F" => "", 
                "Y" => "FFY",
                "X" => "+!X!FF-BQFI-!X!FF+",
                "A" => "BQFI",
                "B" => "AFF"
            ),
        "XFF2-AFF2-XFF2-AFF"
    )
    x, y = place_points(mpeano, its, 45, 1, 0,0)
    return x, y
end

function mnpeano(its)
    mpeano = LSystem(
        Dict("F" => "", 
                "Y" => "FFY",
                "X" => "+!X!HFJ-BQFI-!X!FF+",
                "A" => "BQFI",
                "B" => "AFF"
            ),
        "XFF2-AFF2-XFF2-AFF"
    )
    x, y = place_points(mpeano, its, 45, 1, 0, 0)
    return x, y
end

function npoints(n)
    if n == 0
        return 12
    elseif mod(n, 2) == 1
        return npoints(n - 1) * 2 + 4
    else
        return npoints(n - 1) * 2 - 4
    end
end

function pred(lp)
    n = floor(log2(lp) - 4)
    c = ceil(log2(lp) - 4)
    vec = [npoints(n), npoints(c)]
    if npoints(c + 1) <= lp
        return c+1
    elseif npoints(c) <= lp
        return c
    elseif npoints(n) <= lp
        return n
    else
        println("Something has gone horribly wrong")
    end
end
