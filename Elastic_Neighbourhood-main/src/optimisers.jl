function adam!(alpha, beta, eta, t, weights, update, m, v, M, grad, epsilon)
    #a fast inplace memory computation of Adam
    m .= alpha .* m .+ (1 - alpha) .* grad
    v .= beta .* v .+ (1 - beta) .* (grad .^ 2)
    update .= eta .* m ./ (1 - alpha^t) ./ (sqrt.(v ./ (1 - beta^t)) .+ epsilon)
    return nothing
end

function amsgrad!(alpha, beta, eta, t, weights, update, m, v, M, grad, epsilon)
    #a fast inplace memory computation of Adam
    m .= alpha .* m .+ (1 - alpha) .* grad
    v .= beta .* v .+ (1 - beta) .* (grad .^ 2)
    M .= max.(v, M)
    update .= eta .* m ./ (sqrt.(M) .+ epsilon)
    return nothing
end

function adamw!(alpha, beta, eta, t, weights, update, m, v, M, grad, epsilon)
    #a fast inplace memory computation of Adam
    #update .= update .* (1 - eta * 0.001 * log(length(m)))
    m .= alpha .* m .+ (1 - alpha) .* grad
    v .= beta .* v .+ (1 - beta) .* (grad .^ 2)
    update .= eta .* m ./ (1 - alpha^t) ./ (sqrt.(v ./ (1 - beta^t)) .+ epsilon) .+ weights * eta * 0.0001
    return nothing
end

function nesterov!(alpha, beta, eta, t, weights, update, m, v, M, grad, epsilon)
    #a fast inplace memory computation of Adam
    m .= update .- eta .* grad
    
    update .= (1 - alpha_T) .* m + alpha_t * M
    M .= m
    return nothing
end

function gd!(alpha, beta, eta, t, weights, update, m, v, M, grad, epsilon)
    update .= grad
    return nothing
end

function momentum!(alpha, beta, eta, t, weights, update, m, v, M, grad, epsilon)
    m .= alpha .* m .+  (1 - alpha) .* grad
    update .= 1 .* m
    return nothing
end

function nadam!(alpha, beta, eta, t, weights, update, m, v, M, grad, epsilon)
    #a fast inplace memory computation of Nadam
    m .= alpha .* m .+ (1 - alpha) .* grad
    M .= (alpha .* m .+ (1 - alpha) .* grad) ./ (1 - alpha^t)
    v .= beta .* v .+ (1 - beta) .* grad .^ 2
    update .= eta .* M ./ (sqrt.(v ./ (1 - beta^t)) .+ epsilon)
    return nothing
end
