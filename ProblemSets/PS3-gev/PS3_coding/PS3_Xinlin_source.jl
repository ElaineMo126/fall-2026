#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data(url)
    # Added: also accept the supplied local CSV file.
    df = startswith(url, "http") ? CSV.read(HTTP.get(url).body, DataFrame) : CSV.read(url, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation
    return df, X, Z, y
end

#---------------------------------------------------
# Question 1: Multinomial Logit with Alternative-Specific Covariates
#---------------------------------------------------

function mlogit_with_Z(theta, X, Z, y)
    # Extract parameters
    # theta = [alpha1, alpha2, ..., alpha21, gamma]
    # alpha has K*(J-1) = 3*7 = 21 elements
    # gamma is the coefficient on Z
    alpha = theta[1:end-1]  # first 21 elements
    gamma = theta[end]      # last element
    
    K = size(X, 2)  # number of covariates in X
    J = size(Z, 2)  # Correction: count available, not only observed, choices.  # number of choices (8)
    N = length(y)   # number of observations
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha into K x (J-1) matrix, add zeros for normalized choice
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # TODO: Compute choice probabilities
    # Hint: Pi,j = exp(Xi*beta_j + gamma*(Zi,j - Zi,J)) / denominator
    # where denominator sums over all choices
    
    # Initialize probability matrix
    T = promote_type(eltype(X), eltype(Z), eltype(theta))
    num = zeros(T, N, J)
    dem = zeros(T, N)
    
    # Added: subtract a common row maximum to avoid exponential overflow.
    # A common factor cancels between num and dem; probabilities are unchanged.
    shift = vec(maximum(X*bigAlpha .+ gamma .* (Z .- Z[:, J:J]), dims=2))
    # Fill in: compute numerator for each choice j
    for j = 1:J
        # num[:,j] = exp.(...)
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma .- shift)
    end
    
    # Fill in: compute denominator (sum of numerators)
    # dem = sum(num, dims=2)
    dem = sum(num, dims=2)
    
    # Fill in: compute probabilities
    # P = num ./ dem
    P = num ./ dem
    
    # Fill in: compute negative log-likelihood
    # loglike = -sum(bigY .* log.(P))
    # Added: select observed choices to avoid 0*log(0) for unchosen choices.
    loglike = -sum(log.(P[bigY .== 1]))
    
    return isfinite(loglike) ? loglike : Inf
end

#---------------------------------------------------
# Question 2: Nested Logit
#---------------------------------------------------

function nested_logit_with_Z(theta, X, Z, y, nesting_structure)
    # Extract parameters
    # theta = [beta_WC (3 elements), beta_BC (3 elements), lambda_WC, lambda_BC, gamma]
    alpha = theta[1:end-3]      # first 6 elements (3 for WC, 3 for BC)
    lambda = theta[end-2:end-1] # lambda_WC, lambda_BC
    gamma = theta[end]          # coefficient on Z
    # Added: lambda must be positive; reject invalid optimizer trial points.
    if any(lambda .<= 0) || !all(isfinite, theta)
        return Inf
    end
    
    K = size(X, 2)
    J = size(Z, 2)  # Correction: count available, not only observed, choices.
    N = length(y)
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Create coefficient matrix for nested structure
    # First K columns for WC nest, next K for BC nest, zeros for Other
    # Correction: concatenate horizontally (3 WC columns, 4 BC columns, Other).
    bigAlpha = hcat(repeat(alpha[1:K], 1, length(nesting_structure[1])),
                    repeat(alpha[K+1:2K], 1, length(nesting_structure[2])),
                    zeros(K))
    
    # TODO: Implement nested logit probability calculation
    # This is complex - refer to the formula in the problem set
    
    T = promote_type(eltype(X), eltype(Z), eltype(theta))
    num = zeros(T, N, J)
    lidx = zeros(T, N, J)  # linear index for each choice
    dem = zeros(T, N)
    
    # Added: numerical scaling, needed when lambda is close to zero.
    # X*beta is common within a nest, so factor it out before exponentiating.
    # The original formulas below remain as comments for comparison.
    shift_WC = vec(maximum((Z[:,nesting_structure[1]] .- Z[:,J:J])*gamma/lambda[1], dims=2))
    shift_BC = vec(maximum((Z[:,nesting_structure[2]] .- Z[:,J:J])*gamma/lambda[2], dims=2))

    # Fill in: compute linear indices for each choice
    for j = 1:J
        if j in nesting_structure[1]  # White collar
            # lidx[:,j] = exp.((X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma) ./ lambda[1])
            lidx[:,j] = exp.((Z[:,j] .- Z[:,J])*gamma/lambda[1] .- shift_WC)
        elseif j in nesting_structure[2]  # Blue collar
            # lidx[:,j] = exp.((X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma) ./ lambda[2])
            lidx[:,j] = exp.((Z[:,j] .- Z[:,J])*gamma/lambda[2] .- shift_BC)
        else  # Other
            # lidx[:,j] = exp.(zeros(N))
            lidx[:,j] = exp.(zeros(N))
        end
    end
    
    sum_WC = vec(sum(lidx[:,nesting_structure[1]], dims=2))
    sum_BC = vec(sum(lidx[:,nesting_structure[2]], dims=2))
    # Restore the factored nest terms; subtract one common scale from all num.
    base_WC = X*bigAlpha[:,first(nesting_structure[1])] .+ lambda[1]*shift_WC
    base_BC = X*bigAlpha[:,first(nesting_structure[2])] .+ lambda[2]*shift_BC
    shift = max.(0, base_WC .+ lambda[1]*log.(sum_WC),
                    base_BC .+ lambda[2]*log.(sum_BC))

    # Fill in: compute numerators using nested logit formula
    for j = 1:J
        if j in nesting_structure[1]
            # num[:,j] = lidx[:,j] .* (sum of lidx for WC nest)^(lambda[1]-1)
            num[:,j] = lidx[:,j] .* exp.((lambda[1]-1)*log.(sum_WC) .+ base_WC .- shift)
        elseif j in nesting_structure[2]
            # num[:,j] = lidx[:,j] .* (sum of lidx for BC nest)^(lambda[2]-1)
            num[:,j] = lidx[:,j] .* exp.((lambda[2]-1)*log.(sum_BC) .+ base_BC .- shift)
        else
            # num[:,j] = lidx[:,j]
            num[:,j] = lidx[:,j] .* exp.(-shift)
        end
        # dem .+= num[:,j]
        dem .+= num[:,j]
    end
    
    # Fill in: compute probabilities and log-likelihood
    # P = num ./ repeat(dem, 1, J)
    P = num ./ repeat(dem, 1, J)
    # loglike = -sum(bigY .* log.(P))
    # Added: select observed choices to avoid 0*log(0) for unchosen choices.
    loglike = -sum(log.(P[bigY .== 1]))
    
    return isfinite(loglike) ? loglike : Inf
end

#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------

function optimize_mlogit(X, Z, y)
    # Starting values: 21 alphas + 1 gamma
    startvals = [2*rand(7*size(X,2)).-1; 0.1]
    
    # TODO: Use optimize() function
    # Hint: Use LBFGS() algorithm with g_tol = 1e-5
    
    result = optimize(theta -> mlogit_with_Z(theta, X, Z, y), 
                     startvals, LBFGS(), 
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    
    # Added: report convergence rather than silently accepting an unfinished fit.
    println("Optimizer converged: ", Optim.converged(result))
    println("Final negative log-likelihood: ", Optim.minimum(result))
    return result.minimizer
end

function optimize_nested_logit(X, Z, y, nesting_structure)
    # Starting values: 6 alphas + 2 lambdas + 1 gamma
    startvals = [2*rand(2*size(X,2)).-1; 1.0; 1.0; 0.1]
    
    # TODO: Use optimize() function for nested logit
    
    result = optimize(theta -> nested_logit_with_Z(theta, X, Z, y, nesting_structure), 
                     startvals, LBFGS(), 
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    
    # Added: report convergence rather than silently accepting an unfinished fit.
    println("Optimizer converged: ", Optim.converged(result))
    println("Final negative log-likelihood: ", Optim.minimum(result))
    return result.minimizer
end

