using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

# include ("PS3-gev_source.jl")

 # call main function
allwrap()

# interpretation of ̂γ:
# change in utility if I increase expected log wage by 1 unit
# (on average)
# ̂γ = -0.09
# says people don't want higher-paying occupation
# (probably, our model is bad ["misspecified"])
# (e.g. work hours, or working consitions, etc.)




# y = x*β + ϵ
# interpretation of β
# change in y if change x by 1 unit
# (on average)



function nested_logit_with_Z(theta, X, Z, y, nesting_structure)
    # Extract parameters
    # theta = [beta_WC (3 elements), beta_BC (3 elements), lambda_WC, lambda_BC, gamma]
    alpha = theta[1:end-3]      # first 6 elements (3 for WC, 3 for BC)
    lambda = theta[end-2:end-1] # lambda_WC, lambda_BC
    gamma = theta[end]          # coefficient on Z
    
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Create coefficient matrix for nested structure
    # First K columns for WC nest, next K for BC nest, zeros for Other
    bigAlpha = [repeat(alpha[1:K], 1, length(nesting_structure[1])) 
                repeat(alpha[K+1:2K], 1, length(nesting_structure[2])) 
                zeros(K)]
    
    # TODO: Implement nested logit probability calculation
    # This is complex - refer to the formula in the problem set
    
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T, N, J)
    lidx = zeros(T, N, J)  # linear index for each choice
    dem = zeros(T, N)
    
    # Fill in: compute linear indices for each choice
    for j = 1:J
        if j in nesting_structure[1]  # White collar
            num[:,j] = lidx[:,j] .*(sum(lidx[:,nesting_structyre[1]], dims=2)).^(lambda[1]-1)
        elseif j in nesting_structure[2]  # Blue collar
            num[:,j] = lidx[:,j] .*(sum(lidx[:,nesting_structyre[1]], dims=2)).^(lambda[2]-1)
        else  # Other
            lidx[:,j] = exp.(zeros(N)) # ones(N) would also work
        end
    end
    
    # Fill in: compute numerators using nested logit formula
    for j = 1:J
        if j in nesting_structure[1]
            # num[:,j] = lidx[:,j] .* (sum of lidx for WC nest)^(lambda[1]-1)
        elseif j in nesting_structure[2]
            # num[:,j] = lidx[:,j] .* (sum of lidx for BC nest)^(lambda[2]-1)
        else
            # num[:,j] = lidx[:,j]
        end
        # dem .+= num[:,j]
    end
    
    # Fill in: compute probabilities and log-likelihood
    # P = num ./ repeat(dem, 1, J)
    # loglike = -sum(bigY .* log.(P))
    
    return loglike
end