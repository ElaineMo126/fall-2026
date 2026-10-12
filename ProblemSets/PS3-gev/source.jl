
#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation
    return df, X, Z, y
end

#---------------------------------------------------
# Question 1: Multinomial Logit with Alternative-Specific Covariates
#---------------------------------------------------

function mlogit_with_Z(θ, X, Z, y)
    # Extract parameters
    # theta = [alpha1, alpha2, ..., alpha21, gamma]
    # alpha has K*(J-1) = 3*7 = 21 elements
    # gamma is the coefficient on Z
    α = θ[1:end-1]  # first 21 elements
    γ = theta[end]      # last element
    
    K = size(X, 2)  # number of covariates in X
    J = length(unique(y))  # number of choices (8)
    N = length(y)   # number of observations
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha into K x (J-1) matrix, add zeros for normalized choice
    bigα = [reshape(α, K, J-1) zeros(K)]
    
    # TODO: Compute choice probabilities
    # Hint: Pi,j = exp(Xi*beta_j + gamma*(Zi,j - Zi,J)) / denominator
    # where denominator sums over all choices
    
    # Initialize probability matrix
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T, N, J)
    dem = zeros(T, N)
    
    # Fill in: compute numerator for each choice j
    for j = 1:J
        num[:,j] .= exp.(X*bigα[:,j] .+ γ*(Z[:,j]) .- Z[:,j])
    end
    
    # Fill in: compute denominator (sum of numerators)
    dem = sum(num, dims=2)
    
    # Fill in: compute probabilities
    P = num ./ dem # ./ means it auto expands the dem to match num dimensions
    
    # Fill in: compute negative log-likelihood
    # loglike = -sum(bigY .* log.(P))
    
    return loglike
end


# Estimate multinomial logit
println("\n=== MULTINOMIAL LOGIT RESULTS ===")
# theta_hat_mle = optimize_mlogit(X, Z, y)
# println("Estimates: ", theta_hat_mle)

# Estimate nested logit
println("\n=== NESTED LOGIT RESULTS ===")
nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]  # WC and BC occupations
# nlogit_theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
# println("Estimates: ", nlogit_theta_hat)