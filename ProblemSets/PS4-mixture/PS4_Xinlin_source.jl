include(joinpath(@__DIR__, "lgwt.jl"))

#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    # Added: prefer the supplied local data; retain the course URL as a fallback.
    local_file = joinpath(@__DIR__, "nlsw88t.csv")
    df = isfile(local_file) ? CSV.read(local_file, DataFrame) : CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code
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
    
    K = size(X, 2)  # number of covariates in X (3)
    J = size(Z, 2)  # Correction: count available alternatives, even in small tests.  # number of choices (8)
    N = length(y)   # number of observations
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha into K x (J-1) matrix, add zeros for normalized choice J
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # TODO: Compute choice probabilities
    # Hint: P_ij = exp(X_i*beta_j + gamma*(Z_ij - Z_iJ)) / denominator
    # where denominator sums over all choices
    
    # Initialize probability matrix  
    T = promote_type(eltype(X), eltype(Z), eltype(theta))
    num = zeros(T, N, J)
    dem = zeros(T, N)
    
    # Added: a common row shift prevents overflow without changing probabilities.
    lidx = X*bigAlpha .+ gamma .* (Z .- Z[:,J:J])
    shift = maximum(lidx, dims=2)
    # Fill in: compute numerator for each choice j
    for j = 1:J
        # num[:,j] = exp.(X * bigAlpha[:,j] .+ gamma .* (Z[:,j] .- Z[:,J]))
        num[:,j] = exp.(lidx[:,j] .- vec(shift))
    end
    
    # Fill in: compute denominator (sum of numerators)
    # dem = sum(num, dims=2)
    dem = sum(num, dims=2)
    
    # Fill in: compute probabilities
    # P = num ./ dem
    P = num ./ dem
    
    # Fill in: compute negative log-likelihood
    # loglike = -sum(bigY .* log.(P))
    # Added: log(P) evaluated before exponentiation avoids log(0) underflow.
    logP = lidx .- shift .- log.(dem)
    loglike = -sum(bigY .* logP)
    
    return loglike
end

#---------------------------------------------------
# Question 3a: Quadrature Practice
#---------------------------------------------------

function practice_quadrature()
    println("=== Question 3a: Quadrature Practice ===")
    
    # Define standard normal distribution
    d = Normal(0, 1)
    
    # Get quadrature nodes and weights for 7 grid points
    nodes, weights = lgwt(7, -4, 4)
    
    # TODO: Verify integral of density equals 1
    # integral_density = sum(weights .* pdf.(d, nodes))
    integral_density = sum(weights .* pdf.(d, nodes))
    println("∫φ(x)dx = $integral_density (should be ≈ 1)")
    
    # TODO: Verify expectation equals 0  
    # expectation = sum(weights .* nodes .* pdf.(d, nodes))
    expectation = sum(weights .* nodes .* pdf.(d, nodes))
    println("∫xφ(x)dx = $expectation (should be ≈ 0)")
    println("Exact probability on [-4,4]: ", cdf(d,4)-cdf(d,-4))
    println("Seven nodes are an approximation; the density integral need not equal one exactly.")
    return (density=integral_density, expectation=expectation)
end

#---------------------------------------------------
# Question 3b: More Quadrature Practice
#---------------------------------------------------

function variance_quadrature()
    println("\n=== Question 3b: Variance using Quadrature ===")
    
    # Define N(0,2) distribution
    d = Normal(0, 2)
    σ = 2
    
    # TODO: Use quadrature to compute ∫x²f(x)dx with 7 points
    nodes7, weights7 = lgwt(7, -5*σ, 5*σ)
    # variance_7pts = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))
    variance_7pts = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))
    
    # TODO: Use quadrature to compute ∫x²f(x)dx with 10 points
    nodes10, weights10 = lgwt(10, -5*σ, 5*σ)  
    # variance_10pts = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
    variance_10pts = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
    
    println("Variance with 7 quadrature points: $variance_7pts")
    println("Variance with 10 quadrature points: $variance_10pts")
    println("True variance: $(σ^2)")
    
    # TODO: Comment on approximation quality
    # Added: Normal(0,2) uses standard deviation 2, so its variance is 4.
    # Both tail truncation and the number of quadrature nodes cause error.
    println("Absolute errors: 7 nodes = ", abs(variance_7pts-σ^2),
            "; 10 nodes = ", abs(variance_10pts-σ^2))
    println("Ten nodes improve this approximation; seven nodes are too coarse over [-10,10].")
    return (variance7=variance_7pts, variance10=variance_10pts)
end

#---------------------------------------------------
# Question 3c: Monte Carlo Practice  
#---------------------------------------------------

function practice_monte_carlo()
    println("\n=== Question 3c: Monte Carlo Integration ===")
    
    σ = 2
    d = Normal(0, σ)
    a, b = -5*σ, 5*σ
    
    # TODO: Implement Monte Carlo integration function
    function mc_integrate(f, a, b, D)
        # ∫f(x)dx ≈ (b-a) * (1/D) * Σf(X_i) where X_i ~ U[a,b]
        # draws = rand(D) * (b - a) .+ a  # uniform draws on [a,b]
        draws = rand(D) * (b - a) .+ a  # uniform draws on [a,b]
        # return (b - a) * mean(f.(draws))
        return (b - a) * mean(f.(draws))
    end
    
    # Added: reproducible exercise results, kept separate from optimization draws.
    Random.seed!(6343)
    results = Dict{Int, NamedTuple}()
    # Test with different numbers of draws
    for D in [1000, 1000000]
        println("\nWith D = $D draws:")
        
        # TODO: Variance: ∫x²f(x)dx  
        # variance_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
        variance_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
        println("MC Variance: $variance_mc (true: $(σ^2))")
        
        # TODO: Mean: ∫xf(x)dx
        # mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)
        mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)  
        println("MC Mean: $mean_mc (true: 0)")
        
        # TODO: Density integral: ∫f(x)dx
        # density_mc = mc_integrate(x -> pdf(d, x), a, b, D)
        density_mc = mc_integrate(x -> pdf(d, x), a, b, D)
        println("MC Density integral: $density_mc (true: 1)")
        results[D] = (variance=variance_mc, expectation=mean_mc, density=density_mc)
    end
    println("MC error is random and generally falls at rate 1/sqrt(D).")
    println("Increasing D from 1,000 to 1,000,000 reduces typical sampling error by sqrt(1000), about 31.6.")
    println("A particular draw need not improve every integral; truncation error also remains.")
    return (results=results, mc_integrate=mc_integrate)
end

#---------------------------------------------------
# Question 4: Mixed Logit with Quadrature (DO NOT RUN!)
#---------------------------------------------------

function mixed_logit_quad(theta, X, Z, y, nodes, weights)
    # Extract parameters
    # theta = [alpha1, ..., alpha21, mu_gamma, sigma_gamma]
    K = size(X, 2)
    J = size(Z, 2)  # Correction: count available alternatives, even in small tests.
    N = length(y)
    
    alpha = theta[1:(K*(J-1))]  # coefficients on X
    mu_gamma = theta[end-1]     # mean of gamma distribution
    sigma_gamma = theta[end]    # std dev of gamma distribution
    sigma_gamma >= 0 || return Inf
    # Follow the assignment: integrate each observation separately (not a
    # product across years inside one individual-level integral).
    # Nodes are standardized: gamma = mu + sigma*z, f_gamma(gamma)d_gamma = phi(z)dz.
    # Raw weights approximate normal mass; do not silently normalize the 7-node rule.
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    # Reshape alpha 
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # TODO: Implement mixed logit with quadrature
    # This involves integrating over the distribution of gamma
    
    # Initialize integrated probabilities
    T = promote_type(eltype(X), eltype(Z), eltype(theta))
    P_integrated = zeros(T, N, J)
    
    # TODO: For each quadrature point r:
    # 1. Transform node: gamma_r = mu_gamma + sigma_gamma * nodes[r]
    # 2. Compute choice probabilities for this gamma_r (like regular logit)
    # 3. Weight by quadrature weight and normal density
    # 4. Add to integrated probabilities
    
    for r in eachindex(nodes)
        # gamma_r = mu_gamma + sigma_gamma * nodes[r]
        gamma_r = mu_gamma + sigma_gamma * nodes[r]
        
        # Compute probabilities for this gamma_r
        # num_r = zeros(T, N, J)
        # for j = 1:J
        #     num_r[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_r .* (Z[:,j] .- Z[:,J]))
        # end
        # dem_r = sum(num_r, dims=2)
        # P_r = num_r ./ dem_r
        # Added: same common-shift stabilization as Question 1.
        num_r = zeros(T, N, J)
        shift_r = vec(maximum(X*bigAlpha .+ gamma_r .* (Z .- Z[:,J:J]), dims=2))
        for j = 1:J
            num_r[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_r .* (Z[:,j] .- Z[:,J]) .- shift_r)
        end
        dem_r = sum(num_r, dims=2)
        P_r = num_r ./ dem_r
        
        # Weight and add to integrated probabilities
        # density_weight = weights[r] * pdf(Normal(0,1), nodes[r])
        density_weight = weights[r] * pdf(Normal(0,1), nodes[r])
        # P_integrated .+= P_r * density_weight
        P_integrated .+= P_r * density_weight
    end
    
    # TODO: Compute log-likelihood  
    # loglike = -sum(bigY .* log.(P_integrated))
    # Added: selecting observed choices avoids 0*log(0) on unchosen alternatives.
    loglike = -sum(log.(P_integrated[bigY .== 1]))
    
    return loglike
end

#---------------------------------------------------
# Question 5: Mixed Logit with Monte Carlo (DO NOT RUN!)
#---------------------------------------------------

function mixed_logit_mc(theta, X, Z, y, D)
    # Extract parameters (same as quadrature version)
    K = size(X, 2)
    J = size(Z, 2)  # Correction: count available alternatives, even in small tests.
    N = length(y)
    
    alpha = theta[1:(K*(J-1))]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]
    sigma_gamma >= 0 || return Inf
    # Normal-distribution draws already incorporate the density. The weight
    # here is 1/D, unlike the uniform-draw (b-a)/D formula in Question 3c.
    
    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end
    
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
    
    # TODO: Implement mixed logit with Monte Carlo
    # Similar to quadrature but with random draws instead of nodes/weights
    
    T = promote_type(eltype(X), eltype(Z), eltype(theta))
    P_integrated = zeros(T, N, J)
    
    # TODO: For d = 1 to D:
    # 1. Draw gamma_d from N(mu_gamma, sigma_gamma^2)
    # 2. Compute choice probabilities for this draw
    # 3. Add to running average (P_integrated += P_d / D)
    
    # Correction: use fixed standard-normal draws and transform them.
    # Redrawing on each objective evaluation would invalidate optimization/AD.
    gamma_dist = Normal(0, 1)
    rng = MersenneTwister(6343)
    
    for d = 1:D
        # gamma_d = rand(gamma_dist)
        gamma_d = mu_gamma + sigma_gamma * rand(rng, gamma_dist)
        
        # Compute probabilities for this draw (same as regular logit)
        # num_d = zeros(T, N, J)  
        # for j = 1:J
        #     num_d[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_d .* (Z[:,j] .- Z[:,J]))
        # end
        # dem_d = sum(num_d, dims=2)
        # P_d = num_d ./ dem_d
        # Added: same common-shift stabilization as Question 1.
        num_d = zeros(T, N, J)
        shift_d = vec(maximum(X*bigAlpha .+ gamma_d .* (Z .- Z[:,J:J]), dims=2))
        for j = 1:J
            num_d[:,j] = exp.(X * bigAlpha[:,j] .+ gamma_d .* (Z[:,j] .- Z[:,J]) .- shift_d)
        end
        dem_d = sum(num_d, dims=2)
        P_d = num_d ./ dem_d
        
        # Add to running average
        # P_integrated .+= P_d / D
        P_integrated .+= P_d / D
    end
    
    # TODO: Compute log-likelihood
    # loglike = -sum(bigY .* log.(P_integrated))
    # Added: selecting observed choices avoids 0*log(0) on unchosen alternatives.
    loglike = -sum(log.(P_integrated[bigY .== 1]))
    
    return loglike
end

#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------

function optimize_mlogit(X, Z, y)
    K = size(X, 2)
    J = size(Z, 2)  # Correction: count available alternatives, even in small tests.
    
    # Starting values: K*(J-1) alphas + 1 gamma
    # TODO: You might want to use estimates from PS3 as starting values
    # Original random-start option: startvals = [2*rand(K*(J-1)).-1; 0.1]
    # Added: use the PS3 MNL estimates from our completed PS3 run.
    startvals = [0.055707609298218594, 0.08343063279609504, -2.3448879648298564, 0.04500069129170546, 0.7365814269885503, -3.153244667114907, 0.09264598266845206, -0.08417244629879995, -4.273280688096721, 0.023903340425153616, 0.723070952452537, -3.749394548433251, 0.03608719405857005, -0.6437576097574926, -4.279686120855719, 0.08531086087497199, -1.1714254376180888, -6.678676671579062, 0.0866201075060891, -0.7978730681222609, -4.969132753402835, -0.09419508388004806]
    
    # TODO: Use optimize() function with automatic differentiation
    # Hint: Use LBFGS() algorithm with autodiff = Optim.ADTypes.AutoForwardDiff()
    # initialize the twice differentiable object
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y),
                             startvals, autodiff = Optim.ADTypes.AutoForwardDiff())
    
    result = optimize(td, startvals, LBFGS(), 
                     Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
        
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, result.minimizer)
    result_se = sqrt.(diag(inv(H)))
    # These are the assignment's inverse-Hessian SEs, not person-clustered SEs.
    println("Optimizer converged: ", Optim.converged(result))
    println("Negative log-likelihood: ", Optim.minimum(result))
    println("Hessian positive definite: ", isposdef(Symmetric(H)))
    return result.minimizer, result_se
end

function optimize_mixed_logit_quad(X, Z, y; theta_mnl=nothing)
    K = size(X, 2)  
    J = size(Z, 2)  # Correction: count available alternatives, even in small tests.
    
    # Get quadrature nodes and weights
    nodes, weights = lgwt(7, -4, 4)
    
    # Starting values: K*(J-1) alphas + mu_gamma + sigma_gamma
    # TODO: Use regular logit estimates as starting values for alpha and gamma
    startvals = [2*rand(K*(J-1)).-1; 0.1; 1.0]  # last element is sigma_gamma
    
    # Added: use Question 1 estimates when supplied by allwrap().
    if theta_mnl !== nothing
        startvals = [theta_mnl[1:end-1]; theta_mnl[end]; 1.0]
    end
    # TODO: Set up optimization (DON'T ACTUALLY RUN - TOO SLOW!)
    # result = optimize(theta -> mixed_logit_quad(theta, X, Z, y, nodes, weights),
    #                  startvals, LBFGS(),
    #                  Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
    #                  autodiff = Optim.ADTypes.AutoForwardDiff())
    
    # Completed setup, following Question 1; intentionally NOT executed:
    # td = TwiceDifferentiable(theta -> mixed_logit_quad(theta, X, Z, y, nodes, weights),
    #                          startvals; autodiff=Optim.ADTypes.AutoForwardDiff())
    # result = optimize(td, startvals, LBFGS(),
    #                   Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    # estimates = result.minimizer
    println("Mixed logit quadrature optimization setup complete (not executed)")
    return startvals  # Return starting values instead of running
end

function optimize_mixed_logit_mc(X, Z, y; theta_mnl=nothing)
    K = size(X, 2)
    J = size(Z, 2)  # Correction: count available alternatives, even in small tests.
    
    D = 1000  # Number of Monte Carlo draws
    
    # Starting values: same as quadrature version
    startvals = [2*rand(K*(J-1)).-1; 0.1; 1.0]
    
    # Added: use Question 1 estimates when supplied by allwrap().
    if theta_mnl !== nothing
        startvals = [theta_mnl[1:end-1]; theta_mnl[end]; 1.0]
    end
    # TODO: Set up optimization (DON'T ACTUALLY RUN - TOO SLOW!)
    # result = optimize(theta -> mixed_logit_mc(theta, X, Z, y, D),
    #                  startvals, LBFGS(),
    #                  Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true);
    #                  autodiff = Optim.ADTypes.AutoForwardDiff())
    
    # Completed setup, following Question 1; intentionally NOT executed:
    # td = TwiceDifferentiable(theta -> mixed_logit_mc(theta, X, Z, y, D),
    #                          startvals; autodiff=Optim.ADTypes.AutoForwardDiff())
    # result = optimize(td, startvals, LBFGS(),
    #                   Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    # estimates = result.minimizer
    println("Mixed logit Monte Carlo optimization setup complete (not executed)")
    return startvals  # Return starting values instead of running
end

