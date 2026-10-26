################################################################################
# Data Loading and Preparation Functions
################################################################################

"""
    load_data(url)

Load wage data and create design matrix for OLS regression.
Returns: DataFrame, X matrix, log wage vector
"""
function load_data(url)
    # Local data are optional; otherwise use the unchanged course URL.
    df = isfile(url) ? CSV.read(url, DataFrame) : CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :log_wage = log.(:wage))
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.log_wage
    # could also do:
    # y = log.(df.wage)
    return df, X, y
end


"""
    prepare_occupation_data(df)

Prepare occupation data for multinomial logit.
Collapse occupation categories and create covariates.
"""
function prepare_occupation_data(df)
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    df.white = df.race .== 1
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
    return df, X, y
end

################################################################################
# Question 1: OLS via GMM
################################################################################

"""
    ols_gmm(β, X, y)

GMM objective function for OLS regression.
Uses identity weighting matrix.

Mathematical form:
    J(β) = g'Wg where W = I and g = (y - Xβ)

Arguments:
- β: coefficient vector
- X: N×K design matrix
- y: N×1 outcome vector

Returns: scalar objective function value
"""
function ols_gmm(β, X, y)
    # Compute predicted values ŷ = Xβ
    ŷ = X*β
    
    # Compute residuals g = y - ŷ
    g = y - ŷ
    
    # Compute objective function J = g'Ig
    # Hint: Use the identity matrix or just g'*g
    J = g⋅g # g \cdot g
    # I prefer \cdot because it will return a scalar;
    # Julia detail: with vector g, g'g also returns a scalar.
    
    return J
end

################################################################################
# Question 2: Multinomial Logit via MLE and GMM
################################################################################

"""
    mlogit_mle(alpha, X, y)

Maximum likelihood objective function for multinomial logit.

Model:
    P(y_i = j) = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    where β_J = 0 (normalization)

Arguments:
- alpha: vectorized coefficients of dimension K*(J-1)
- X: N×K covariate matrix
- y: N×1 choice vector (integer values 1,2,...,J)

Returns: negative log-likelihood value
"""
function mlogit_mle(alpha, X, y)
    N = size(X,1) # no. of obs
    K = size(X,2) # no. of covariates
    J = length(alpha) ÷ K + 1 # number of choice alternatives
    
    # unslice the parameter vector
    alpha_mat = [reshape(alpha, K, J-1) zeros(K, 1)] # add a column of zeros for the base category
    
    # make d matrix
    d = zeros(N, J)
    for j = 1:J
        d[:,j] .= y .== j
    end
    
    # make p matrix
    # Same corrected row-wise logit formula as the demo, evaluated stably.
    V = X * alpha_mat
    V .-= maximum(V; dims=2)
    denom = vec(sum(exp.(V); dims=2))
    p = zeros(N, J)
    for j = 1:J
        p[:,j] .= exp.(V[:,j]) ./ denom
    end
    
    # log-likelihood expression
    logp = V .- reshape(log.(denom), N, 1)
    loglike = -sum(d .* logp)
    
    return loglike
end

"""
    mlogit_gmm(α, X, y)

GMM objective function for multinomial logit using just-identified moments.

Moment conditions:
    E[X_i(d_ij - P_ij(α))] = 0 for j = 1,...,J-1

This gives K*(J-1) moments for K*(J-1) parameters.
"""
function mlogit_gmm(α, X, y)
    # for j = 1:J
    # end
    # Each element is: mean((d_ij - P_ij) * X_ik)
    # for j = 1:(J-1)
    #     for k = 1:K
    #     end
    # end

    K = size(X, 2)
    # Infer J from parameters so simulation samples may omit a category.
    length(α) % K == 0 || throw(DimensionMismatch("length(alpha) must be divisible by K"))
    J = div(length(α), K) + 1
    N = length(y)
    N == size(X,1) || throw(DimensionMismatch("X and y have different row counts"))
    all(v -> v isa Integer && 1 <= v <= J, y) || throw(ArgumentError("choices must be integers 1:J"))
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end
    bigα = [reshape(α, K, J-1) zeros(K)]
    V = X * bigα
    V = V .- maximum(V; dims=2)  # stable softmax, same choice probabilities
    P = exp.(V) ./ sum(exp.(V); dims=2)
    g = zeros((J-1)*K)
    for j = 1:(J-1)
        for k = 1:K
            g[(j-1)*K+k] = mean((bigY[:,j] .- P[:,j]) .* X[:,k])
        end
    end
    obj = N * dot(g, g)
    return obj
end

"""
    mlogit_gmm_overid(α, X, y)

Over-identified GMM for multinomial logit.

Uses N*J moments: d_ij - P_ij(α) for all i,j
This is over-identified since we only have K*(J-1) parameters.
"""
function mlogit_gmm_overid(α, X, y)
    K = size(X, 2)
    J = length(α) ÷ K + 1 # α has K*(J-1) elements
    N = length(y)
    
    # Create choice indicators
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end
    
    # Reshape and compute probabilities
    bigα = [reshape(α, K, J-1) zeros(K, 1)] # add a column of zeros for the base category
    # Same corrected row-wise logit formula as the demo, evaluated stably.
    V = X * bigα
    V .-= maximum(V; dims=2)
    denom = vec(sum(exp.(V); dims=2))
    P = zeros(N, J)
    for j = 1:J
        P[:,j] .= exp.(V[:,j]) ./ denom
    end
    
    # Stack moments as vector: g = vec(d - P)
    # This creates an N*J dimensional vector
    # NOTE: THIS IS NOT -SUM(D .* LOG(P)) AS IN MLE
    g = bigY[:] .- P[:]
    
    # Compute objective obj = g'Wg with W = I
    obj = g⋅g # g \cdot g
    
    return obj 
end

################################################################################
# Question 3: Simulate Data from Multinomial Logit
################################################################################

"""
    sim_logit(N, J)

Simulate multinomial logit data using inverse CDF method.

Steps:
1. Generate X ~ N(μ, Σ)
2. Set coefficient matrix β (K×J)
3. Compute choice probabilities P_ij
4. Draw uniform random variable ε_i
5. Assign y_i based on which cumulative probability bracket ε_i falls in

Arguments:
- N: number of observations (default: 100,000)
- J: number of choice alternatives (default: 4)

Returns: (Y, X) where Y is N×1 choice vector, X is N×K covariate matrix
"""
function sim_logit(N=100_000, J=4; rng=Random.default_rng(), β=nothing)
    # Generate X matrix (let's choose K=3 covariates)
    # Include intercept, and K-1 other covariates with different distributions
    X = hcat(ones(N), randn(rng, N), 2 .+ 2 .* randn(rng, N))
    
    # Create coefficient matrix β (dimension K×J)
    # Last column should be zeros (normalization)
    # Example for J=4: β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
    if isnothing(β)
        if J == 4
            β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
        else
            # Generate random coefficients
            β = hcat(2 .* rand(rng, size(X,2), J-1) .- 1, zeros(size(X,2)))
        end
    end
    size(β) == (3,J) || throw(DimensionMismatch("beta must be 3 by J"))
    
    # Compute choice probabilities P (dimension N×J)
    # P_ij = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    V = X * β
    V .-= maximum(V; dims=2)
    denom = vec(sum(exp.(V); dims=2))
    P = zeros(N, J)
    for j = 1:J
        P[:,j] .= exp.(V[:,j]) ./ denom
    end
    
    # Draw uniform random variables
    draw = rand(rng, N)
    
    # Generate choices based on cumulative probabilities
    # For each person i, find j such that:
    # Σ_{k=j}^J P_ik > ε_i
    Y = zeros(Int, N)
    for j = 1:J
        # Hint: sum(P[:,j:J]; dims=2) gives cumulative probabilities
        Y += vec(draw .< sum(P[:,j:J]; dims=2))
    end
    
    return Y, X
end

"""
    sim_logit_w_gumbel(N, J)

Simulate multinomial logit data using Gumbel shocks (alternative method).

This method directly uses the fact that if ε ~ Gumbel(0,1), then
    y_i = argmax_j (X_i'β_j + ε_ij)
generates choices from a multinomial logit model.

This is often simpler and more numerically stable than the inverse CDF method.
"""
function sim_logit_w_gumbel(N=100_000, J=4; rng=Random.default_rng(), β=nothing)
    # Generate X matrix (let's choose K=3 covariates)
    # Include intercept, and K-1 other covariates with different distributions
    X = hcat(ones(N), randn(rng, N), 2 .+ 2 .* randn(rng, N))
    
    # Create coefficient matrix β (dimension K×J)
    # Last column should be zeros (normalization)
    # Example for J=4: β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
    if isnothing(β)
        if J == 4
            β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
        else
            # Generate random coefficients
            β = hcat(2 .* rand(rng, size(X,2), J-1) .- 1, zeros(size(X,2)))
        end
    end
    size(β) == (3,J) || throw(DimensionMismatch("beta must be 3 by J"))
    
    # Draw Gumbel errors
    # Draws and argmax are shared with Q5 through the helper below.
    
    # Choose alternative that maximizes utility
    # Y_i = argmax_j (X_i'β_j + ε_ij)
    # Hint: Use argmax.(eachrow(...))
    Y = draw_gumbel_choices(X, β, rng)
    # X*β should be N x J
    # ε is also N x J
    
    return Y, X
end

################################################################################
# Question 5: Multinomial Logit via SMM
################################################################################

"""
    mlogit_smm_overid(α, X, y, D)

Simulated Method of Moments for multinomial logit.

Algorithm:
1. For given parameters α, compute model-implied probabilities
2. Simulate D datasets using the Gumbel method
3. Compute average simulated choice frequencies
4. Match simulated frequencies to actual frequencies

Arguments:
- α: parameter vector
- X: covariate matrix
- y: actual choices
- D: number of simulation draws

Returns: SMM objective function value
"""
function mlogit_smm_overid(α, X, y, D)
    # for j = 1:J
    # end
    # for d = 1:D
    #     # Draw Gumbel shocks
    #     
    #     # Generate simulated choices
    #     # ỹ = argmax_j(X*bigα + ε)
    #     
    #     # Update frequency counts
    #     # for j = 1:J
    #     #     bigỸ[:,j] .+= (ỹ .== j) * (1/D)
    #     # end
    # end
    # g = bigY[:] .- bigỸ[:]

    K = size(X, 2)
    D > 0 && D isa Integer || throw(ArgumentError("D must be a positive integer"))
    length(α) % K == 0 || throw(DimensionMismatch("invalid parameter length"))
    J = div(length(α), K) + 1
    N = length(y)
    N == size(X,1) || throw(DimensionMismatch("X and y have different row counts"))
    all(v -> v isa Integer && 1 <= v <= J, y) || throw(ArgumentError("choices must be integers 1:J"))
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:,j] = y .== j
    end
    bigỸ = zeros(N, J)
    bigα = [reshape(α, K, J-1) zeros(K)]
    # Same shocks at every parameter guess, without resetting the caller's RNG.
    rng = MersenneTwister(1234)
    for d = 1:D
        ỹ = draw_gumbel_choices(X, bigα, rng)  # reuse Q3's simulation code
        for j = 1:J
            bigỸ[:,j] .+= (ỹ .== j) ./ D
        end
    end
    g = vec(bigY - bigỸ)
    obj = dot(g, g)
    return obj
end

################################################################################
# Main Function - Question 6: Wrap everything in a function
################################################################################

"""
    main()

Main function that runs all estimation procedures.
Keeps everything out of global scope for better performance.
"""
function main(; run_q4=true, run_q5=false, D=1000)
    println("="^80)
    println("Problem Set 7: GMM and SMM Estimation")
    println("="^80)
    
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    local_data = joinpath(@__DIR__, "nlsw88.csv")
    df, X_wage, y_wage = load_data(isfile(local_data) ? local_data : url)
    df, X, y = prepare_occupation_data(df)
    
    #--------------------------------------------------------------------------
    # Question 1: OLS via GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 1: OLS Estimation via GMM")
    println("="^80)
    
    # Estimate β using GMM
    # Hint: Use optimize() with LBFGS algorithm
    β_hat_gmm = optimize(b -> ols_gmm(b, X_wage, y_wage), 
                         rand(size(X_wage,2)), 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    
    # Compare with closed-form OLS: (X'X)^(-1)X'y
    β_ols = X_wage \ y_wage
    
    println("GMM estimates: ", β_hat_gmm.minimizer)
    println("OLS estimates: ", β_ols)
    
    #--------------------------------------------------------------------------
    # Question 2: Multinomial Logit via MLE and GMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 2: Multinomial Logit via MLE and GMM")
    println("="^80)
    
    # Get starting values from the demo's series of LPMs
    # Create dummy variables for each occupation
    for j = 1:7
        tempname = Symbol(string("occ", j))
        df[:, tempname] = df.occupation .== j
    end
    
    # Run LPMs for each occupation
    svals = zeros(size(X,2), 7)
    for j = 1:7
        tempname = Symbol(string("occ", j))
        f = term(tempname) ~ term(:age) + term(:white) + term(:collgrad)
        svals[:,j] = coef(lm(f, df))
    end
    # Take differences relative to base category
    svals = svals[:,1:6] .- svals[:,7]
    svals = svals[:]
    
    # 2(a)
    # Estimate via MLE
    α_hat_mle = optimize(a -> mlogit_mle(a, X, y), 
                         svals, 
                         LBFGS(), 
                         Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    println("MLE estimates: ", α_hat_mle.minimizer)
    
    # 2(b)
    # Estimate via GMM using MLE estimates as starting values
    α_hat_gmm_mle_start = optimize(a -> mlogit_gmm_overid(a, X, y), 
                                    α_hat_mle.minimizer, 
                                    LBFGS(), 
                                    Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    println("GMM estimates (MLE start): ", α_hat_gmm_mle_start.minimizer)
    println("MLE estimates: ", α_hat_mle.minimizer)
    
    # 2(c)
    # Estimate via GMM using random starting values
    α_hat_gmm_random_start = optimize(a -> mlogit_gmm_overid(a, X, y), 
                                       rand(length(svals)), 
                                       LBFGS(), 
                                       Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    println("GMM estimates (random start): ", α_hat_gmm_random_start.minimizer)
    println("MLE estimates: ", α_hat_mle.minimizer)
    
    # Compare estimates and objective function values
    println("GMM objective (MLE start): ", α_hat_gmm_mle_start.minimum)
    println("GMM objective (random start): ", α_hat_gmm_random_start.minimum)
    println("Max abs difference in estimates: ",
            maximum(abs.(α_hat_gmm_mle_start.minimizer .- α_hat_gmm_random_start.minimizer)))
    
    #--------------------------------------------------------------------------
    # Question 3: Simulate and Recover Parameters
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 3: Simulate Data and Recover Parameters")
    println("="^80)
    
    # Simulate data
    ySim, XSim = sim_logit(100_000, 4)
    ySimG, XSimG = sim_logit_w_gumbel(100_000, 4)
    
    println("Note: X and ε drawn separately so they won't match exactly")
    println("but with same β and distribution of X and large N they should be close")
    println("freq of j==1 in ySim: ",mean(ySim.==1))
    println("freq of j==1 in ySimG: ",mean(ySimG.==1))
    
    # Estimate parameters from simulated data and compare method
    α_hat_sim = optimize(a -> mlogit_mle(a, XSim, ySim), 
                         rand(9),  # 3 covariates × 3 non-base alternatives
                         LBFGS(), 
                         Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    
    # Try alternative simulation method with Gumbel
    α_hat_simG = optimize(a -> mlogit_mle(a, XSimG, ySimG), 
                         rand(9),  # 3 covariates × 3 non-base alternatives
                         LBFGS(), 
                         Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=100))
    
    println("True β used in simulation:", hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3)))
    println("Estimated β from simulated data:")
    println("Method 1 (sim_logit): ", α_hat_sim.minimizer)
    println("Method 2 (sim_logit_w_gumbel): ", α_hat_simG.minimizer)
    
    # Question 4: Lecture 9 slide 21; run after Q3 so its draws cannot affect
    # reproduction of the demo's Q1-Q3 random sequence.
    println("\nQuestion 4: SMM.jl normal example")
    if run_q4
        smm_normal_example()
    else
        println("Q4 skipped explicitly; use run_q4=true for the full assignment.")
    end

    #--------------------------------------------------------------------------
    # Question 5: Multinomial Logit via SMM
    #--------------------------------------------------------------------------
    println("\n" * "="^80)
    println("Question 5: Multinomial Logit via SMM")
    println("="^80)
    
    # Teacher update: Question 5 needs to be coded, but need not be run.
    α_hat_smm = nothing
    if run_q5
        # Discrete simulated choices are non-differentiable: use Nelder-Mead.
        α_hat_smm = Optim.optimize(th -> mlogit_smm_overid(th, X, y, D),
                                  copy(α_hat_mle.minimizer), NelderMead(),
                                  Optim.Options(iterations=1000, show_trace=true))
        println("MLE estimates: ", α_hat_mle.minimizer)
        println("GMM estimates: ", α_hat_gmm_mle_start.minimizer)
        println("SMM estimates: ", α_hat_smm.minimizer)
        println("SMM objective: ", α_hat_smm.minimum)
        println("SMM converged: ", Optim.converged(α_hat_smm))
    else
        println("Q5 coded but not run, following the instructor's update.")
    end

    println("\n" * "="^80)
    println("Estimation Complete!")
    println("="^80)
    
    # Return estimates for testing
    return β_hat_gmm, α_hat_mle, α_hat_gmm_mle_start, α_hat_gmm_random_start, α_hat_sim, α_hat_smm
end


function draw_gumbel_choices(X, β, rng)
    ε = rand(rng, Gumbel(0,1), size(X,1), size(β,2))
    Y = argmax.(eachrow(X * β + ε))
    return Y
end

function smm_normal_example()
    MA = SMM.parallelNormal()  # The lecture warns this can take about five minutes.
    dc = SMM.history(MA.chains[1])
    dc = dc[dc[!, :accepted] .== true, :]
    display(describe(dc))
    println("Compare the p1 and p2 means above with the true values -1 and 1.")
    return MA, dc
end

# Q2(c): The stacked-residual GMM criterion is generally neither globally convex
# nor globally concave. Different initial values may reach different solutions.
# The MNL LOG likelihood is concave; mlogit_mle returns its negative (convex).
# The score-moment mlogit_gmm function is not the N*J criterion requested in Q2.
# Reproduction: use the demo's Julia/package versions for the closest numerical
# agreement. Stable logit evaluation can change final floating-point digits.
