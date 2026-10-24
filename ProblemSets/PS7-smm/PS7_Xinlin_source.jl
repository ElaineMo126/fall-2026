################################################################################
# Data Loading and Preparation Functions
################################################################################

"""
    load_data(url)

Load PS2 data and create design matrix for the married linear probability model.
Returns: DataFrame, X matrix, married indicator vector
"""
function load_data(url)
    # Completed starter task: Create X matrix with intercept, age, race indicator, college grad indicator
    # Completed starter task: Create y as log wage
    # X = ...
    # y = (df.log.wage))

    # Correction: PS7 Q1 refers to PS2 Q2, whose outcome is married, NOT log wage.
    df = isfile(url) ? CSV.read(url, DataFrame) : CSV.read(HTTP.get(url).body, DataFrame)
    X = hcat(ones(nrow(df)), df.age, df.race .== 1, df.collgrad .== 1)
    y = Float64.(df.married .== 1)
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
    # Completed starter task: Compute predicted values ŷ = Xβ
    # ŷ = X*β
    # Completed starter task: Compute residuals g = y - ŷ
    # g = y - ̂y
    # Completed starter task: Compute objective function J = g'Ig
    # Hint: Use the identity matrix or just g'*g
    # J = g'*g
    # prefer  g'*g will return a 1*1 matrix, which I then have to return J[1]

    ŷ = X * β
    g = y - ŷ
    # For Julia vectors g'g is already a scalar; no J[1] is needed.
    J = dot(g, g)  # g'I_N*g, without allocating an N-by-N identity matrix
    return J
end

################################################################################
# Question 2: Multinomial Logit via MLE and GMM
################################################################################

"""
    mlogit_mle(α, X, y)

Maximum likelihood objective function for multinomial logit.

Model:
    P(y_i = j) = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    where β_J = 0 (normalization)

Arguments:
- α: vectorized coefficients of dimension K*(J-1)
- X: N×K covariate matrix
- y: N×1 choice vector (integer values 1,2,...,J)

Returns: negative log-likelihood value
"""
function mlogit_mle(α, X, y)
    # Completed starter task: Create N×J matrix of choice indicators (bigY)
    # bigY[i,j] = 1 if y[i]==j, 0 otherwise
    # for j = 1:J
    #     bigY[:,j] = ...
    # end
    # Completed starter task: Reshape α into K×(J-1) matrix and append zeros for normalization
    # bigα = [reshape(α, K, J-1) zeros(K)]
    # Completed starter task: Compute choice probabilities P
    # P[i,j] = exp(X[i,:]'β_j) / sum_k(exp(X[i,:]'β_k))
    # Hint: Use broadcasting and eachrow
    # P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))
    # Completed starter task: Compute negative log-likelihood: -Σ_i Σ_j d_ij * log(P_ij)
    # loglike = ...

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
    # Use log probabilities directly; do not take log of an underflowed P.
    logP = V .- log.(sum(exp.(V); dims=2))
    loglike = -sum(bigY .* logP)
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
    # Completed starter task: Create choice indicator matrix
    # for j = 1:J
    #     bigY[:,j] = ...
    # end
    # Completed starter task: Reshape coefficients
    # bigα = ...
    # Completed starter task: Compute choice probabilities
    # P = ...
    # Completed starter task: Compute moment vector g of dimension K*(J-1)
    # Each element is: mean((d_ij - P_ij) * X_ik)
    # for j = 1:(J-1)
    #     for k = 1:K
    #         g[(j-1)*K + k] = ...
    #     end
    # end
    # Completed starter task: Compute objective function J = N * g'g
    # J = ...

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
    J = N * dot(g, g)
    return J
end

"""
    mlogit_gmm_overid(α, X, y)

Over-identified GMM for multinomial logit.

Uses N*J moments: d_ij - P_ij(α) for all i,j
This is over-identified since we only have K*(J-1) parameters.
"""
function mlogit_gmm_overid(α, X, y)
    # Completed starter task: Create choice indicators
    # for j = 1:J
    #     bigY[:,j] = ...
    # end
    # Completed starter task: Reshape and compute probabilities
    # bigα = ...
    # P = ...
    # Completed starter task: Stack moments as vector: g = vec(d - P)
    # This creates an N*J dimensional vector
    # g = bigY[:] .- P[:]
    # Completed starter task: Compute objective J = g'Wg with W = I
    # J = ...

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
    # This is the N*J vector explicitly required by Question 2(b)-(c).
    g = vec(bigY - P)
    J = dot(g, g)
    return J
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
    # Completed starter task: Generate X matrix
    # Include intercept, and K-1 other covariates with different distributions
    # Example: X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))
    # X = ...
    # Completed starter task: Create coefficient matrix β (dimension K×J)
    # Last column should be zeros (normalization)
    # Example for J=4: β = hcat([1, -1, 0.5], [-2, 0.5, 0.3], [0, -0.5, 2], zeros(3))
        # β = ...
        # Generate random coefficients
        # β = ...
    # Completed starter task: Compute choice probabilities P (dimension N×J)
    # P_ij = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    # P = ...
    # Completed starter task: Draw uniform random variables
    # draw = rand(N)
    # Completed starter task: Generate choices based on cumulative probabilities
    # For each person i, find j such that:
    # Σ_{k=j}^J P_ik > ε_i
    # for j = 1:J
    #     # Hint: sum(P[:,j:J]; dims=2) gives cumulative probabilities
    #     # Y += ...
    # end

    N > 0 && J > 2 || throw(ArgumentError("N must be positive and J > 2"))
    X = hcat(ones(N), randn(rng, N), 2 .+ 2 .* randn(rng, N))
    if isnothing(β)
        if J == 4
            β = hcat([1.0,-1.0,0.5], [-2.0,0.5,0.3], [0.0,-0.5,2.0], zeros(3))
        else
            β = hcat(randn(rng, 3, J-1), zeros(3))
        end
    end
    size(β) == (3,J) || throw(DimensionMismatch("beta must be 3 by J"))
    all(iszero, β[:,end]) || throw(ArgumentError("last beta column must be zero"))
    V = X * β
    V = V .- maximum(V; dims=2)
    P = exp.(V) ./ sum(exp.(V); dims=2)
    draw = rand(rng, N)
    Y = zeros(Int, N)
    for j = 1:J
        # Tail-sum rule in the assignment; the j=1 tail is exactly 1.
        tail = j == 1 ? ones(N) : vec(sum(P[:,j:J]; dims=2))
        Y .+= tail .> draw
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
    # Completed starter task: Generate X and β (same as above)
    # X = ...
    # β = ...
    # Completed starter task: Draw Gumbel errors
    # ε = rand(Gumbel(0,1), N, J)
    # Completed starter task: Choose alternative that maximizes utility
    # Y_i = argmax_j (X_i'β_j + ε_ij)
    # Hint: Use argmax.(eachrow(...))
    # Y = ...

    N > 0 && J > 2 || throw(ArgumentError("N must be positive and J > 2"))
    X = hcat(ones(N), randn(rng, N), 2 .+ 2 .* randn(rng, N))
    if isnothing(β)
        if J == 4
            β = hcat([1.0,-1.0,0.5], [-2.0,0.5,0.3], [0.0,-0.5,2.0], zeros(3))
        else
            β = hcat(randn(rng, 3, J-1), zeros(3))
        end
    end
    size(β) == (3,J) || throw(DimensionMismatch("beta must be 3 by J"))
    all(iszero, β[:,end]) || throw(ArgumentError("last beta column must be zero"))
    Y = draw_gumbel_choices(X, β, rng)
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
    # Completed starter task: Create actual choice indicators
    # for j = 1:J
    #     bigY[:,j] = ...
    # end
    # Completed starter task: Initialize simulated choice frequencies
    # Completed starter task: Reshape parameters
    # bigα = ...
    # Completed starter task: Simulate D datasets and accumulate frequencies
    # for d = 1:D
    #     # Draw Gumbel shocks
    #     # ε = ...
    #     
    #     # Generate simulated choices
    #     # ỹ = argmax_j(X*bigα + ε)
    #     # ỹ = ...
    #     
    #     # Update frequency counts
    #     # for j = 1:J
    #     #     bigỸ[:,j] .+= (ỹ .== j) * (1/D)
    #     # end
    # end
    # Completed starter task: Compute moment vector (actual - simulated frequencies)
    # g = bigY[:] .- bigỸ[:]
    # Completed starter task: Compute objective function
    # J = ...

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
    J = dot(g, g)
    return J
end


# Shared Q3 simulation component, reused by Q5 (conditional on observed X).
function draw_gumbel_choices(X, β, rng)
    ε = rand(rng, Gumbel(0,1), size(X,1), size(β,2))
    Y = argmax.(eachrow(X * β + ε))
    return Y
end


################################################################################
# Question 4: Lecture 9 slide 21 -- the actual SMM.jl example
################################################################################
function smm_normal_example()
    MA = SMM.parallelNormal()  # The lecture warns this can take about five minutes.
    dc = SMM.history(MA.chains[1])
    dc = dc[dc[!, :accepted] .== true, :]
    display(describe(dc))
    println("Compare the p1 and p2 means above with the true values -1 and 1.")
    return MA, dc
end
