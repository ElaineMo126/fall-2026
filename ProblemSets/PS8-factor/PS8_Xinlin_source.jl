#==================================================================================
# Question 1: Load data and estimate base regression
==================================================================================#

"""
    load_data(url::String)

Load the NLSY dataset from the given URL and return as a DataFrame.

# Arguments
- `url::String`: URL to the CSV file

# Returns
- DataFrame containing the NLSY data
"""
function load_data(url::String)
    return isfile(url) ? CSV.read(url, DataFrame) : CSV.File(IOBuffer(HTTP.get(url).body)) |> DataFrame
end

"""
    estimate_base_regression(df::DataFrame)

Estimate the baseline wage regression without ASVAB scores.
Model: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results from GLM.lm()
"""
function estimate_base_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
end

#==================================================================================
# Question 2: Compute correlations among ASVAB scores
==================================================================================#

"""
    compute_asvab_correlations(df::DataFrame)

Compute the correlation matrix for the six ASVAB test scores.

# Arguments
- `df::DataFrame`: Data containing the six ASVAB scores

# Returns
- DataFrame with correlation matrix (6x6)

# Hint
- The ASVAB scores are in columns: asvabAR, asvabCS, asvabMK, asvabNO, asvabPC, asvabWK
- Select these columns by name (e.g. df[:, [:asvabAR, :asvabCS, ...]]) rather than position
- Use cor() function on the matrix of ASVAB scores
"""
function compute_asvab_correlations(df::DataFrame)
    # Extract the six ASVAB columns (by name) as a matrix
    #ASVAB_COLS = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    asvabs = df[:, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]]
    # Compute correlation matrix using cor()
    correlation = cor(Matrix{Float64}(asvabs))
    # Return as a formatted DataFrame
    cordf = DataFrame(
        cor1 = correlation[:,1],
        cor2 = correlation[:,2],
        cor3 = correlation[:,3],
        cor4 = correlation[:,4],
        cor5 = correlation[:,5],
        cor6 = correlation[:,6]
    )
    return cordf
end

#==================================================================================
# Question 3: Regression with all ASVAB scores
==================================================================================#

"""
    estimate_full_regression(df::DataFrame)

Estimate wage regression including all six ASVAB scores.
Model: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results from GLM.lm()

# Question to consider:
- Given the correlations you computed, what problems might arise?
"""
function estimate_full_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
end

#==================================================================================
# Question 4: PCA regression
==================================================================================#

"""
    estimate_pca_regression(df::DataFrame)

Perform PCA on ASVAB scores and include first principal component in regression.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results including the first principal component

# Key steps:
1. Extract ASVAB scores as a matrix
2. IMPORTANT: Transpose to J×N (MultivariateStats requires features × observations)
3. Fit PCA model with maxoutdim=1
4. Transform the data to get principal component scores
5. Add PC scores to DataFrame and run regression

# Hints:
- asvabMat = Matrix(df[:, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]])'
  (note the transpose!)
- M = MultivariateStats.fit(PCA, asvabMat; maxoutdim=1)
- asvabPCA = MultivariateStats.transform(M, asvabMat)
- asvabPCA will be 1×N, need to reshape for regression
- The sign of a principal component is arbitrary, so interpret its coefficient with care
"""
function estimate_pca_regression(df::DataFrame)
    # Extract ASVAB matrix and transpose to J×N
    asvabs = df[:, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]]
    asvabMat = Matrix{Float64}(asvabs)'
    
    # Fit PCA model with maxoutdim=1
    M = MultivariateStats.fit(PCA, asvabMat; maxoutdim=1)
    
    # Transform data to get principal component
    asvabPCA = MultivariateStats.transform(M, asvabMat)
    
    # Add PC to dataframe (careful with dimensions!)
    df = @transform(df, :asvabPCA = asvabPCA[:])
    
    println(cor(Matrix{Float64}(df[:, [:grad4yr, :asvabPCA]])))
    
    # Run regression with PC included
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df)
end

#==================================================================================
# Question 5: Factor Analysis regression
==================================================================================#

"""
    estimate_factor_regression(df::DataFrame)

Perform Factor Analysis on ASVAB scores and include first factor in regression.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- Regression results including the first factor

# Note:
- Syntax is nearly identical to PCA, just use FactorAnalysis instead
- Make sure you analyze only the six ASVAB scores. If you added the PCA
  column to df in Question 4, "the last 6 columns" now includes it!
"""
function estimate_factor_regression(df::DataFrame)
    # TODO: Follow same steps as PCA but use FactorAnalysis

    asvabs = df[:, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]]
    asvabMat = Matrix{Float64}(asvabs)'
    M = MultivariateStats.fit(FactorAnalysis, asvabMat; maxoutdim=1)
    asvabFA = MultivariateStats.transform(M, asvabMat)
    df = @transform(df, :asvabFA = asvabFA[:])
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFA), df)

end

#==================================================================================
# Question 6: Full factor model with MLE
==================================================================================#

"""
    prepare_factor_matrices(df::DataFrame)

Prepare the data matrices needed for the factor model estimation.

# Arguments
- `df::DataFrame`: The data containing all variables

# Returns
- `X`: Covariates for wage equation (N×7: black, hispanic, female, schoolt, gradHS, grad4yr, constant)
- `y`: Log wage outcomes (N×1)
- `Xfac`: Covariates for measurement equations (N×4: black, hispanic, female, constant)
- `asvabs`: Matrix of all 6 ASVAB scores (N×6)
"""
function prepare_factor_matrices(df::DataFrame)
    # TODO: Create X matrix for wage equation (include constant at end)
    # TODO: Extract y (logwage)
    # TODO: Create Xfac matrix for measurement equations
    # TODO: Create asvabs matrix

    X = hcat(Matrix{Float64}(df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]]), ones(nrow(df)))
    y = Vector{Float64}(df.logwage)
    Xfac = hcat(Matrix{Float64}(df[:, [:black, :hispanic, :female]]), ones(nrow(df)))
    asvabs = Matrix{Float64}(df[:, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]])
    return X, y, Xfac, asvabs

end

"""
    factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix, 
                 y::Vector, R::Integer) where T<:Real

Compute the negative log-likelihood for the factor model.

# Arguments
- `θ`: Parameter vector containing:
  * α parameters (L×J matrix, vectorized): coefficients in measurement equations
  * β parameters (K×1 vector): coefficients in wage equation  
  * γ parameters (J vector): factor loadings in measurement equations
  * δ parameter (scalar): factor loading in wage equation
  * σ parameters (J+1 vector): standard deviations (J for measurements, 1 for wage)
- `X`: Wage equation covariates (N×K)
- `Xfac`: Measurement equation covariates (N×L)
- `Meas`: ASVAB test scores (N×J)
- `y`: Log wages (N×1)
- `R`: Number of quadrature points

# Returns
- Negative log-likelihood value (scalar)

# Model Structure:
## Measurement equations (for each j=1,...,J):
   asvab_j = Xfac*α_j + γ_j*ξ + ε_j,  ε_j ~ N(0, σ_j²)

## Wage equation:
   logwage = X*β + δ*ξ + ε,  ε ~ N(0, σ_{J+1}²)

## Latent factor:
   ξ ~ N(0,1)

# Likelihood for person i:
   L_i = ∫ [∏_j φ((M_ij - Xfac_i*α_j - γ_j*ξ)/σ_j) / σ_j] 
          × [φ((y_i - X_i*β - δ*ξ)/σ_{J+1}) / σ_{J+1}]
          × φ(ξ) dξ

# Key Steps:
1. Unpack θ into α, β, γ, δ, σ parameters
2. Set up Gauss-Legendre quadrature nodes and weights
3. For each quadrature point:
   a. Compute likelihood contribution from each measurement equation
   b. Compute likelihood contribution from wage equation
   c. Weight by quadrature weight and ξ density
4. Sum log-likelihoods across all observations
5. Return negative for minimization
"""
function factor_model(θ::Vector{T}, X::Matrix, Xfac::Matrix, Meas::Matrix, 
                     y::Vector, R::Integer) where T<:Real
    # Get dimensions
    # TODO: Unpack parameters from θ vector
    # α should be L×J matrix (reshape from θ[1:J*L])
    # β should be K×1 vector
    # γ should be J×1 vector (measurement factor loadings)
    # δ should be a scalar (wage factor loading)
    # σ should be (J+1)×1 vector (standard deviations)
    # TODO: Get quadrature nodes (ξ) and weights (ω) using lgwt()
    # Recommended: lgwt(R, -5, 5) for standard normal integration
    # Initialize likelihood storage
    # TODO: Loop over quadrature points
        # TODO: For each ASVAB test j, compute:
        # - Residuals: M_ij - Xfac*α_j - γ_j*ξ_r
        # - Standardize by σ_j
        # - Evaluate normal PDF
        # - Store in Mlike[:,j]
        # TODO: For wage equation, compute:
        # - Residuals: y_i - X*β - δ*ξ_r
        # - Standardize by σ_{J+1}
        # - Evaluate normal PDF
        # - Store in Ylike
        # TODO: Update likelihood by:
        # like += ω[r] * (product of all Mlike) * Ylike * φ(ξ_r)
    # TODO: Return negative sum of log-likelihoods

    K = size(X, 2)
    L = size(Xfac, 2)
    J = size(Meas, 2)
    N = length(y)
    length(θ) == L*J + K + 2*J + 2 || throw(DimensionMismatch("Incorrect parameter count"))
    size(X,1) == size(Xfac,1) == size(Meas,1) == N || throw(DimensionMismatch("Row counts differ"))
    R >= 2 || throw(ArgumentError("Use at least two quadrature nodes"))
    α = reshape(θ[1:L*J], L, J)
    β = θ[L*J+1:L*J+K]
    γ = θ[L*J+K+1:L*J+K+J]
    δ = θ[L*J+K+J+1]
    σ = θ[L*J+K+J+2:end]
    all(v -> v > 0, σ) || return oftype(θ[1], Inf)
    ξ, ω = lgwt(R, -5, 5)

    # Compute residuals once. Keep the demo's quadrature/test loops.
    Mres = Meas - Xfac*α
    Yres = y - X*β
    # Numerical correction: Mlike/Ylike/like below store LOG densities.
    # This is equivalent to the product in the assignment and avoids underflow.
    like = zeros(T, N)
    Mlike = zeros(T, N, J)
    half_log2pi = log(2*pi)/2
    for r in 1:R
        for j in 1:J
            Mlike[:,j] = -half_log2pi .- log(σ[j]) .-
                         0.5 .* ((Mres[:,j] .- γ[j]*ξ[r]) ./ σ[j]).^2
        end
        Ylike = -half_log2pi .- log(σ[J+1]) .-
                0.5 .* ((Yres .- δ*ξ[r]) ./ σ[J+1]).^2
        # Include BOTH the Lebesgue quadrature weight and the N(0,1) factor density.
        term = vec(sum(Mlike; dims=2)) + Ylike .+
               (log(ω[r]) - half_log2pi - ξ[r]^2/2)
        if r == 1
            like .= term
        else
            for i in 1:N
                hi = max(like[i], term[i])
                like[i] = hi + log(exp(like[i]-hi) + exp(term[i]-hi))
            end
        end
    end
    # [-5,5] truncates a very small normal tail; R controls quadrature resolution.
    return -sum(like)

end

"""
    run_estimation(df::DataFrame, start_vals::Vector)

Run the full MLE estimation procedure for the factor model.

# Arguments
- `df::DataFrame`: The data
- `start_vals::Vector`: Starting values for optimization

# Returns
- `θ̂`: Estimated parameters
- `se`: Standard errors
- `loglike`: Log-likelihood at optimum

# Steps:
1. Prepare data matrices
2. Set up TwiceDifferentiable objective (for Hessian-based optimization)
3. Optimize using Newton method with line search
4. Compute standard errors from inverse Hessian
"""
function run_estimation(df::DataFrame, start_vals::Vector; R::Integer=80)
    # TODO: Prepare data matrices
    # TODO: Create TwiceDifferentiable objective
    # Use autodiff = Optim.ADTypes.AutoForwardDiff() for ForwardDiff
    # TODO: Optimize using Newton method
    # e.g. Newton(linesearch = BackTracking())  (BackTracking comes from LineSearches)
    # Optim.Options: set g_tol, iterations, show_trace as appropriate
    # TODO: Compute Hessian and standard errors
    # TODO: Return estimates, standard errors, and log-likelihood

    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    J = size(asvabs,2)
    nσ = J+1
    cut = length(start_vals)-nσ
    all(start_vals[cut+1:end] .> 0) || throw(ArgumentError("Starting SDs must be positive"))
    # Preserve factor_model's parameter order/SD interpretation. Optimize log(SD)
    # internally so Newton steps can never propose a negative standard deviation.
    η_start = vcat(start_vals[1:cut], log.(start_vals[cut+1:end]))
    unpack = η -> vcat(η[1:cut], exp.(η[cut+1:end]))
    objective = η -> factor_model(unpack(η), X, Xfac, asvabs, y, R)
    # Explicit ForwardDiff callbacks avoid version-specific autodiff keyword syntax.
    g! = (G, η) -> ForwardDiff.gradient!(G, objective, η)
    h! = (H, η) -> ForwardDiff.hessian!(H, objective, η)
    objective_twice = Optim.TwiceDifferentiable(objective, g!, h!, η_start)
    result = Optim.optimize(objective_twice, η_start,
                            Newton(linesearch=BackTracking()),
                            Optim.Options(g_tol=1e-5, iterations=1000,
                                          show_trace=true, show_every=10))
    println("Converged: ", Optim.converged(result))
    println("Iterations: ", Optim.iterations(result))
    if !Optim.converged(result)
        @warn "Optimizer did not converge; do not treat the estimates as final."
    end
    η̂ = Optim.minimizer(result)
    θ̂ = unpack(η̂)
    H = ForwardDiff.hessian(objective, η̂)
    # Delta method converts log(SD) standard errors to SD standard errors.
    if isposdef(Symmetric(H))
        Vη = inv(Symmetric(H))
        jacobian_diag = vcat(ones(cut), θ̂[cut+1:end])
        se = sqrt.(diag(Vη)) .* jacobian_diag
    else
        @warn "Hessian is not positive definite; standard errors are reported as NaN."
        se = fill(NaN, length(θ̂))
    end
    # Factor sign is arbitrary: report a positive first measurement loading.
    first_loading = size(Xfac,2)*J + size(X,2) + 1
    if θ̂[first_loading] < 0
        θ̂[first_loading:first_loading+J] .*= -1
    end
    loglike = -Optim.minimum(result)
    return θ̂, se, loglike

end

"""
    format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)

Format estimation results into a readable DataFrame.

# Arguments
- `θ`: Estimated parameters
- `se`: Standard errors
- `loglike`: Log-likelihood value
- `asvabs`: ASVAB matrix (used to get J)

# Returns
- DataFrame with columns: equation, variable, coefficient, std_error
"""
function format_results(θ::Vector, se::Vector, loglike::Real, asvabs::Matrix)
    # TODO: Create vectors for equation names and variable names
    # Structure should match the parameter ordering in θ
    # TODO: Combine into DataFrame with appropriate column names

    J = size(asvabs,2)
    J == 6 || throw(ArgumentError("This assignment's labels require six ASVAB tests"))
    tests = ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
    equation = String[]
    variable = String[]
    # Column-major vec(alpha): all four covariates of test 1, then test 2, etc.
    for test in tests, v in ["black", "hispanic", "female", "constant"]
        push!(equation, test); push!(variable, v)
    end
    for v in ["black", "hispanic", "female", "schoolt", "gradHS", "grad4yr", "constant"]
        push!(equation, "logwage"); push!(variable, v)
    end
    for test in tests
        push!(equation, test); push!(variable, "factor_loading")
    end
    push!(equation, "logwage"); push!(variable, "factor_loading")
    for eq in vcat(tests, ["logwage"])
        push!(equation, eq); push!(variable, "residual_SD")
    end
    length(θ) == length(se) == length(equation) || throw(DimensionMismatch("Results have wrong length"))
    println("Joint log likelihood: ", loglike)
    return DataFrame(equation=equation, variable=variable, coefficient=θ, std_error=se)

end

#==================================================================================
# Main execution function
==================================================================================#

"""
    main()

Execute the complete analysis workflow for Problem Set 8.

This function runs through all questions sequentially:
1. Load data and run base regression
2. Compute ASVAB correlations
3. Run full regression with all ASVABs
4. Run PCA regression
5. Run Factor Analysis regression
6. Estimate full factor model via MLE
"""
function main()
    # Data URL
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
    
    println("="^80)
    println("Problem Set 8: Factor Models and Dimension Reduction")
    println("="^80)
    
    # Load data
    println("\nLoading data...")
    local_data = joinpath(@__DIR__, "nlsy.csv")
    df = load_data(isfile(local_data) ? local_data : url)
    println("Data loaded successfully. Dimensions: ", size(df))
    
    # Question 1
    println("\n" * "="^80)
    println("Question 1: Base Regression (without ASVAB)")
    println("="^80)
    # Call estimate_base_regression() and display results
    est = estimate_base_regression(df)
    display(coeftable(est))
    
    # Question 2
    println("\n" * "="^80)
    println("Question 2: ASVAB Correlations")
    println("="^80)
    cordfout = compute_asvab_correlations(df)
    println(cordfout)
    println("Consider: Are these correlations high? What might this imply?")
    # Q2-Q3: High correlations imply multicollinearity: individual score effects
    # can be imprecise/unstable. This does not itself bias OLS coefficients or
    # invalidate joint prediction. Including scores may reduce omitted ability bias.
    # PCA/FA summarize shared variation, but one component need not capture it all.
    
    # Question 3
    println("\n" * "="^80)
    println("Question 3: Full Regression (with all ASVAB)")
    println("="^80)
    estfull = estimate_full_regression(df)
    display(coeftable(estfull))
    println("Consider: How do results compare to Question 1? Any concerns?")
    
    # Question 4
    println("\n" * "="^80)
    println("Question 4: PCA Regression")
    println("="^80)
    estpca = estimate_pca_regression(df)
    display(coeftable(estpca))
    # Removed the demo's early return so Questions 5 and 6 are reached.
    
    # Question 5
    println("\n" * "="^80)
    println("Question 5: Factor Analysis Regression")
    println("="^80)
    # TODO: Call estimate_factor_regression() and display results
    estfactor = estimate_factor_regression(df)
    display(coeftable(estfactor))
    
    # Question 6
    println("\n" * "="^80)
    println("Question 6: Full Factor Model (MLE)")
    println("="^80)
    
    # TODO: Prepare starting values
    # Hint: Can use OLS estimates as starting values:
    # - For α: regress each ASVAB on Xfac
    # - For β: regress wage on X
    # - For γ and δ: start with nonzero values (e.g. 0.5); all-zero loadings are a
    #   stationary point because the likelihood is symmetric in their sign
    # - For σ: start with sample standard deviations or 0.5
    
    X, y, Xfac, asvabs = prepare_factor_matrices(df)
    α_start = Xfac \ asvabs
    β_start = X \ y
    measurement_sd = vec(std(asvabs - Xfac*α_start; dims=1))
    wage_sd = std(y - X*β_start)
    γ_start = 0.5 .* measurement_sd
    δ_start = 0.5 * wage_sd
    σ_start = sqrt.(vcat(measurement_sd.^2 - γ_start.^2, wage_sd^2 - δ_start^2))
    start_vals = vcat(vec(α_start), β_start, γ_start, δ_start, σ_start)

    println("\nEstimating full factor model...")
    # TODO: Call run_estimation() with starting values
    θ̂, se, loglike = run_estimation(df, start_vals)
    
    # TODO: Format and display results
    results = format_results(θ̂, se, loglike, asvabs)
    show(stdout, MIME"text/plain"(), results); println()
    
    println("\n" * "="^80)
    println("Analysis complete!")
    println("="^80)
end

#==================================================================================
# Question 7: Unit Tests
==================================================================================#
function lgwt(N0::Integer,a::Real=-1,b::Real=1)
    # This script is for computing definite integrals using Legendre-Gauss 
    # Quadrature. Computes the Legendre-Gauss nodes and weights on an interval
    # [a,b] with truncation order N
    #
    # Suppose you have a continuous function f(x) which is defined on [a,b]
    # which you can evaluate at any x in [a,b]. Simply evaluate it at all of
    # the values contained in the x vector to obtain a vector f. Then compute
    # the definite integral using sum(f.*w);

    
    N  = N0-1
    N1 = N+1
    N2 = N+2
    
    xu = range(-1,stop=1,length=N1)
    
    # Initial guess
    y = cos.((2*(0:N) .+ 1)*pi/(2*N .+ 2))  .+  ( 0.27/N1 ) .* sin.( pi .* xu .* N/N2 )
    
    # Legendre-Gauss Vandermonde Matrix
    L  = zeros(N1,N2)
    
    # Derivative of LGVM
    Lp = zeros(N1,N2)
    
    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method
    
    y0 = 2
    
    
    vareps = 2e-52
    # Iterate until new points are uniformly within epsilon of old points
    
    
    i = 0
    tracker=0
    it_max=10
    while (norm(y.-y0,Inf)>vareps && tracker<=it_max)
        d=norm(y.-y0,Inf)
    
        L[:,1]  .= 1
        Lp[:,1] .= 0
        
        L[:,2] .= y
        
        for k=2:N1
            L[:,k+1] = ( (2*k-1)*y .* L[:,k] .- (k-1)*L[:,k-1] )/k
        end
        
        Lp = (N2)*( L[:,N1] .- y .* L[:,N2] )./(1 .- y.^2)
        y0 = y
        y  = y0 - L[:,N2]./Lp
        if norm(y.-y0,Inf)==d
            tracker+=1
        end
        i+=1

    end
    
    # Linear map from[-1,1] to [a,b]
    x = (a.*(1 .- y) .+ b .* (1 .+ y))./2
    
    # Compute the weights
    w=(b-a)./((1 .- y.^2).*Lp.^2)*(N2/N1)^2
    
    return x,w
end
