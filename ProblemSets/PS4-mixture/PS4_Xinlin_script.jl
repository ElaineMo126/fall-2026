using ForwardDiff, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions
include(joinpath(@__DIR__, replace(basename(@__FILE__), "Xinlin_script.jl" => "Xinlin_source.jl")))

#---------------------------------------------------
# Question 6: Main Function
#---------------------------------------------------

function allwrap()
    println("=== Problem Set 4: Multinomial and Mixed Logit ===")
    
    Random.seed!(6343)
    # Load data
    df, X, Z, y = load_data()
    
    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))
    
    # Question 1: Estimate multinomial logit
    println("\n=== QUESTION 1: MULTINOMIAL LOGIT RESULTS ===")
    # TODO: Uncomment when ready to estimate
    # theta_hat_mle = optimize_mlogit(X, Z, y)
    # println("Estimates: ", theta_hat_mle)
    # alpha_hat = theta_hat_mle[1:end-1]
    # gamma_hat = theta_hat_mle[end]
    # println("γ̂ = ", gamma_hat)
    # Correction: optimize_mlogit returns BOTH estimates and standard errors.
    theta_hat_mle, result_se = optimize_mlogit(X, Z, y)
    println("Estimates: ", theta_hat_mle)
    println("Standard errors: ", result_se)
    alpha_hat = theta_hat_mle[1:end-1]
    gamma_hat = theta_hat_mle[end]
    println("γ̂ = ", gamma_hat, "; SE = ", result_se[end])
    println("Coefficient order: (age, white, collgrad) for occupations 1:7, then gamma.")
    
    # Question 2: Interpret gamma
    println("\n=== QUESTION 2: INTERPRETATION ===")
    # Added: Question 2 writeup.
    println("PS3 gamma was approximately -0.094195; PS4 gamma is ", gamma_hat, ".")
    if gamma_hat > 0
        println("The positive PS4 coefficient is more consistent with workers preferring higher expected wages.")
    else
        println("The PS4 coefficient is not positive, so it does not restore the usual positive wage interpretation.")
    end
    println("Holding X and other wages fixed, a one-unit rise in Z_j changes log(P_j/P_8) by gamma.")
    println("A 1% rise in exp(Z_j) changes the odds by approximately ", gamma_hat, "%.")
    println("The own-probability derivative is gamma*P_j*(1-P_j).")
    println("The PS4 sample contains repeated person-year observations; this comparison does not by itself identify a causal effect.")
    
    # Question 3: Practice with quadrature and Monte Carlo
    quad_practice = practice_quadrature()
    quad_variance = variance_quadrature() 
    mc_practice = practice_monte_carlo()
    
    # Added: Question 3d comparison.
    println("Quadrature uses deterministic nodes and unequal weights; uniform MC uses random nodes and equal (b-a)/D weights.")
    # Question 4: Mixed logit with quadrature (setup only)
    println("\n=== QUESTION 4: MIXED LOGIT QUADRATURE (SETUP) ===")
    start_quad = optimize_mixed_logit_quad(X, Z, y; theta_mnl=theta_hat_mle)
    
    # Question 5: Mixed logit with Monte Carlo (setup only)  
    println("\n=== QUESTION 5: MIXED LOGIT MONTE CARLO (SETUP) ===")
    start_mc = optimize_mixed_logit_mc(X, Z, y; theta_mnl=theta_hat_mle)
    
    println("\n=== ALL ANALYSES COMPLETE ===")
    return (theta=theta_hat_mle, se=result_se, quad_practice=quad_practice,
            quad_variance=quad_variance, mc_practice=mc_practice,
            start_quad=start_quad, start_mc=start_mc)
end

# TODO: Uncomment to run
# allwrap()

# println("Starter code loaded successfully!")
# println("Remember to:")
# println("1. Fill in all TODO sections")
# println("2. Test functions step by step")  
# println("3. Don't run mixed logit estimations (too computationally intensive)")
# println("4. Use automatic differentiation in optimization")

# Added: run allwrap at the bottom, save output, and allow inclusion by tests.
if abspath(PROGRAM_FILE) == @__FILE__
    output_file = joinpath(@__DIR__, replace(basename(@__FILE__), "Xinlin_script.jl" => "Xinlin_output.txt"))
    open(output_file, "w") do io
        redirect_stdout(io) do
            allwrap()
        end
    end
    print(read(output_file, String))
end
