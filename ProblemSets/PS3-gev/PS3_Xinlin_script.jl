using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables
include(joinpath(@__DIR__, replace(basename(@__FILE__), "_script.jl" => "_source.jl")))

#---------------------------------------------------
# Main Function (Question 4)
#---------------------------------------------------

function allwrap()
    # Added: make the starter random initial values reproducible.
    Random.seed!(6343)
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"
    # Added: prefer the supplied local file; use the corrected course URL otherwise.
    local_file = joinpath(@__DIR__, "nlsw88w.csv")
    df, X, Z, y = load_data(isfile(local_file) ? local_file : url)
    
    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))
    
    # TODO: Estimate multinomial logit
    println("\n=== MULTINOMIAL LOGIT RESULTS ===")
    # theta_hat_mle = optimize_mlogit(X, Z, y)
    theta_hat_mle = optimize_mlogit(X, Z, y)
    # println("Estimates: ", theta_hat_mle)
    println("Estimates: ", theta_hat_mle)
    println("beta matrix: rows = age, white, collgrad; columns = occupations 1:7")
    show(stdout, "text/plain", reshape(theta_hat_mle[1:end-1], size(X,2), 7))
    println()
    println("gamma = ", theta_hat_mle[end])

    # Added: Question 2 in the PDF (the starter labels Nested Logit Question 2).
    # Z_j is expected log wage. Holding X and other Z values fixed,
    # d log(P_j/P_8)/d Z_j = gamma. Gamma is not a probability marginal effect.
    # A 1% increase in exp(Z_j) changes the odds by approximately gamma percent.
    # For j != 8, d P_j/d Z_j = gamma*P_j*(1-P_j).
    println("A one-unit increase in Z_j changes log(P_j/P_8) by ", theta_hat_mle[end])
    println("A 1% increase in exp(Z_j) changes these odds by approximately ",
            theta_hat_mle[end], "%.")
    println("This is a model association, not automatically a causal effect.")
    
    # TODO: Estimate nested logit
    println("\n=== NESTED LOGIT RESULTS ===")
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]  # WC and BC occupations
    # nlogit_theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
    nlogit_theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
    # println("Estimates: ", nlogit_theta_hat)
    println("Estimates: ", nlogit_theta_hat)
    println("Order: beta_WC (age, white, collgrad), beta_BC (age, white, collgrad), lambda_WC, lambda_BC, gamma")
    if any(nlogit_theta_hat[end-2:end-1] .< 0.01)
        println("CAUTION: lambda is near zero; this is a boundary approximation, not a stable interior estimate.")
    end
    if any(nlogit_theta_hat[end-2:end-1] .> 1)
        println("CAUTION: lambda > 1 violates the usual global RUM consistency restriction.")
    end
    return theta_hat_mle, nlogit_theta_hat
end

# Uncomment to run
# allwrap()
# Added: save the printed estimates and optimizer trace, then display the output.
# Skip execution when this script is included by the tests.
if abspath(PROGRAM_FILE) == @__FILE__
    output_file = joinpath(@__DIR__, replace(basename(@__FILE__), "_script.jl" => "_output.txt"))
    open(output_file, "w") do io
        redirect_stdout(io) do
            allwrap()
        end
    end
    print(read(output_file, String))
end