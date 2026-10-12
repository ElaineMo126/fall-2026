using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
include(joinpath(@__DIR__, "PS5_Xinlin_source.jl"))

################################################################################
# MAIN EXECUTION WRAPPER (Question 3f)
################################################################################

"""
    main()

Main wrapper function that runs all estimation procedures.
Wrapping everything in a function (rather than running in global scope) 
is a Julia best practice for performance.
"""
function main()
    println("\n" * "="^70)
    println("PROBLEM SET 5: BUS ENGINE REPLACEMENT MODEL")
    println("="^70)
    
    #---------------------------------------------------------------------------
    # Part 1: Static Estimation (Questions 1-2)
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 1: STATIC (MYOPIC) ESTIMATION")
    println("-"^70)
    
    # Load data
    println("\nLoading and reshaping data...")
    df_long = load_static_data()
    println("Observations: ", nrow(df_long))
    println("First few rows:")
    println(first(df_long, 5))
    
    # Estimate static model
    println("\nEstimating static logit model...")
    θ̂_static = estimate_static_model(df_long)
    
    println("\nStatic estimates:")
    println(coeftable(θ̂_static))
    println("Static log likelihood: ", loglikelihood(θ̂_static))
    # Extract coefficients to use as starting values for dynamic model
    θ_start_dynamic = coef(θ̂_static)

    #---------------------------------------------------------------------------
    # Part 2: Dynamic Estimation (Question 3)
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 2: DYNAMIC ESTIMATION")  
    println("-"^70)
    
    # Load data
    println("\nLoading dynamic data...")
    d = load_dynamic_data()
    println("Buses (N): ", d.N)
    println("Time periods (T): ", d.T)
    println("Mileage bins: ", d.xbin)
    println("Route usage bins: ", d.zbin)
    println("Total state space size: ", d.xbin * d.zbin)
    println("Discount factor (β): ", d.β)
    
    # Estimate dynamic model
    println("\nSetting up dynamic estimation...")
    estimate_dynamic_model(d, θ_start=θ_start_dynamic)
    
    println("\n" * "="^70)
    println("END OF PROBLEM SET")
    println("="^70)
    
    return nothing
end

main()
