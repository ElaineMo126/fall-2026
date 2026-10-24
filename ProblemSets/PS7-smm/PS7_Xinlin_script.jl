using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM
import SMM
include(joinpath(@__DIR__, "PS7_Xinlin_source.jl"))

################################################################################
# Main Function - Question 6: Wrap everything in a function
################################################################################
function main(; D=1000, Nsim=100_000, run_q4=true)
    Random.seed!(2026)
    println("="^80)
    println("Problem Set 7: GMM and SMM Estimation")
    println("="^80)

    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    # Use a local copy if available; otherwise use the starter's course URL.
    local_data = joinpath(@__DIR__, "nlsw88.csv")
    df, X_wage, y_wage = load_data(isfile(local_data) ? local_data : url)
    # The starter's variable names are retained, but y_wage is married (PS2 Q2).
    df, X, y = prepare_occupation_data(df)
    y = Int.(y)
    options = Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true, show_every=50)

    #--------------------------------------------------------------------------
    # Question 1: OLS via GMM
    #--------------------------------------------------------------------------
    println("\nQuestion 1: OLS Estimation via GMM (outcome = married)")
    # Estimate β using GMM
    β_hat_gmm = Optim.optimize(b -> ols_gmm(b, X_wage, y_wage),
                              rand(size(X_wage,2)), LBFGS(),
                              Optim.Options(g_tol=1e-6, iterations=100_000))
    # Compare with closed-form OLS: (X'X)^(-1)X'y
    β_ols = (X_wage' * X_wage) \ (X_wage' * y_wage)
    println("GMM estimates: ", Optim.minimizer(β_hat_gmm))
    println("OLS estimates: ", β_ols)
    println("Maximum absolute difference: ", maximum(abs.(Optim.minimizer(β_hat_gmm)-β_ols)))
    println("Converged: ", Optim.converged(β_hat_gmm))
    # g(beta)=y-X*beta; W=I_N; g'Wg is residual sum of squares.
    # Its first-order condition X'(y-X*beta)=0 yields the OLS formula.

    #--------------------------------------------------------------------------
    # Question 2: Multinomial Logit via MLE and GMM
    #--------------------------------------------------------------------------
    println("\nQuestion 2: Multinomial Logit via MLE and GMM")
    # Get starting values from series of binary logits
    # Create dummy variables for each occupation
    for j = 1:7
        tempname = Symbol(string("occ", j))
        df[!, tempname] = df.occupation .== j
    end
    # Run binary logits for each occupation
    svals = zeros(size(X,2), 7)
    for j = 1:7
        tempname = Symbol(string("occ", j))
        # glm with Binomial/LogitLink, not lm (the commented starter used lm).
        svals[:,j] = coef(glm(X, Float64.(df[!,tempname]), Binomial(), LogitLink()))
    end
    # Take differences relative to base category
    svals = svals[:,1:6] .- svals[:,7]
    svals = vec(svals)
    # Estimate via MLE
    α_hat_mle = Optim.optimize(a -> mlogit_mle(a,X,y), svals, LBFGS(), options)
    # Estimate via GMM using MLE estimates as starting values
    α_hat_gmm_mle_start = Optim.optimize(a -> mlogit_gmm_overid(a,X,y),
                                       copy(Optim.minimizer(α_hat_mle)), LBFGS(), options)
    # Estimate via GMM using random starting values
    random_start = rand(length(svals))
    α_hat_gmm_random_start = Optim.optimize(a -> mlogit_gmm_overid(a,X,y),
                                          random_start, LBFGS(), options)
    # Compare estimates and objective function values
    for (label, result) in (("MLE",α_hat_mle), ("GMM: MLE start",α_hat_gmm_mle_start),
                            ("GMM: random start",α_hat_gmm_random_start))
        println("\n",label," coefficients (rows: intercept, age, white, collgrad; columns: occupations 1:6; base=7):")
        display(reshape(Optim.minimizer(result), size(X,2), 6))
        println("Own objective: ",Optim.minimum(result),"; converged: ",Optim.converged(result))
        println("Common GMM criterion: ",mlogit_gmm_overid(Optim.minimizer(result),X,y))
    end
    println("Max coefficient difference between GMM starts: ",
            maximum(abs.(Optim.minimizer(α_hat_gmm_mle_start)-Optim.minimizer(α_hat_gmm_random_start))))
    # Discuss: Is the objective function globally concave?
    # No: the squared-probability-error criterion is generally neither globally
    # convex nor globally concave. Its negative is not globally concave either.
    # Different starts can reach different local minima or saturated flat regions.
    # Even matching results from these two starts do not prove global optimality.
    # This N*J residual criterion is nonlinear least squares, so its estimates need
    # not equal MLE. The separate mlogit_gmm function uses score moments instead;
    # that just-identified criterion can share the MLE solution. It is not Q2(b).
    println("See the comments above for the global-concavity discussion.")

    #--------------------------------------------------------------------------
    # Question 3: Simulate and Recover Parameters
    #--------------------------------------------------------------------------
    println("\nQuestion 3: Simulate Data and Recover Parameters")
    β_true = hcat([1.0,-1.0,0.5], [-2.0,0.5,0.3], [0.0,-0.5,2.0], zeros(3))
    # Simulate data
    ySim, XSim = sim_logit(Nsim,4; rng=MersenneTwister(2027), β=β_true)
    # Estimate parameters from simulated data
    α_hat_sim = Optim.optimize(a -> mlogit_mle(a,XSim,ySim), zeros(9), LBFGS(), options)
    # Compare estimated parameters to true parameters
    println("True β used in simulation (including the zero base column):")
    display(β_true)
    println("Estimated β from simulated data:")
    display(hcat(reshape(Optim.minimizer(α_hat_sim),3,3),zeros(3)))
    println("Maximum absolute recovery error: ",maximum(abs.(Optim.minimizer(α_hat_sim)-vec(β_true[:,1:3]))))
    println("Converged: ",Optim.converged(α_hat_sim))
    # Try alternative simulation method with Gumbel; estimate and compare
    ySim_gumbel, XSim_gumbel = sim_logit_w_gumbel(Nsim,4; rng=MersenneTwister(2028), β=β_true)
    α_hat_sim_gumbel = Optim.optimize(a -> mlogit_mle(a,XSim_gumbel,ySim_gumbel), zeros(9), LBFGS(), options)
    println("Gumbel-method estimates:")
    display(hcat(reshape(Optim.minimizer(α_hat_sim_gumbel),3,3),zeros(3)))
    println("Maximum absolute recovery error: ",maximum(abs.(Optim.minimizer(α_hat_sim_gumbel)-vec(β_true[:,1:3]))))
    println("Converged: ",Optim.converged(α_hat_sim_gumbel))
    # Recovery is approximate due to finite-sample variation, not exact equality.

    #--------------------------------------------------------------------------
    # Question 4: SMM.jl example, Lecture 9 slide 21 (absent from starter main)
    #--------------------------------------------------------------------------
    println("\nQuestion 4: SMM.jl normal example")
    if run_q4
        smm_normal_example()
    else
        println("Q4 explicitly skipped for a quick local run; run with run_q4=true for submission.")
    end

    #--------------------------------------------------------------------------
    # Question 5: Multinomial Logit via SMM
    #--------------------------------------------------------------------------
    println("\nQuestion 5: Multinomial Logit via SMM; D=",D)
    # Note: This will be slow! Start with small D (like 100) for testing.
    # Then increase to 1000-2000 for final estimates; default here is 1000.
    # Correction to starter: hard simulated choices give a discontinuous,
    # piecewise-constant objective. Finite-difference LBFGS can falsely stop with
    # a zero gradient. Use derivative-free NelderMead with fixed simulation draws.
    α_hat_smm = Optim.optimize(th -> mlogit_smm_overid(th,X,y,D),
                              copy(Optim.minimizer(α_hat_mle)), NelderMead(),
                              Optim.Options(iterations=3000, show_trace=true, show_every=50))
    # Compare SMM estimates to MLE and GMM estimates
    println("MLE estimates: ",Optim.minimizer(α_hat_mle))
    println("GMM estimates: ",Optim.minimizer(α_hat_gmm_mle_start))
    println("SMM estimates: ",Optim.minimizer(α_hat_smm))
    println("SMM criterion at MLE start: ",mlogit_smm_overid(Optim.minimizer(α_hat_mle),X,y,D))
    println("SMM final criterion: ",Optim.minimum(α_hat_smm))
    println("SMM converged: ",Optim.converged(α_hat_smm),"; iterations: ",Optim.iterations(α_hat_smm))
    # A convergence flag is not proof of global optimality for this nonsmooth
    # criterion. D controls simulation error; don't report a small-D trial as final.
    println("\nEstimation Complete!")
    return β_hat_gmm, α_hat_mle, α_hat_gmm_mle_start, α_hat_gmm_random_start, α_hat_sim, α_hat_smm
end

# Run main function
@time main()
