using Test, Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, ForwardDiff, LineSearches
include(joinpath(@__DIR__, "PS8_Xinlin_source.jl"))

function run_tests()
    @testset "PS8 short checks" begin
        @testset "Quadrature and joint likelihood" begin
            ξ, ω = lgwt(20, -1, 1)
            @test sum(ω) ≈ 2 atol=1e-12
            @test sum(ω .* ξ.^2) ≈ 2/3 atol=1e-12
            # Small independent reference: integrating a Gaussian factor yields
            # a multivariate normal with covariance lambda*lambda' + diag(sigma^2).
            X = ones(2,1); Xfac = ones(2,1)
            Meas = [0.2 -0.1; -0.3 0.4]; y = [0.1,-0.2]
            # alpha(2), beta(1), gamma(2), delta(1), sigma(3)
            θ = [0.1,-0.1,0.2, 0.3,0.4,0.2, 0.9,1.1,0.8]
            λ = θ[4:6]; σ = θ[7:9]
            μ = θ[1:3]
            dist = MvNormal(μ, Symmetric(λ*λ' + Diagonal(σ.^2)))
            exact = -sum(logpdf(dist, vcat(Meas[i,:],y[i])) for i in 1:2)
            @test factor_model(θ,X,Xfac,Meas,y,80) ≈ exact atol=1e-5
            flipped = copy(θ); flipped[4:6] .*= -1
            @test factor_model(flipped,X,Xfac,Meas,y,80) ≈ factor_model(θ,X,Xfac,Meas,y,80) atol=1e-10
            independent = copy(θ); independent[4:6] .= 0
            expected = -sum(logpdf(Normal(μ[j],σ[j]),vcat(Meas[i,:],y[i])[j]) for i in 1:2,j in 1:3)
            expected -= 2log(cdf(Normal(),5)-cdf(Normal(),-5))
            @test factor_model(independent,X,Xfac,Meas,y,80) ≈ expected atol=1e-9
            @test all(isfinite,ForwardDiff.gradient(t -> factor_model(t,X,Xfac,Meas,y,20),θ))
            @test_throws DimensionMismatch factor_model(θ[1:end-1],X,Xfac,Meas,y,20)
        end
        @testset "Data preparation and regression components" begin
            df = load_data(joinpath(@__DIR__, "nlsy.csv"))
            @test nrow(df) > 0
            # Small subsample only: enough observations for the regression/factor fits.
            small = df[1:min(200,nrow(df)),:]
            X,y,Xfac,asvabs = prepare_factor_matrices(small)
            @test size(X) == (nrow(small),7) && size(Xfac) == (nrow(small),4)
            @test X[:,end] == ones(nrow(small)) && Xfac[:,end] == ones(nrow(small))
            @test y == small.logwage && size(asvabs,2) == 6
            C = Matrix(compute_asvab_correlations(small))
            @test C ≈ cor(asvabs)
            @test diag(C) ≈ ones(6)
            base = estimate_base_regression(small)
            @test coef(base) ≈ hcat(ones(nrow(small)),X[:,1:6]) \ y
            @test length(coef(estimate_full_regression(small))) == 13
            @test length(coef(estimate_pca_regression(small))) == 8
            @test length(coef(estimate_factor_regression(small))) == 8
            table = format_results(collect(1.0:45.0),ones(45),-100.0,asvabs)
            @test nrow(table) == 45
            @test table.variable[31] == "constant" && table.equation[31] == "logwage"
            @test table.variable[38] == "factor_loading" && table.equation[38] == "logwage"
            @test table.variable[45] == "residual_SD"
        end
    end
    # run_estimation's full Newton/Hessian fit is deliberately excluded from this
    # short suite. Its likelihood, AD gradient, matrix construction and result
    # ordering are checked above; main() performs the actual assignment fit.
end
run_tests()
