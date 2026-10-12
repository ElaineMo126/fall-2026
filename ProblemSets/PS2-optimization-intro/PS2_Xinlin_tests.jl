using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
include("PS2_Xinlin_source.jl")
Random.seed!(1234)

@testset "Question 1: maximization" begin
    result = q1()
    x = Optim.minimizer(result)[1]
    @test x ≈ -7.3782434055 atol=1e-4
    @test -Optim.minimum(result) ≈ 964.313383782421 atol=1e-3
    @test abs(-4*x^3 - 30*x^2 - 4*x - 3) < 1e-3
    @test -12*x^2 - 60*x - 4 < 0
end

@testset "ols: known residual sum of squares" begin
    X = [1.0 0.0; 1.0 1.0; 1.0 2.0]
    beta = [2.0, 3.0]
    @test ols(beta, X, [2.0, 5.0, 8.0]) == 0
    @test ols(beta, X, [3.0, 3.0, 10.0]) == 9
end

@testset "logit_like: sign, probabilities, and stability" begin
    X = ones(4, 1)
    y = [1, 1, 1, 0]
    @test logit_like([0.0], X, y) ≈ 4*log(2)
    @test logit_like([log(3.0)], X, y) ≈ -(3*log(0.75)+log(0.25))
    @test logit_like([1000.0], ones(2,1), [1,0]) ≈ 1000
    @test logit_like([-1000.0], ones(2,1), [1,0]) ≈ 1000
end

@testset "mlogit: base category, row normalization, and stability" begin
    X = ones(6, 1)
    y = [1, 2, 2, 3, 3, 3]
    @test mlogit(zeros(2), X, y) ≈ 6*log(3)
    # With base category 3, exp(alpha) = (1/3, 2/3), giving p=(1/6,2/6,3/6).
    a = log.([1/3, 2/3])
    @test mlogit(a, X, y) ≈ -(log(1/6)+2*log(2/6)+3*log(3/6))
    @test mlogit(a, vcat(X,X), vcat(y,y)) ≈ 2*mlogit(a,X,y)
    @test isfinite(mlogit([1000.0,-1000.0], X, y))
    # Different covariates across people: compare with explicit choice probabilities.
    X2 = [1.0 -1.0; 1.0 0.0; 1.0 2.0]
    a2 = [0.3, -0.2, -0.4, 0.5]
    u = X2 * [reshape(a2,2,2) zeros(2,1)]
    p = exp.(u) ./ sum(exp.(u); dims=2)
    @test mlogit(a2, X2, [1,2,3]) ≈ -sum(log(p[i,i]) for i in 1:3)
    @test_throws DimensionMismatch mlogit(zeros(3), X2, [1,2,3])
end

# Integration tests use the actual course data, loaded once.
df = read_data()
@testset "read_data: course data" begin
    @test nrow(df) == 2246
    @test all(name -> name in propertynames(df), [:married,:age,:race,:collgrad,:occupation])
end

@testset "Question 2: Optim, matrix OLS, and GLM agree" begin
    r = q2(df; show_trace=false)
    @test Optim.minimizer(r.result) ≈ r.bols atol=1e-4 rtol=1e-4
    @test r.bols ≈ coef(r.model) atol=1e-8
    @test sqrt.(diag(r.vcov)) ≈ stderror(r.model) atol=1e-8
end

@testset "Questions 3-4: Optim logit agrees with GLM" begin
    r = q3_q4(df; show_trace=false)
    @test Optim.minimizer(r.result) ≈ coef(r.model) atol=1e-4 rtol=1e-4
    @test Optim.minimum(r.result) ≈ -loglikelihood(r.model) atol=1e-5
end

@testset "Question 5: multinomial logit estimates" begin
    r = q5(df; show_trace=false)
    @test size(r.X,1) == count(!ismissing, df.occupation)
    @test sort(unique(r.y)) == collect(1:7)
    @test length(Optim.minimizer(r.result)) == 24
    @test size(r.alpha_mat) == (4,7)
    @test all(r.alpha_mat[:,7] .== 0)
    @test Optim.minimum(r.result) < size(r.X,1)*log(7)
    # Independently evaluate the analytical score at the fitted coefficients.
    u = r.X*r.alpha_mat
    e = exp.(u .- maximum(u; dims=2))
    p = e ./ sum(e; dims=2)
    observed = hcat([r.y .== j for j in 1:7]...)
    score = r.X'*(p[:,1:6] .- observed[:,1:6])
    @test maximum(abs.(score)) < 1e-2
    @test all(abs.(sum(p; dims=2) .- 1) .< 1e-12)
    @test all(isfinite, r.alpha_mat)
    # A second starting point checks agreement of the optimized objective.
    second = optimize(a -> mlogit(a,r.X,r.y), zeros(24), LBFGS(),
        Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false))
    @test Optim.minimum(second) ≈ Optim.minimum(r.result) atol=1e-5
end
println("All PS2 tests completed.")
