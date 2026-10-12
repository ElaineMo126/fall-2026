using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
include(joinpath(@__DIR__, "PS5_Xinlin_source.jl"))

# Independent, tiny reference: recurse over next states using UNDISCOUNTED V.
# This checks that the production code's already-discounted FV does not use beta twice.
function reference_value(θ, d, z, x, b, t)
    t > d.T && return 0.0
    row0 = (z-1)*d.xbin + 1
    next_values = [reference_value(θ, d, z, k, b, t+1) for k in 1:d.xbin]
    v0 = d.β * sum(d.xtran[row0,k] * next_values[k] for k in 1:d.xbin)
    v1 = θ[1] + θ[2]*d.xval[x] + θ[3]*b +
         d.β * sum(d.xtran[row0+x-1,k] * next_values[k] for k in 1:d.xbin)
    return log(exp(v0) + exp(v1))
end

function run_tests()
    @testset "PS5" begin
        @testset "Provided state grids and transitions" begin
            zval, zbin, xval, xbin, xtran = create_grids()
            @test (zbin, xbin) == (101, 201)
            @test size(xtran) == (zbin*xbin, xbin)
            @test (first(zval), last(zval)) == (0.25, 1.25)
            @test (first(xval), last(xval)) == (0.0, 25.0)
            @test all(isfinite, xtran)
            @test minimum(xtran) >= -1e-12
            @test maximum(abs.(sum(xtran, dims=2) .- 1)) < 1e-12
            @test xtran[1,1] ≈ 1-exp(-0.25*0.125)
            @test xtran[xbin,xbin] ≈ 1.0
            @test all(abs.(xtran[xbin,1:xbin-1]) .< 1e-12)
            @test xtran[xbin+1,1] ≈ 1-exp(-0.26*0.125)
        end
        @testset "Data loaders and long-panel alignment" begin
            df_long = load_static_data()
            raw = CSV.read(joinpath(@__DIR__, "busdataBeta0.csv"), DataFrame)
            @test nrow(df_long) == 20*nrow(raw)
            @test length(unique(zip(df_long.bus_id, df_long.time))) == nrow(df_long)
            for (i,t) in ((1,1), (1,20), (500,9), (1000,20))
                r = (i-1)*20+t
                @test df_long.Y[r] == raw[i, Symbol("Y$t")]
                @test df_long.Odometer[r] == raw[i, Symbol("Odo$t")]
                @test df_long.Branded[r] == raw.Branded[i]
                @test df_long.time[r] == t
            end
            fit = estimate_static_model(df_long)
            @test length(coef(fit)) == 3
            @test all(isfinite, coef(fit))
            design = hcat(ones(nrow(df_long)), df_long.Odometer, df_long.Branded)
            @test maximum(abs.(design'*(df_long.Y .- predict(fit)))) < 0.01
            d = load_dynamic_data()
            @test size(d.Y) == size(d.X) == size(d.Xstate) == (1000,20)
            @test (d.N, d.T, d.β) == (1000,20,0.9)
            @test all(y -> y in (0,1), d.Y)
            @test all(b -> b in (0,1), d.B)
            @test all(x -> 1 <= x <= d.xbin, d.Xstate)
            @test all(z -> 1 <= z <= d.zbin, d.Zstate)
        end
        d = (N=2, T=2, zbin=2, xbin=2, xval=[0.0,1.0], β=0.9,
             xtran=[0.7 0.3; 0.0 1.0; 0.2 0.8; 0.0 1.0],
             Y=[1 0; 0 1], X=[0.2 1.1; 1.3 0.4],
             Xstate=[1 2; 2 1], Zstate=[1,2], B=[0,1])
        θ = [0.4,-0.7,0.3]
        @testset "Backward recursion and terminal condition" begin
            FV = fill(99.0, 4,2,3)
            @test compute_future_value!(FV, θ, d) === nothing
            @test all(iszero, FV[:,:,3])
            for t in 1:2, b in 0:1, z in 1:2, x in 1:2
                @test FV[(z-1)*2+x,b+1,t] ≈ d.β*reference_value(θ,d,z,x,b,t)
            end
            compute_future_value!(FV, θ, merge(d,(β=0.0,)))
            @test all(iszero, FV)
            compute_future_value!(FV, [1000.0,-0.1,0.2], d)
            @test all(isfinite, FV)
        end
        @testset "Observed-state likelihood" begin
            expected = 0.0
            for i in 1:d.N, t in 1:d.T
                z, b, x = d.Zstate[i], d.B[i], d.Xstate[i,t]
                V = [reference_value(θ,d,z,k,b,t+1) for k in 1:2]
                r0 = (z-1)*2+1
                delta = θ[1]+θ[2]*d.X[i,t]+θ[3]*b +
                        d.β*sum((d.xtran[r0+x-1,k]-d.xtran[r0,k])*V[k] for k in 1:2)
                prob = exp(delta)/(1+exp(delta))
                expected -= d.Y[i,t]==1 ? log(prob) : log(1-prob)
            end
            @test log_likelihood_dynamic(θ,d) ≈ expected
            myopic = sum(log(1+exp(θ[1]+θ[2]*d.X[i,t]+θ[3]*d.B[i])) -
                         d.Y[i,t]*(θ[1]+θ[2]*d.X[i,t]+θ[3]*d.B[i]) for i in 1:2,t in 1:2)
            @test log_likelihood_dynamic(θ,merge(d,(β=0.0,))) ≈ myopic
            @test log_likelihood_dynamic(zeros(3),d) ≈ 4*log(2)
            @test isfinite(log_likelihood_dynamic([1000.0,0.0,0.0],d))
            @test isfinite(log_likelihood_dynamic([-1000.0,0.0,0.0],d))
        end
        @testset "Optimizer recovers a known finite MLE" begin
            # Four (mileage,brand) cells; frequencies exactly match theta=(0,-log(3),log(3)).
            X = reshape(repeat([0.0,1.0,0.0,1.0],inner=20),80,1)
            B = repeat([0,0,1,1],inner=20)
            Y = reshape(vcat([vcat(ones(Int,k),zeros(Int,20-k)) for k in (10,5,15,10)]...),80,1)
            fixture = (N=80,T=1,zbin=1,xbin=2,xval=[0.0,1.0],β=0.0,
                       xtran=[0.7 0.3;0.0 1.0],Y=Y,X=X,B=B,
                       Xstate=Int.(X).+1,Zstate=ones(Int,80))
            result = estimate_dynamic_model(fixture; θ_start=zeros(3))
            @test Optim.converged(result)
            @test Optim.minimizer(result) ≈ [0.0,-log(3),log(3)] atol=1e-4
            @test Optim.minimum(result) < log_likelihood_dynamic(zeros(3),fixture)
        end
    end
end
run_tests()
