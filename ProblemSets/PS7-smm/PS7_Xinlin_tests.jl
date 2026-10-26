using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
include(joinpath(@__DIR__, "PS7_Xinlin_source.jl"))

function run_tests()
    @testset "PS7 short unit tests" begin
        @testset "Data preparation" begin
            df = DataFrame(age=[25,30,35,40], race=[1,2,1,2], collgrad=[0,1,1,0],
                           wage=[1.0,2.0,4.0,8.0], occupation=[1,8,13,missing])
            mktemp() do path, io
                close(io)
                CSV.write(path,df)
                loaded, X, y = load_data(path)
                @test y ≈ log.([1.0,2.0,4.0,8.0])  # updated demo: log wage
                @test X == [1 25 1 0;1 30 0 1;1 35 1 1;1 40 0 0]
                occ, Xocc, yocc = prepare_occupation_data(loaded)
                @test yocc == [1,7,7]
                @test size(Xocc) == (3,4)
            end
        end
        @testset "OLS criterion and closed form" begin
            X = hcat(ones(5), collect(-2.0:2.0))
            y = [0.2,0.8,2.1,2.7,4.2]
            β = X \ y
            @test ols_gmm(β,X,y) ≈ sum((y-X*β).^2)
            result = Optim.optimize(b -> ols_gmm(b,X,y), zeros(2), LBFGS())
            @test Optim.minimizer(result) ≈ β atol=1e-5
        end
        @testset "MNL criteria" begin
            X = ones(3,1); y = [1,2,3]; α = zeros(2)
            @test mlogit_mle(α,X,y) ≈ 3log(3)
            @test mlogit_gmm_overid(α,X,y) ≈ 2.0
            @test mlogit_gmm(α,X,y) ≈ 0.0 atol=1e-25
            # Non-uniform probabilities with a hand-computable result.
            a = [log(2.0),log(3.0)]
            @test mlogit_mle(a,X,y) ≈ -log((2/6)*(3/6)*(1/6))
            @test mlogit_gmm_overid(a,X,y) ≈ 3*(1+sum(([2,3,1]./6).^2))-2
            @test isfinite(mlogit_mle([1000.0,-1000.0],X,y))
            @test isfinite(mlogit_mle(α,X,[1,1,1])) # absent sample categories
        end
        @testset "Simulation and common draws" begin
            β = zeros(3,4)
            y1, X1 = sim_logit(200,4; rng=MersenneTwister(1),β=β)
            y2, X2 = sim_logit(200,4; rng=MersenneTwister(1),β=β)
            @test y1 == y2 && X1 == X2
            @test size(X1) == (200,3) && all(v -> 1 <= v <= 4,y1)
            yg, Xg = sim_logit_w_gumbel(200,4; rng=MersenneTwister(2),β=β)
            @test size(Xg) == (200,3) && all(v -> 1 <= v <= 4,yg)
            @test draw_gumbel_choices(Xg,β,MersenneTwister(3)) == draw_gumbel_choices(Xg,β,MersenneTwister(3))
            # Deterministic draw-by-draw reference for the SMM frequency criterion.
            X = hcat(ones(6),collect(1.0:6.0)); y = [1,2,3,1,2,3]
            α = [0.2,-0.1,0.3,0.05]; D = 5
            B = hcat(reshape(α,2,2),zeros(2)); counts = zeros(6,3)
            rng = MersenneTwister(1234)
            for d in 1:D
                shocks = rand(rng,Gumbel(0,1),6,3)
                for i in 1:6
                    j = argmax(vec(X[i,:]'*B) + shocks[i,:])
                    counts[i,j] += 1/D
                end
            end
            observed = [y[i]==j for i in 1:6,j in 1:3]
            expected = sum((observed-counts).^2)
            @test mlogit_smm_overid(α,X,y,D) ≈ expected
            @test mlogit_smm_overid(α,X,y,D) == mlogit_smm_overid(α,X,y,D)
            @test_throws ArgumentError mlogit_smm_overid(α,X,y,0)
        end
    end
    # Q4's external SMM.parallelNormal example is deliberately not executed here;
    # it is run by main(). This short suite also avoids 100,000-row recovery runs.
end
run_tests()
