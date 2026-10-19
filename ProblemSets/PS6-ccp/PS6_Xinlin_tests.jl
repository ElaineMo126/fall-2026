# Unit tests requested in Question 4.

using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
include(joinpath(@__DIR__, "PS6_Xinlin_source.jl"))

@testset "PS6 CCP functions" begin

    @testset "Question 1: load and reshape" begin
        mktempdir() do dir
            test_data = DataFrame(
                Y1=[1,0], Y2=[0,1],
                Odo1=[1.0,2.0], Odo2=[1.5,2.5],
                Xst1=[2,3], Xst2=[3,4],
                RouteUsage=[0.5,0.7], Zst=[1,2], Branded=[0,1]
            )
            path = joinpath(dir, "small_busdata.csv")
            CSV.write(path, test_data)

            df_long, Xstate, Zstate, Branded = load_and_reshape_data(path)

            @test nrow(df_long) == 4
            @test names(df_long) == ["bus_id", "RouteUsage", "Branded", "Y", "time", "Odometer"]
            @test df_long.bus_id == [1,1,2,2]
            @test df_long.time == [1,2,1,2]
            @test df_long.Y == [1,0,0,1]
            @test Xstate == [2 3; 3 4]
            @test Zstate == [1,2]
            @test Branded == [0,1]
        end
    end

    @testset "Question 2: flexible logit" begin
        # A rich synthetic sample gives the fully interacted formula enough
        # independent observations without downloading the full bus data.
        Random.seed!(6343)
        n = 400
        df = DataFrame(
            Odometer = 5 .* rand(n),
            RouteUsage = 0.25 .+ rand(n),
            Branded = rand(0:1, n),
            time = 1 .+ 19 .* rand(n)
        )
        p = 1.0 ./ (1.0 .+ exp.(-(-0.4 .+ 0.08 .* df.Odometer .-
                                  0.3 .* df.Branded)))
        df.Y = Int.(rand(n) .< p)

        flex_logit = estimate_flexible_logit(df)
        p_hat = predict(flex_logit, df)
        @test length(p_hat) == n
        @test all((0 .<= p_hat) .& (p_hat .<= 1))
    end

    @testset "Question 3a-3b: grids and state space" begin
        xval = [0.0, 1.0]
        zval = [0.5, 1.0]
        state_df = construct_state_space(2, 2, xval, zval)

        @test nrow(state_df) == 4
        @test state_df.Odometer == [0.0,1.0,0.0,1.0]
        @test state_df.RouteUsage == [0.5,0.5,1.0,1.0]
        @test all(state_df.Branded .== 0)
        @test all(state_df.time .== 0)

        zgrid, zbin, xgrid, xbin, xtran = create_grids()
        @test size(xtran) == (zbin*xbin, xbin)
        @test all(isapprox.(sum(xtran, dims=2), 1.0; atol=1e-10))
        @test length(zgrid) == zbin
        @test length(xgrid) == xbin
    end

    @testset "Question 3b: CCP future values" begin
        Random.seed!(6)
        training = DataFrame(
            Odometer = repeat([0.0,1.0], 50),
            RouteUsage = repeat([0.5,1.0], inner=50),
            Branded = repeat([0,1], 50),
            time = repeat(collect(1:10), 10)
        )
        training.Y = Int.(rand(100) .< 0.5)
        simple_logit = glm(@formula(Y ~ Odometer + RouteUsage + Branded + time),
                           training, Binomial(), LogitLink())
        state_df = construct_state_space(2, 1, [0.0,1.0], [0.5])
        xtran = [0.8 0.2; 0.3 0.7]
        FV = compute_future_values(state_df, simple_logit, xtran, 2, 3, 0.9)

        @test size(FV) == (2,2,4)
        @test all(FV[:,:,1] .== 0)
        @test all(FV[:,:,4] .== 0)
        @test all(FV[:,:,2:3] .>= 0)
    end

    @testset "Question 3b: map future values" begin
        df_long = DataFrame(bus_id=[1,1], time=[1,2])
        FV = zeros(2,2,3)
        FV[:,1,2] = [1.0,2.0]
        xtran = [0.8 0.2; 0.3 0.7]
        Xstate = [2 2]
        Zstate = [1]
        B = [0]

        fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, 2, B)
        @test length(fvt1) == 2
        @test isapprox(fvt1[1], 0.5; atol=1e-12)
        @test isapprox(fvt1[2], 0.0; atol=1e-12)
    end

    @testset "Question 3c: structural logit with offset" begin
        Random.seed!(63)
        n = 100
        df_long = DataFrame(
            Y = repeat([0,1], 50),
            Odometer = rand(n),
            Branded = repeat([0,1], inner=50)
        )
        theta_hat = estimate_structural_params(df_long, zeros(n))

        @test length(coef(theta_hat)) == 3
        @test all(isfinite, coef(theta_hat))
    end

end
