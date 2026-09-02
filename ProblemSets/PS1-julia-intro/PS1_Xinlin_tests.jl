using Test

include(joinpath(@__DIR__, "PS1_Xinlin_source.jl"))

@testset "Question 1 tests" begin
    A, B, C, D = q1()

    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C) == (5, 7)
    @test size(D) == (10, 7)
    @test all(D[A .<= 0] .== A[A .<= 0])
    @test all(D[A .> 0] .== 0)
    @test isfile(joinpath(@__DIR__, "matrixpractice.jld"))
    @test isfile(joinpath(@__DIR__, "firstmatrix.jld"))
    @test isfile(joinpath(@__DIR__, "Cmatrix.csv"))
    @test isfile(joinpath(@__DIR__, "Dmatrix.dat"))
end

@testset "Question 2 tests" begin
    A, B, C, D = q1()
    @test isnothing(q2(A, B, C))
end

@testset "Question 3 tests" begin
    @test isnothing(q3())
    @test isfile(joinpath(@__DIR__, "nlsw88_processed.csv"))

    processed = CSV.read(joinpath(@__DIR__, "nlsw88_processed.csv"), DataFrame)
    @test nrow(processed) > 0
    @test :never_married in propertynames(processed)
    @test :collgrad in propertynames(processed)
    @test count(ismissing, processed.grade) == 2
end

@testset "matrixops tests" begin
    A = [1.0 2.0; 3.0 4.0]
    B = [5.0 6.0; 7.0 8.0]

    element_product, transpose_product, total_sum = matrixops(A, B)

    @test element_product == A .* B
    @test transpose_product == A' * B
    @test total_sum == sum(A .+ B)
    @test_throws ErrorException matrixops(zeros(2, 2), zeros(3, 2))
end

@testset "Question 4 tests" begin
    q1()
    q3()
    @test isnothing(q4())
end