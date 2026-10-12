using Test, JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

# Read in the source code
include("PS1_Ransom_source.jl")

#-------------------------------------------------------------------------------
# Helper: capture what a function prints to stdout.
# q2() returns nothing (as the PS requires), so the only way to check its
# internals from outside is to read what it @show's.
#-------------------------------------------------------------------------------
function capture_stdout(f)
    old_stdout = stdout
    rd, wr = redirect_stdout()
    reader = @async read(rd, String)   # read concurrently so we can't fill the pipe
    try
        f()
    finally
        redirect_stdout(old_stdout)
        close(wr)
    end
    out = fetch(reader)
    close(rd)
    return out
end

# Set the seed and generate the objects the tests below run on
Random.seed!(1234)
A, B, C, D = q1()

#-------------------------------------------------------------------------------
# q1
#-------------------------------------------------------------------------------
@testset "q1: matrix dimensions and types" begin
    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C) == (5, 7)
    @test size(D) == (10, 7)
    @test eltype(A) == Float64
    @test eltype(B) == Float64
    @test eltype(C) == Float64
    @test eltype(D) == Float64
end

@testset "q1: A and B come from the right distributions" begin
    # A ~ U[-5, 10]: hard support bounds, plus a 4-standard-error check on the mean
    @test all(-5 .<= A .<= 10)
    @test abs(mean(A) - 2.5) < 4 * (15 / sqrt(12)) / sqrt(length(A))

    # B ~ N(-2, 15): unbounded, so check moments instead (4 SE on the mean,
    # generous band on the standard deviation given only 70 draws)
    @test abs(mean(B) - (-2)) < 4 * 15 / sqrt(length(B))
    @test 0.5 * 15 < std(B) < 1.7 * 15
    @test any(B .< -5) || any(B .> 10)   # B is not accidentally A
end

@testset "q1: C is the correct submatrix" begin
    @test C[:, 1:5] == A[1:5, 1:5]
    @test C[:, 6:7] == B[1:5, end-1:end]
end

@testset "q1: D is A censored above at 0" begin
    @test all(D .<= 0)
    @test D == A .* (A .<= 0)
    @test all(D[A .> 0] .== 0)
    @test all(D[A .<= 0] .== A[A .<= 0])
    @test count(D .== 0) == count(A .> 0)
end

@testset "q1: output files exist" begin
    @test isfile("matrixpractice.jld")
    @test isfile("firstmatrix.jld")
    @test isfile("Cmatrix.csv")
    @test isfile("Dmatrix.dat")
end

@testset "q1: matrixpractice.jld round-trips E, F and G" begin
    d = load("matrixpractice.jld")
    @test Set(keys(d)) == Set(["A", "B", "C", "D", "E", "F", "G"])
    @test d["A"] == A
    @test d["B"] == B
    @test d["C"] == C
    @test d["D"] == D

    # E is B vectorized (column-major)
    E = d["E"]
    @test E == vec(B)
    @test length(E) == 70
    @test E[1:10] == B[:, 1]

    # F stacks A and B along the *first* dimension after permutedims
    F = d["F"]
    @test size(F) == (2, 10, 7)
    @test F[1, :, :] == A
    @test F[2, :, :] == B

    # G is the Kronecker product of B and C
    G = d["G"]
    @test size(G) == (size(B, 1) * size(C, 1), size(B, 2) * size(C, 2)) == (50, 49)
    @test G == kron(B, C)
    @test G[1:5, 1:7] == B[1, 1] * C
end

@testset "q1: firstmatrix.jld holds only A, B, C, D" begin
    d = load("firstmatrix.jld")
    @test Set(keys(d)) == Set(["A", "B", "C", "D"])
    @test d["A"] == A && d["B"] == B && d["C"] == C && d["D"] == D
end

@testset "q1: Cmatrix.csv round-trips to C" begin
    df = CSV.read("Cmatrix.csv", DataFrame)
    @test size(df) == size(C)
    @test names(df) == ["x$i" for i in 1:7]
    @test Matrix(df) ≈ C
end

@testset "q1: Dmatrix.dat is tab-delimited and round-trips to D" begin
    header = first(eachline("Dmatrix.dat"))
    @test occursin('\t', header)
    @test !occursin(',', header)
    df = CSV.read("Dmatrix.dat", DataFrame; delim='\t')
    @test size(df) == size(D)
    @test Matrix(df) ≈ D
end

@testset "q1: is reproducible given the seed" begin
    Random.seed!(1234)
    A2, B2, C2, D2 = q1()
    @test A2 == A
    @test B2 == B
    @test C2 == C
    @test D2 == D
end

#-------------------------------------------------------------------------------
# q2
#-------------------------------------------------------------------------------
Random.seed!(1234)
q2_out = capture_stdout(() -> q2(A, B, C, D))

@testset "q2: returns nothing" begin
    Random.seed!(1234)
    @test capture_stdout(() -> q2(A, B, C, D)) isa String
    Random.seed!(1234)
    local ret
    capture_stdout(() -> (ret = q2(A, B, C, D)))
    @test ret === nothing
end

@testset "q2: both ways of subsetting C agree" begin
    # q2 builds Cprime with a loop and Cprime2 with logical indexing; they should
    # be the same vector, and it should be exactly the elements of C in [-5, 5]
    Cprime_expected = C[(C .>= -5) .& (C .<= 5)]
    @test occursin("Cprime = " * repr(Cprime_expected), q2_out)
    @test occursin("Cprime2 = " * repr(Cprime_expected), q2_out)
    @test all(-5 .<= Cprime_expected .<= 5)
end

@testset "q2: beta matches the coefficient evolution in the PS" begin
    # The PS specifies, for t = 1...5:
    #   1, 1.25, 1.5, ...   |   log(t)   |   -sqrt(t)
    #   e^t - e^(t+1)       |   t        |   t/3
    T = 5
    β_expected = zeros(6, T)
    β_expected[1, :] = [1 + 0.25 * (t - 1) for t in 1:T]
    β_expected[2, :] = [log(t) for t in 1:T]
    β_expected[3, :] = [-sqrt(t) for t in 1:T]
    β_expected[4, :] = [exp(t) - exp(t + 1) for t in 1:T]
    β_expected[5, :] = [t for t in 1:T]
    β_expected[6, :] = [t / 3 for t in 1:T]

    @test occursin("typeof(β) = Matrix{Float64}", q2_out)
    @test occursin("β = " * repr(β_expected), q2_out)

    # row 4 is decreasing and negative -- the common sign error is e^t - e^(t-1)
    @test all(β_expected[4, :] .< 0)
    @test issorted(β_expected[4, :]; rev=true)
end

@testset "q2: Y has the dimensions implied by N, K and T" begin
    @test occursin("typeof(Y) = Matrix{Float64}", q2_out)
    @test occursin("size(Y) = (15169, 5)", q2_out)
end

@testset "q2: is reproducible given the seed" begin
    Random.seed!(1234)
    out2 = capture_stdout(() -> q2(A, B, C, D))
    @test out2 == q2_out
end
