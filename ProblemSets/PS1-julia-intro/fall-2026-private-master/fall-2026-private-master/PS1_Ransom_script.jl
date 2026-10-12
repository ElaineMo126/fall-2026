using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

# Set the seed
Random.seed!(1234)

# Read in the source code
include("PS1_Ransom_source.jl")

# call the function from q1
A, B, C, D = q1()

# call the function from q2
q2(A, B, C, D)

# call the function from q3
q3()
println("all done")
