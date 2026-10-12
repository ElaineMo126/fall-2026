using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

# Read in function to create state transitions for dynamic model
include("create_grids.jl")

# read in the source code
include("PS5_demo_source.jl")

# run the main source function
main()