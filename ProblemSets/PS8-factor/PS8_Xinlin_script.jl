using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, FreqTables, ForwardDiff, LineSearches

include(joinpath(@__DIR__, "PS8_Xinlin_source.jl"))

main()

# Q2: are these corelations high? what might this imply?
# might not be a good idea because they could cause multicollinearity
# even if they resolve some of the omitted variable bias 
