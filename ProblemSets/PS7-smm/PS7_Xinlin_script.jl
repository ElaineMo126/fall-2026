using Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, FreqTables
import SMM

Random.seed!(1234)
include(joinpath(@__DIR__, "PS7_Xinlin_source.jl"))
main()  # Q1-Q4 run; Q5 is fully coded but OFF by default.

# To save output: julia PS7_Xinlin_script.jl > PS7_output.txt 2>&1

# 2(c)
# Is the GMM objective function globally concave?
# No. Compare the estimates/objectives printed for the two starting values.
# The MNL log likelihood is concave; its negative, minimized by Optim, is convex.
# Same seed AND same draw order are necessary for reproducing the demo.
# Julia/package versions and stable arithmetic can still affect final digits.
