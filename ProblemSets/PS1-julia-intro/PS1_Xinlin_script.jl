#include(joinpath(@__DIR__, "PS1_Xinlin_source.jl"))

#A, B, C, D = q1()
#q2(A, B, C)
#q3()
#q4()

# read in the source code
include("PS1_Xinlin_source.jl")

# call the function from q1
A, B, C, D = q1()

# call the function from q2
q2(A, B, C, D)

# call the function from q3
q3()
println("all done")