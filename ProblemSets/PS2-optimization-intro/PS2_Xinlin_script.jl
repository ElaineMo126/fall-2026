using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

include("PS2_Xinlin_source.jl")

# Question 6: wrap the calculations in a function and call it below.
function run_questions()
    Random.seed!(1234)
    df = read_data()
    q1()
    q2(df)
    q3_q4(df)
    q5(df)
    println("Ran successfully")
    return nothing
end

# Save printed estimates and optimizer traces for submission with the code.
function main()
    open(joinpath(@__DIR__, "PS2_Xinlin_output.txt"), "w") do io
        redirect_stdout(io) do
            run_questions()
        end
    end
    println("Ran successfully. Results saved to PS2_Xinlin_output.txt")
    return nothing
end

main()
