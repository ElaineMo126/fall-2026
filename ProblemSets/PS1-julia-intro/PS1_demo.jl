using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
Random.seed!(1234)

#--------------------------
# question 1, part (a)
#--------------------------
# draw 10x7 array uniform random numbers U[-5,10]
# buit-in-generator (rand())
function q1()

    A = -5 .+ 15 * rand(10, 7)
    println(A)
    # useDistributions.jl
    A = rand(Uniform(-5, 10), 10, 7)
    println(A)
    # draw normal Distributions
    B = -2 .+ 15 * rand(10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    # indexing
    C = [A[1:5, 1:5] B[1:5, end-1:end]]
    # Dummy/Bit array 
    D = A .* (A .<= 0)

    # ------------------------------------------
    # question 1, part (b)
    # ------------------------------------------
    size_A = size(A)
    size_dim1 = size(A, 1)
    size_dim2 = size(A, 2)
    len_A = length(A)
    vec_A = A[:]

    # ------------------------------------------
    # question 1, part (c)
    # ------------------------------------------
    total_len_D = length(D)
    unique_len_D = length(unique(D))

    # ------------------------------------------
    # question 1, part (d)
    # ------------------------------------------
    # Reshape B into a 70x1 vector
    E = reshape(B, 70, 1)
    # Alternative ways shown:
    E = reshape(B, (70, 1))
    E = reshape(B, length(B), 1)
    E = reshape(B, size(B, 1) * size(B, 2), 1)
    E = B[:]

    # ------------------------------------------
    # question 1, part (e)
    # ------------------------------------------
    # 3D arrays
    F = cat(A, B, dims=3)

    # ------------------------------------------
    # question 1, part (f)
    # ------------------------------------------
    # 3-D array reshape
    F = permutedims(F, (3, 1, 2))

    # ------------------------------------------
    # question 1, part (g)
    # ------------------------------------------
    # Kron
    G = kron(B, C)

    # kron(C, F) fails because F is a 3-dimensional array,
    # while kron expects vectors or matrices.

    # ------------------------------------------
    # question 1, part (h)
    # ------------------------------------------
    save(joinpath(@__DIR__, "matrixpractice.jld"), "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

    # ------------------------------------------
    # question 1, part (i)
    # ------------------------------------------
    save(joinpath(@__DIR__, "firstmatrix.jld"), "A", A, "B", B, "C", C, "D", D)

    # ------------------------------------------
    # question 1, part (j)
    # ------------------------------------------
    CSV.write(joinpath(@__DIR__, "Cmatrix.csv"), DataFrame(C, :auto))

    # ------------------------------------------
    # question 1, part (k)
    # ------------------------------------------
    #CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim='\t')
    # Equivalent pipe syntax demonstrated in the video:
    CSV.write(joinpath(@__DIR__, "Dmatrix.dat"), DataFrame(D, :auto); delim='\t')

    # ------------------------------------------
    # Return outputs
    # ------------------------------------------
    return A, B, C, D
end

# ==========================================
# Run the Function
# ==========================================
# Calling the function and unpacking the return values
A, B, C, D = q1()


function q2(A, B, C)
    #-------------------------------------
    #Question 2. part (a)
    #-------------------------------
    AB = zeros(size(A))
    for r in axes(A, 1)
        for c in axes(A, 2)
            AB[r, c] = A[r, c] * B[r, c]
        end
    end

    AB3 = [A[r, c] * B[r, c] for r in axes(A, 1), c in axes(A, 2)]
    isequal(AB, AB3)

    AB2 = A .* B
    #-------------------------------------
    #Question 2. part (b)
    #-------------------------------
    Cprime = Float64[]
    for c in axes(C, 2)
        for r in axes(C, 1)
            if C[r, c] >= -5 && C[r, c] <= 5
                push!(Cprime, C[r, c])
            end
        end
    end

    Cprime2 = C[(C .>= -5) .& (C .<= 5)]
    isequal(Cprime, Cprime2)
    @show Cprime
    @show Cprime2

    #-------------------------------------
    #Question 2. part (c)
    #-------------------------------
    N = 15_169
    k = 6
    T = 5
    X = zeros(N, k, T)
    #colum 1: intercept
    #colum 2: dummy variable
    #colum 3: continious (normal) variable
    #colum 4: normal
    #colum 5: binomial
    #colum 6: another binomial
    for i in axes(X, 1)
        X[i, 1, :] .= 1.0
        X[i, 5, :] .= rand(Binomial(20, 0.6))
        X[i, 6, :] .= rand(Binomial(20, 0.5))
        for t in axes(X, 3)
            X[i, 2, t] = rand() <= 0.75 * (6 - t) / 5
            # using max to prevent std dev of 0 when t = 1
            if t == 1
                X[i, 3, t] = 15
            else
                X[i, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)))
            end
            X[i, 4, t] = rand(Normal(π * (6 - t)/3, 1 / exp(1)))
        end
    end

    #-------------------------------------
    #Question 2. part (d)
    #-------------------------------
    #comprehension
    β = zeros(k, T)
    β[1, :] = [1 + 0.25 * (t - 1) for t in 1:T]
    β[2, :] = [log(t) for t in 1:T]
    β[3, :] = [-sqrt(t) for t in 1:T]
    β[4, :] = [exp(t) - exp(t + 1) for t in 1:T]
    β[5, :] = [t for t in 1:T]
    β[6, :] = [t / 3 for t in 1:T]
    @show typeof(β)
    @show β

    #-------------------------------------
    #Question 2. part (e)
    #-------------------------------
    Y = zeros(N, T)
    for t in 1:T
        Y[:, t] = X[:, :, t] * β[:, t] .+ rand(Normal(0, 0.36 * t), N)
    end
    @show typeof(Y)
    @show size(Y)
    #@show Y[1]

    return nothing
end

# ==========================================
# Run Question 2
# ==========================================
q2(A, B, C)

# ==========================================
# Question 3: Reading data and summary statistics
# ==========================================
function q3()
    # ------------------------------------------
    # Question 3, part (a)
    # ------------------------------------------
    # Blank cells in the CSV file are read as missing values.
    input_path = joinpath(@__DIR__, "nlsw88.csv")
    output_path = joinpath(@__DIR__, "nlsw88_processed.csv")

    nlsw88 = CSV.read(input_path, DataFrame; normalizenames=true)

    CSV.write(output_path, nlsw88)

    # ------------------------------------------
    # Question 3, part (b)
    # ------------------------------------------
    never_married_pct = 100 * mean(skipmissing(nlsw88.never_married))

    college_grad_pct = 100 * mean(skipmissing(nlsw88.collgrad))

    println("Percentage never married: ", round(never_married_pct, digits=2), "%")

    println("Percentage college graduates: ", round(college_grad_pct, digits=2), "%")

    # ------------------------------------------
    # Question 3, part (c)
    # ------------------------------------------
    race_counts = freqtable(nlsw88.race)

    race_percent = 100 .* race_counts ./ sum(race_counts)

    println("\nPercentage in each race category:")
    println(race_percent)

    # ------------------------------------------
    # Question 3, part (d)
    # ------------------------------------------
    summarystats = describe(nlsw88, :mean, :median, :std, :min, :max, :nunique)

    missing_grade = count(ismissing, nlsw88.grade)

    println("\nSummary statistics:")

    show(summarystats; allrows=true, allcols=true)

    println("\n\nNumber of missing grade observations: ", missing_grade)

    # ------------------------------------------
    # Question 3, part (e)
    # ------------------------------------------
    industry_occupation =
        freqtable(
            nlsw88.industry,
            nlsw88.occupation
        )

    println("\nJoint distribution of industry and occupation:")

    println(industry_occupation)

    # ------------------------------------------
    # Question 3, part (f)
    # ------------------------------------------
    wage_data = select(
        nlsw88,
        :industry,
        :occupation,
        :wage
    )

    dropmissing!(wage_data)

    mean_wage = combine(groupby(wage_data, [:industry, :occupation]), :wage => mean => :mean_wage)

    sort!(mean_wage, [:industry, :occupation])

    println("\nMean wage by industry and occupation:")
    println(mean_wage)

    return nothing
end

# ==========================================
# Question 4: Practice with functions
# ==========================================

"""
    matrixops(A, B)

Takes two arrays A and B of the same size.

Returns:

1. The element-by-element product of A and B.
2. The matrix product A'B.
3. The sum of all elements of A + B.
"""
function matrixops(A, B)
    # ------------------------------------------
    # Question 4, part (e)
    # ------------------------------------------
    if size(A) != size(B)
        error("inputs must have the same size")
    end

    # Element-by-element product
    element_product = A .* B

    # Matrix product A'B
    transpose_product = A' * B

    # Sum of every element in A + B
    total_sum = sum(A .+ B)

    return element_product, transpose_product, total_sum
end


function q4()
    # ------------------------------------------
    # Question 4, part (a)
    # ------------------------------------------
    matrices = load(joinpath(@__DIR__, "firstmatrix.jld"))

    A = matrices["A"]
    B = matrices["B"]
    C = matrices["C"]
    D = matrices["D"]

    # ------------------------------------------
    # Question 4, part (d)
    # ------------------------------------------
    AB_element, AB_transpose, AB_sum = matrixops(A, B)

    println("\nmatrixops(A, B):")

    println("Size of element-by-element product: ", size(AB_element))

    println("Size of A'B: ", size(AB_transpose))

    println("Sum of all elements of A + B: ", AB_sum)

    # ------------------------------------------
    # Question 4, part (f)
    # ------------------------------------------
    # C is 5×7, while D is 10×7.
    # Therefore, matrixops should produce an error.
    try
        matrixops(C, D)
    catch err
        println("\nmatrixops(C, D) error: ", err)
    end

    # ------------------------------------------
    # Question 4, part (g)
    # ------------------------------------------
    nlsw88 = CSV.read(joinpath(@__DIR__, "nlsw88_processed.csv"), DataFrame)

    earnings = select(nlsw88, :ttl_exp, :wage)

    dropmissing!(earnings)

    ttl_exp = Vector{Float64}(earnings.ttl_exp)

    wage = Vector{Float64}(earnings.wage)

    tw_element, tw_product, tw_sum = matrixops(ttl_exp, wage)

    println("\nmatrixops(ttl_exp, wage):")

    println("Length of element-by-element product: ", length(tw_element))

    println("ttl_exp' * wage: ", tw_product)

    println("Sum of ttl_exp + wage: ", tw_sum)

    return nothing
end

q3()
q4()