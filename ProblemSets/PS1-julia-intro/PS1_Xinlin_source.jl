using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

# ==================================================
# Question 1
# ==================================================
function q1()
    Random.seed!(1234)

    # 1(a)
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    C = [A[1:5, 1:5] B[1:5, end-1:end]]
    D = ifelse.(A .<= 0, A, 0.0)

    # 1(b)
    println("Number of elements in A: ", length(A))

    # 1(c)
    println("Number of unique elements in D: ", length(unique(D)))

    # 1(d)
    E = B[:]

    # 1(e)-(f)
    F = cat(A, B; dims=3)
    F = permutedims(F, (3, 1, 2))

    # 1(g)
    G = kron(B, C)

    # C ⊗ F produces an error because F is a three-dimensional array.
    try
        kron(C, F)
    catch err
        println("C ⊗ F error: ", err)
    end

    # 1(h)
    save(joinpath(@__DIR__, "matrixpractice.jld"), "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

    # 1(i)
    save(joinpath(@__DIR__, "firstmatrix.jld"), "A", A, "B", B, "C", C, "D", D)

    # 1(j)
    CSV.write(joinpath(@__DIR__, "Cmatrix.csv"), DataFrame(C, :auto))

    # 1(k)
    CSV.write(joinpath(@__DIR__, "Dmatrix.dat"), DataFrame(D, :auto); delim='\t')

    return A, B, C, D
end


# ==================================================
# Question 2
# ==================================================
function q2(A, B, C)
    # 2(a): loop method
    AB = zeros(size(A))

    for r in axes(A, 1), c in axes(A, 2)
        AB[r, c] = A[r, c] * B[r, c]
    end

    # Without a loop
    AB2 = A .* B
    println("AB equals AB2: ", AB == AB2)

    # 2(b): loop method
    Cprime = Float64[]

    for c in axes(C, 2), r in axes(C, 1)
        if -5 <= C[r, c] <= 5
            push!(Cprime, C[r, c])
        end
    end

    # Without a loop
    Cprime2 = C[(C .>= -5) .& (C .<= 5)]
    println("Cprime equals Cprime2: ", Cprime == Cprime2)

    # 2(c)
    N, K, T = 15_169, 6, 5
    X = zeros(N, K, T)

    for i in axes(X, 1)
        X[i, 1, :] .= 1.0
        X[i, 5, :] .= rand(Binomial(20, 0.6))
        X[i, 6, :] .= rand(Binomial(20, 0.5))

        for t in axes(X, 3)
            X[i, 2, t] = rand() <= 0.75 * (6 - t) / 5
            X[i, 3, t] = t == 1 ? 15.0 : rand(Normal(15 + t - 1, 5 * (t - 1)))
            X[i, 4, t] = rand(Normal(π * (6 - t) / 3, 1 / exp(1)))
        end
    end

    # 2(d)
    β = zeros(K, T)
    β[1, :] = [1 + 0.25 * (t - 1) for t in 1:T]
    β[2, :] = [log(t) for t in 1:T]
    β[3, :] = [-sqrt(t) for t in 1:T]
    β[4, :] = [exp(t) - exp(t + 1) for t in 1:T]
    β[5, :] = [t for t in 1:T]
    β[6, :] = [t / 3 for t in 1:T]

    # 2(e)
    Y = hcat([X[:, :, t] * β[:, t] .+ rand(Normal(0, 0.36), N) for t in 1:T]...)

    println("Size of X: ", size(X))
    println("Size of β: ", size(β))
    println("Size of Y: ", size(Y))

    return nothing
end


# ==================================================
# Question 3
# ==================================================
function q3()
    # 3(a)
    input_path = joinpath(@__DIR__, "nlsw88.csv")
    output_path = joinpath(@__DIR__, "nlsw88_processed.csv")
    nlsw88 = CSV.read(input_path, DataFrame; normalizenames=true)
    CSV.write(output_path, nlsw88)

    # 3(b)
    never_married_pct = 100 * mean(skipmissing(nlsw88.never_married))
    college_grad_pct = 100 * mean(skipmissing(nlsw88.collgrad))
    println("Percentage never married: ", round(never_married_pct; digits=2), "%")
    println("Percentage college graduates: ", round(college_grad_pct; digits=2), "%")

    # 3(c)
    race_counts = freqtable(nlsw88.race)
    race_percent = 100 .* race_counts ./ sum(race_counts)
    println("\nPercentage in each race category:")
    println(race_percent)

    # 3(d)
    summarystats = describe(nlsw88, :mean, :median, :std, :min, :max, :nunique)
    missing_grade = count(ismissing, nlsw88.grade)
    println("\nSummary statistics:")
    println(summarystats)
    println("Number of missing grade observations: ", missing_grade)

    # 3(e)
    industry_occupation = freqtable(nlsw88.industry, nlsw88.occupation)
    println("\nJoint distribution of industry and occupation:")
    println(industry_occupation)

    # 3(f)
    wage_data = dropmissing(select(nlsw88, :industry, :occupation, :wage))
    mean_wage = combine(groupby(wage_data, [:industry, :occupation]), :wage => mean => :mean_wage)
    sort!(mean_wage, [:industry, :occupation])
    println("\nMean wage by industry and occupation:")
    println(mean_wage)

    return nothing
end


# ==================================================
# Question 4
# ==================================================

"""
    matrixops(A, B)

Take two arrays of the same size and return:

1. The element-by-element product of A and B.
2. The matrix product A'B.
3. The sum of all elements of A + B.
"""
function matrixops(A, B)
    size(A) == size(B) || error("inputs must have the same size")
    return A .* B, A' * B, sum(A .+ B)
end


function q4()
    # 4(a)
    matrices = load(joinpath(@__DIR__, "firstmatrix.jld"))
    A, B, C, D = matrices["A"], matrices["B"], matrices["C"], matrices["D"]

    # 4(d)
    AB_element, AB_transpose, AB_sum = matrixops(A, B)
    println("\nmatrixops(A, B):")
    println("Element-by-element product size: ", size(AB_element))
    println("A'B size: ", size(AB_transpose))
    println("Sum of A + B: ", AB_sum)

    # 4(f): C and D have different sizes
    try
        matrixops(C, D)
    catch err
        println("matrixops(C, D) error: ", err)
    end

    # 4(g)
    nlsw88 = CSV.read(joinpath(@__DIR__, "nlsw88_processed.csv"), DataFrame)
    earnings = dropmissing(select(nlsw88, :ttl_exp, :wage))
    ttl_exp = Vector{Float64}(earnings.ttl_exp)
    wage = Vector{Float64}(earnings.wage)

    tw_element, tw_product, tw_sum = matrixops(ttl_exp, wage)
    println("\nmatrixops(ttl_exp, wage):")
    println("Element-by-element product length: ", length(tw_element))
    println("ttl_exp' * wage: ", tw_product)
    println("Sum of ttl_exp + wage: ", tw_sum)

    return nothing
end