
function q1()
    #-------------------------------------------------------------------------------
    # Question 1, part (a)
    #-------------------------------------------------------------------------------
    # Draw uniform random numbers
    A = -5 .+ 15*rand(10,7)
    A = rand(Uniform(-5, 10), 10, 7)

    # Draw normal random numbers
    B = -2 .+ 15*randn(10,7)
    B = rand(Normal(-2, 15), 10, 7)

    # Indexing
    C = [A[1:5, 1:5] B[1:5, end-1:end]]

    # Bit Array / dummy variable
    D = A.*(A .<= 0)

    #-------------------------------------------------------------------------------
    # Question 1, part (b)
    #-------------------------------------------------------------------------------
    size(A)
    size(A, 1) * size(A, 2)
    length(A)
    size(A[:])


    #-------------------------------------------------------------------------------
    # Question 1, part (c)
    #-------------------------------------------------------------------------------
    length(D)
    length(unique(D))

    #-------------------------------------------------------------------------------
    # Question 1, part (d)
    #-------------------------------------------------------------------------------
    E = reshape(B, 70, 1)
    E = reshape(B, (70, 1))
    E = reshape(B, length(B), 1)
    E = reshape(B, size(B, 1) * size(B, 2), 1)
    E = B[:]

    #-------------------------------------------------------------------------------
    # Question 1, part (e)
    #-------------------------------------------------------------------------------
    F = cat(A, B; dims=3)



    #-------------------------------------------------------------------------------
    # Question 1, part (f)
    #-------------------------------------------------------------------------------
    F = permutedims(F, (3, 1, 2))




    #-------------------------------------------------------------------------------
    # Question 1, part (g)
    #-------------------------------------------------------------------------------
    G = kron(B, C)
    # G = kron(C, F) # this doesn't work


    #-------------------------------------------------------------------------------
    # Question 1, part (h)
    #-------------------------------------------------------------------------------
    #Save the matrices A, B, C, D, E, F and G as a .jld file named matrixpractice.
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)


    #-------------------------------------------------------------------------------
    # Question 1, part (i)
    #-------------------------------------------------------------------------------
    #Save only the matrices A, B, C, and D as a .jld file called firstmatrix
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)



    #-------------------------------------------------------------------------------
    # Question 1, part (j)
    #-------------------------------------------------------------------------------
    # Export C as a .csv file called Cmatrix. You will first need to transform C into a DataFrame.
    # CSV.write(filename, data)
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))




    #-------------------------------------------------------------------------------
    # Question 1, part (k)
    #-------------------------------------------------------------------------------
    #Export D as a tab-delimited .dat file called Dmatrix. You will first need to transform D into a DataFrame.
    df_D = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", df_D, delim='\t')

    DataFrame(D, :auto) |> x -> CSV.write("Dmatrix.dat", x, delim='\t')

    return A, B, C, D
end


function q2(A, B, C, D)
    #-------------------------------------------------------------------------------
    # Question 2, part (a)
    #-------------------------------------------------------------------------------
    AB = zeros(size(A))
    for r in axes(A,1)
        for c in axes(A,2)
            AB[r,c] = A[r,c] * B[r,c]
        end
    end
    
    AB3 = [x for r in axes(A,1), c in axes(A,2) for x in [A[r,c] * B[r,c]]]
    isequal(AB, AB3)
    
    AB2 = A .* B

    #-------------------------------------------------------------------------------
    # Question 2, part (b)
    #-------------------------------------------------------------------------------
    # find indices of C betweeen -5 and 5
    Cprime = Float64[]
    for c in axes(C, 2)
        for r in axes(C, 1)
            if C[r,c] >= -5 && C[r,c] <= 5
                push!(Cprime, C[r,c])
            end
        end
    end
    
    Cprime2 = C[(C.>=-5) .& (C.<=5)]
    isequal(Cprime, Cprime2)
    @show Cprime
    @show Cprime2
    
    #-------------------------------------------------------------------------------
    # Question 2, part (c)
    #-------------------------------------------------------------------------------
    N = 15_169
    K = 6
    T = 5
    X = zeros(N, K, T)
    #X[:, 1, :] .= 1.0
    
    # column 1: intercept
    # column 2: dummy variable
    # column 3: continuous (normal) variable
    # column 4: normal
    # column 5: binomial
    # column 6: another binomial
    for i in axes(X, 1)
        X[i, 1, :] .= 1.0
        X[i, 5, :] .= rand(Binomial(20, 0.6))
        X[i, 6, :] .= rand(Binomial(20, 0.5))
        for t in axes(X, 3)
            X[i, 2, t] = rand() <= .75*(6-t)/5
            X[i, 3, t] = rand(Normal(15 + t - 1, 5*(t-1)))
            X[i, 4, t] = rand(Normal(π*(6-t), 1/exp(1)))
        end
    end
    
    
    #-------------------------------------------------------------------------------
    # Question 2, part (d)
    #-------------------------------------------------------------------------------
    # comprehensions
    β = zeros(K, T)
    β[1, :] = [1 + 0.25*(t-1) for t in 1:T]
    β[2, :] = [log(t) for t in 1:T]
    β[3, :] = [-sqrt(t) for t in 1:T]
    β[4, :] = [exp(t) - exp(t+1) for t in 1:T]
    β[5, :] = [t for t in 1:T]
    β[6, :] = [t/3 for t in 1:T]
    @show typeof(β)
    @show β

    #-------------------------------------------------------------------------------
    # Question 2, part (e)
    #-------------------------------------------------------------------------------
    # Y = X*β + ε ... can't do it here because we have X as 3-D array
    # could do a loop (bleh)
    # or a comprehension
    Y = zeros(N, T)
    for t in 1:T
        Y[:,t] = X[:, :, t]*β[:,t] .+ rand(Normal(0, 0.36), N)
    end
    @show typeof(Y)
    @show size(Y)
    #@show Y[1]
    return nothing
end

function q3()
    #-------------------------------------------------------------------------------
    # part (a)
    #-------------------------------------------------------------------------------
    df = DataFrame(CSV.File("nlsw88.csv"))
    @show df[1:5, :]    # list in 1/5
    @show typeof(df[:,:grade]) # describe grade
    
    #-------------------------------------------------------------------------------
    # part (b)
    #-------------------------------------------------------------------------------
    @show mean(df[:,:never_married])  # sum never_married
    @show mean(df[:,:collgrad])       # sum collgrad
    
    #-------------------------------------------------------------------------------
    # part (c)
    #-------------------------------------------------------------------------------
    @show freqtable(df[:,:race])   # tab race
    
    #-------------------------------------------------------------------------------
    # part (d)
    #-------------------------------------------------------------------------------
    vars = names(df)  # storing the column names in a new vector called "vars"
    summarystats = describe(df) # no stata equivalent, but sort of like summarize
    @show summarystats
    
    #-------------------------------------------------------------------------------
    # part (e)
    #-------------------------------------------------------------------------------
    # cross tab industry and occupation
    @show freqtable(df[:,:industry], df[:,:occupation]) # tab industry occupation
    return nothing
end
