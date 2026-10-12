function q1()
    #----------------------------------------------------
    # question 1
    #----------------------------------------------------
    # identically parameterize the function as:
    #function f(x)
    #    f(x) =-x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    #end
    f(x) =-x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    minusf(x)= x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2 # minus f(x)
    startval= [-7.0] # rand(1) #
    #randomn starting value
    result = optimize(minusf, startval, BFGS())
    println("optimization summary:", result) # for extra data detail
    println("argmin(minizer) is", Optim.minimizer(result)[1])
    println("min(minizer) is", Optim.minimizer(result))
    println("max(minizer) is", Optim.minimizer(result))
end

#----------------------------------------------------
# question 2
#----------------------------------------------------

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
    end

function q2()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/
    master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    beta_hat_ols = optimize(b-> ols(b, X, y), rand(size(X,2)), LBFGS(),
    Optim.Options(g_tol=1e-6, iterations=100_000,
    show_trace=true)) #setting iterations=100_000, if the function is wrong, 
                      #would not iterate forever
    println(beta_hat_ols.minimizer)

    # using GLM (already in the beginning of script)
    bols = inv(X'*X)*X'*y
    @show bols
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    @show bols_lm

    #standard error
    σ² = sum((y.-X*bols).^2)/(size(X,1)-size(X,2))
    vxov_bols = σ²*inv(X'*X) 
    @show [bols sqrt.(diag(vcov_bols))] # print out the coefficients
    return nothing
end


#----------------------------------------------------
# question 3
#----------------------------------------------------

function logit_like(beta, X, y)
    p = exp.(X*beta)./(1 .+exp.(X*beta))
    net_log_like = sum(y.*log.(p).+(1,-y).*log.(1,-p))
    return net_log_like
end

function q3_q4()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/
    master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    @show describe(df)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    beta_hat_logit = optimize(b-> logit_like(b, X, y), rand(size(X,2)), LBFGS(),
    Optim.Options(g_tol=1e-6, iterations=100_000,
    show_trace=true))
    println(beta_hat_logit.minimizer)

    df.white = df.race.==1
    blogit_glm = glm(@formula(married ~ age + white + collgrad), df, Binominal(), Logitlink())
    #glm(@fomula(Y~X), data, Binominal(),Logitlink())
    @show blogit_glm

    return nothing
end



#----------------------------------------------------
# question 5
#----------------------------------------------------
function q5()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/
    master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    @show describe(df)
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solve

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation


    startval = rand(size(X,2)*(maximum(y)-1)) # random starting value
    # @show startval
    alpha_hat_logit = optimize(b -> mlogit(b,X,y),rand(size(X,2)), LBFGS(),
    Optim.Options(g_tol=1e-6, iterations=100_000,
    show_trace=true))
    print(alpha_hat_logit.minimizer)

    return nothing
end

function mlogit(alpha, X, d)
    # unslice the parameter vector
    N = size(X,1) # No. of obs
    K = size(X,2) # No. of covariances
    J = maximum(d) # No. of choice alternatives
    alpha_mat = [reshape((alpha, K, maximumJ-1) zeros(K,1))]
    # alpha_mat = reshape((alpha, size(X,2),maximum(d)-1) zeros(K,1))
    # add a column of zeros for the base category

    # make d matrix
    d = zeros(N,J)
    for j = 1:J
        d[:,j] .=y .== j
    end
    # d = [y .== j for j in 1:J]

    # make p matrix
    p = zeros(N, J)
    for j = 1:J 
        p[:,j] .= exp.(X * alpha_mat[:, j])./sum(exp.(X * alpha_mat[:, j]))
    end
    # p = [exp.(X * alpha_mat[:, j])./sum(exp.(X * alpha_mat[:, j]))for j in 1:J]

    # log likehood expression
    loglike = -sum(d .* log.(p))

    return loglike

end