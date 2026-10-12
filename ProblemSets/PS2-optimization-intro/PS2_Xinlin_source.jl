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
    startval= rand(1)
    #randomn starting value
    result = optimize(minusf, startval, LBFGS())
    println("optimization summary:", result) # for extra data detail
    println("argmin(minizer) is", Optim.minimizer(result)[1])
    println("min(-f) is", Optim.minimum(result))
    println("max(f) is", -Optim.minimum(result))
    return result
end

#----------------------------------------------------
# question 2
#----------------------------------------------------

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
    end

function q2(df=read_data(); show_trace=true)
    df = copy(df)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    beta_hat_ols = optimize(b-> ols(b, X, y), rand(size(X,2)), LBFGS(),
    Optim.Options(g_tol=1e-6, iterations=100_000,
    show_trace=show_trace)) #setting iterations=100_000, if the function is wrong, 
                      #would not iterate forever
    println(beta_hat_ols.minimizer)

    # using GLM (already in the beginning of script)
    bols = inv(X'*X)*X'*y
    @show bols
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    println(coeftable(bols_lm))

    #standard error
    σ² = sum((y.-X*bols).^2)/(size(X,1)-size(X,2))
    vcov_bols = σ²*inv(X'*X) 
    @show [bols sqrt.(diag(vcov_bols))] # print out the coefficients
    return (result=beta_hat_ols, bols=bols, model=bols_lm, vcov=vcov_bols)
end


#----------------------------------------------------
# question 3
#----------------------------------------------------

function logit_like(beta, X, y)
    eta = X * beta
    # Negative log likelihood; the stable form avoids log(0) and overflow.
    # Equivalent to -sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p)).
    net_log_like = sum(max.(eta, 0) .- y .* eta .+ log1p.(exp.(-abs.(eta))))
    return net_log_like
end

function q3_q4(df=read_data(); show_trace=true)
    df = copy(df)
    @show describe(df)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    beta_hat_logit = optimize(b-> logit_like(b, X, y), rand(size(X,2)), LBFGS(),
    Optim.Options(g_tol=1e-6, iterations=100_000,
    show_trace=show_trace))
    println(beta_hat_logit.minimizer)

    df.white = df.race.==1
    blogit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    #glm(@fomula(Y~X), data, Binominal(),Logitlink())
    println(coeftable(blogit_glm))

    return (result=beta_hat_logit, model=blogit_glm)
end



#----------------------------------------------------
# question 5
#----------------------------------------------------
function q5(df=read_data(); show_trace=true)
    df = copy(df)
    @show describe(df)
    @show freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    @show freqtable(df, :occupation) # problem solve

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation


    startval = rand(size(X,2)*(Int(maximum(y))-1)) # random starting value
    # @show startval
    alpha_hat_logit = optimize(b -> mlogit(b,X,y), startval, LBFGS(),
    Optim.Options(g_tol=1e-5, iterations=100_000,
    show_trace=show_trace))
    println(alpha_hat_logit.minimizer)
    alpha_mat = [reshape(Optim.minimizer(alpha_hat_logit), size(X,2), Int(maximum(y))-1) zeros(size(X,2),1)]
    println("Rows: intercept, age, white, collgrad; columns: occupations 1-7 (base = 7)")
    show(stdout, "text/plain", alpha_mat)
    println()
    println("Negative log likelihood: ", Optim.minimum(alpha_hat_logit))
    println("Converged: ", Optim.converged(alpha_hat_logit))

    return (result=alpha_hat_logit, alpha_mat=alpha_mat, X=X, y=y)
end

function mlogit(alpha, X, d)
    # unslice the parameter vector
    N = size(X,1) # No. of obs
    K = size(X,2) # No. of covariances
    J = Int(maximum(d)) # No. of choice alternatives
    alpha_mat = [reshape(alpha, K, J-1) zeros(K,1)]
    # alpha_mat = reshape((alpha, size(X,2),maximum(d)-1) zeros(K,1))
    # add a column of zeros for the base category

    # make d matrix
    y = d
    d = zeros(N,J)
    for j = 1:J
        d[:,j] .=y .== j
    end
    # d = [y .== j for j in 1:J]

    # make p matrix
    # Normalize across alternatives for EACH observation (not across people).
    # Subtract each row's maximum utility for numerical stability.
    utility = X * alpha_mat
    shifted = utility .- maximum(utility; dims=2)
    logp = shifted .- log.(sum(exp.(shifted); dims=2))
    p = zeros(N, J)
    for j = 1:J 
        p[:,j] .= exp.(logp[:,j])
    end
    # p = [exp.(X * alpha_mat[:, j])./sum(exp.(X * alpha_mat[:, j]))for j in 1:J]

    # log likehood expression
    loglike = -sum(d .* logp) # use logp directly to avoid log(0)

    return loglike

end

# Read the course data; a local copy can be placed beside these three files.
function read_data()
    path = joinpath(@__DIR__, "nlsw88.csv")
    if isfile(path)
        return CSV.read(path, DataFrame)
    end
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    return CSV.read(IOBuffer(HTTP.get(url).body), DataFrame)
end
