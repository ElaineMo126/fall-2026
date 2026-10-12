using Test
# allwrap() from executing until the integration test explicitly calls it.
include(joinpath(@__DIR__, replace(basename(@__FILE__), "_tests.jl" => "_script.jl")))

@testset "PS3 starter-based code" begin
    @testset "load_data" begin
        df, X, Z, y = load_data(joinpath(@__DIR__, "nlsw88w.csv"))
        @test size(X) == (2237, 3)
        @test size(Z) == (2237, 8)
        @test X == [df.age df.white df.collgrad]
        @test Z[:,8] == df.elnwage8
        @test y == df.occupation
        @test sort(unique(y)) == collect(1:8)
    end
    rng = MersenneTwister(6343)
    X = randn(rng, 8, 3)
    Z = randn(rng, 8, 8)
    y = collect(1:8)
    nests = [[1,2,3], [4,5,6,7]]
    rowshift = reshape(collect(1.0:8.0), 8, 1)
    @testset "mlogit_with_Z" begin
        @test mlogit_with_Z(zeros(22), X, Z, y) ≈ 8log(8)
        t = 0.2randn(rng, 22)
        B = hcat(reshape(t[1:21], 3, 7), zeros(3))
        V = X*B .+ t[end].*(Z .- Z[:,8:8])
        P = exp.(V) ./ sum(exp.(V), dims=2)
        @test mlogit_with_Z(t, X, Z, y) ≈ -sum(log(P[i,y[i]]) for i in 1:8)
        # Single-observation likelihoods recover every individual probability.
        fittedP = [exp(-mlogit_with_Z(t, X[i:i,:], Z[i:i,:], [j])) for i in 1:8, j in 1:8]
        @test fittedP ≈ P
        @test vec(sum(fittedP, dims=2)) ≈ ones(8)
        @test mlogit_with_Z(t, X, Z .+ rowshift, y) ≈ mlogit_with_Z(t, X, Z, y)
        @test isfinite(mlogit_with_Z(t, X, Z, ones(Int,8)))
        # Extreme utilities may give infinite NLL, but must not produce NaN.
        @test !isnan(mlogit_with_Z(10000t, X, Z, y))
        # Q2: change in log odds is gamma times the change in Z_j.
        Znew = copy(Z); Znew[1,2] += 0.01
        logodds = -mlogit_with_Z(t, X[1:1,:], Z[1:1,:], [2]) +
                   mlogit_with_Z(t, X[1:1,:], Z[1:1,:], [8])
        newlogodds = -mlogit_with_Z(t, X[1:1,:], Znew[1:1,:], [2]) +
                      mlogit_with_Z(t, X[1:1,:], Znew[1:1,:], [8])
        @test newlogodds-logodds ≈ t[end]*0.01
    end
    @testset "nested_logit_with_Z" begin
        t = [0.1randn(rng,6); 0.6; 0.8; 0.3]
        # Independent direct evaluation of the textbook formula.
        P = zeros(8,8)
        for i in 1:8
            numerators = ones(8)
            for g in 1:2
                js = nests[g]; lam = t[6+g]
                v = [dot(X[i,:], t[(g-1)*3+1:g*3]) + t[end]*(Z[i,j]-Z[i,8]) for j in js]
                e = exp.(v./lam)
                numerators[js] = e .* sum(e)^(lam-1)
            end
            P[i,:] = numerators ./ sum(numerators)
        end
        @test nested_logit_with_Z(t,X,Z,y,nests) ≈ -sum(log(P[i,y[i]]) for i in 1:8)
        fittedP = [exp(-nested_logit_with_Z(t,X[i:i,:],Z[i:i,:],[j],nests)) for i in 1:8, j in 1:8]
        @test fittedP ≈ P
        @test vec(sum(fittedP,dims=2)) ≈ ones(8)
        @test all(fittedP .> 0)
        @test nested_logit_with_Z(t,X,Z .+ rowshift,y,nests) ≈ nested_logit_with_Z(t,X,Z,y,nests)
        # Zero utilities imply nest weights 3^lambda_WC, 4^lambda_BC, 1.
        tz = [zeros(6); 0.6; 0.8; 0.0]
        D = 1+3^0.6+4^0.8
        expected = [fill(3^(0.6-1)/D,3); fill(4^(0.8-1)/D,4); 1/D]
        @test [exp(-nested_logit_with_Z(tz,X[1:1,:],Z[1:1,:],[j],nests)) for j in 1:8] ≈ expected
        # lambda=1 reduces to MNL with equal beta within each nest.
        tunit = [t[1:6]; 1.0; 1.0; t[end]]
        tmnl = [repeat(t[1:3],3); repeat(t[4:6],4); t[end]]
        @test nested_logit_with_Z(tunit,X,Z,y,nests) ≈ mlogit_with_Z(tmnl,X,Z,y)
        @test nested_logit_with_Z([zeros(6);0.0;1.0;0.1],X,Z,y,nests) == Inf
        @test nested_logit_with_Z([zeros(6);-0.5;1.0;0.1],X,Z,y,nests) == Inf
        # Near-zero scales with proportionately small gamma remain computable.
        @test isfinite(nested_logit_with_Z([t[1:6];1e-8;2e-8;1e-8],X,Z,y,nests))
    end
    @testset "optimize_mlogit, optimize_nested_logit, allwrap" begin
        # Run the actual wrapper, including the original random starts and LBFGS.
        # Store the long optimizer trace in a temporary file during the test.
        fits, report = mktemp() do path, io
            fits = redirect_stdout(io) do
                allwrap()
            end
            flush(io)
            return fits, read(path,String)
        end
        mnl, nl = fits
        _, Xfull, Zfull, yfull = load_data(joinpath(@__DIR__,"nlsw88w.csv"))
        @test length(mnl) == 22
        @test length(nl) == 9
        @test all(isfinite,mnl)
        @test all(isfinite,nl)
        @test all(nl[7:8] .> 0)
        @test mlogit_with_Z(mnl,Xfull,Zfull,yfull) < length(yfull)*log(8)
        @test nested_logit_with_Z(nl,Xfull,Zfull,yfull,nests) < length(yfull)*log(8)
        # Independent numerical benchmark for the MNL optimum.
        @test mlogit_with_Z(mnl,Xfull,Zfull,yfull) ≈ 3738.11904555 atol=0.001
        @test occursin("MULTINOMIAL LOGIT RESULTS",report)
        @test occursin("NESTED LOGIT RESULTS",report)
        @test occursin("A one-unit increase in Z_j",report)
        @test length(collect(eachmatch(r"Optimizer converged: true",report))) == 2
        # Numerical convergence status is explicitly reported in the output;
        # formula tests passing does not establish an interior maximum for NL.
    end
end
