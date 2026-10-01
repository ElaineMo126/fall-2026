using Test
include(joinpath(@__DIR__, replace(basename(@__FILE__), "Xinlin_tests.jl" => "Xinlin_script.jl")))

@testset "PS4 starter-based code" begin
    @testset "load_data" begin
        df, X, Z, y = load_data()
        @test size(X) == (28365,3)
        @test size(Z) == (28365,8)
        @test length(unique(df.idcode)) == 4698
        @test X == [df.age df.white df.collgrad]
        @test Z[:,8] == df.elnwage8
        @test y == df.occ_code
        @test all(isfinite,X) && all(isfinite,Z)
    end
    @testset "lgwt and quadrature practice" begin
        nodes, weights = lgwt(7,-1,1)
        @test all(weights .> 0)
        @test all(abs.(nodes) .< 1)
        @test sum(weights) ≈ 2
        # Seven-point Gauss-Legendre integrates polynomials through degree 13.
        for k in 0:13
            exact = iseven(k) ? 2/(k+1) : 0.0
            @test sum(weights .* nodes.^k) ≈ exact atol=1e-12
        end
        q = practice_quadrature()
        @test abs(q.density-1) < 0.005
        @test abs(q.expectation) < 1e-12
        v = variance_quadrature()
        # Independent reference values for the specified 7- and 10-node rules.
        @test v.variance7 ≈ 3.265514281891978 atol=1e-10
        @test v.variance10 ≈ 4.038977384853664 atol=1e-10
        @test abs(v.variance10-4) < abs(v.variance7-4)
    end
    @testset "practice_monte_carlo and mc_integrate" begin
        mc = practice_monte_carlo()
        @test sort(collect(keys(mc.results))) == [1000,1000000]
        @test mc.mc_integrate(x -> 1.0,-3,5,100) == 8.0
        @test mc.mc_integrate(x -> 0.0,-3,5,100) == 0.0
        # Loose stochastic tolerances; not an assertion that every draw improves.
        r = mc.results[1000000]
        @test abs(r.variance-4) < 0.04
        @test abs(r.expectation) < 0.02
        @test abs(r.density-1) < 0.01
    end
    rng = MersenneTwister(42)
    X = randn(rng,8,3)
    Z = randn(rng,8,8)
    y = collect(1:8)
    theta = 0.1randn(rng,22)
    @testset "MNL likelihood, gradient, and Hessian" begin
        @test mlogit_with_Z(zeros(22),X,Z,y) ≈ 8log(8)
        B = hcat(reshape(theta[1:21],3,7),zeros(3))
        V = X*B .+ theta[end].*(Z .- Z[:,8:8])
        P = exp.(V)./sum(exp.(V),dims=2)
        @test mlogit_with_Z(theta,X,Z,y) ≈ -sum(log(P[i,y[i]]) for i in 1:8)
        impliedP = [exp(-mlogit_with_Z(theta,X[i:i,:],Z[i:i,:],[j])) for i in 1:8,j in 1:8]
        @test impliedP ≈ P
        @test vec(sum(impliedP,dims=2)) ≈ ones(8)
        @test mlogit_with_Z(theta,X,Z .+ collect(1:8),y) ≈ mlogit_with_Z(theta,X,Z,y)
        @test isfinite(mlogit_with_Z(10000theta,X,Z,y))
        Y = Matrix{Float64}(I,8,8)
        residual = P-Y
        analytic_g = [vec(X'*residual[:,1:7]); sum(residual.*(Z .- Z[:,8:8]))]
        g = ForwardDiff.gradient(t -> mlogit_with_Z(t,X,Z,y),theta)
        @test g ≈ analytic_g atol=1e-10
        H = ForwardDiff.hessian(t -> mlogit_with_Z(t,X,Z,y),theta)
        @test H ≈ H' atol=1e-10
        @test minimum(eigvals(Symmetric(H))) > -1e-9
        # Verify Hessian against centered differences of the gradient.
        h = 1e-5
        Hfd = zeros(22,22)
        for k in 1:22
            tp=copy(theta); tm=copy(theta); tp[k]+=h; tm[k]-=h
            gp=ForwardDiff.gradient(t -> mlogit_with_Z(t,X,Z,y),tp)
            gm=ForwardDiff.gradient(t -> mlogit_with_Z(t,X,Z,y),tm)
            Hfd[:,k]=(gp-gm)/(2h)
        end
        @test H ≈ Hfd atol=1e-8
    end
    @testset "Mixed likelihoods on tiny data only" begin
        t = [theta;0.4]  # theta's last element becomes mu_gamma.
        nodes,weights = lgwt(7,-4,4)
        mass=sum(weights.*pdf.(Normal(),nodes))
        zero_sigma=[theta;0.0]
        # With raw quadrature weights, the normal-mass approximation remains.
        @test mixed_logit_quad(zero_sigma,X,Z,y,nodes,weights) ≈ mlogit_with_Z(theta,X,Z,y)-8log(mass)
        @test mixed_logit_mc(zero_sigma,X,Z,y,40) ≈ mlogit_with_Z(theta,X,Z,y)
        # Independent quadrature summation of observed-choice probabilities.
        Pobs=zeros(8)
        for r in eachindex(nodes)
            gamma=theta[end]+t[end]*nodes[r]
            B=hcat(reshape(theta[1:21],3,7),zeros(3))
            V=X*B .+ gamma.*(Z .- Z[:,8:8])
            P=exp.(V)./sum(exp.(V),dims=2)
            Pobs .+= [P[i,y[i]] for i in 1:8].*weights[r].*pdf(Normal(),nodes[r])
        end
        @test mixed_logit_quad(t,X,Z,y,nodes,weights) ≈ -sum(log.(Pobs))
        # Repeated evaluations must use common simulation draws.
        @test mixed_logit_mc(t,X,Z,y,40) == mixed_logit_mc(t,X,Z,y,40)
        # Nonzero sigma: independently reconstruct the normal-draw MC integral.
        simulation_rng=MersenneTwister(6343)
        mcP=zeros(8)
        for d in 1:40
            gamma=theta[end]+t[end]*rand(simulation_rng,Normal())
            for i in 1:8
                mcP[i]+=exp(-mlogit_with_Z([theta[1:21];gamma],X[i:i,:],Z[i:i,:],[y[i]]))/40
            end
        end
        @test mixed_logit_mc(t,X,Z,y,40) ≈ -sum(log.(mcP))
        @test mixed_logit_quad([theta;-0.1],X,Z,y,nodes,weights)==Inf
        @test mixed_logit_mc([theta;-0.1],X,Z,y,40)==Inf
        # AD must differentiate through gamma=mu+sigma*z, with fixed draws.
        for objective in (v -> mixed_logit_quad(v,X,Z,y,nodes,weights),
                          v -> mixed_logit_mc(v,X,Z,y,40))
            g=ForwardDiff.gradient(objective,t)
            h=1e-5
            fd=[(objective(t+h.*(collect(1:23).==k))-objective(t-h.*(collect(1:23).==k)))/(2h) for k in 1:23]
            @test all(isfinite,g)
            @test g ≈ fd atol=1e-7
        end
        @test optimize_mixed_logit_quad(X,Z,y;theta_mnl=theta) == [theta;1.0]
        @test optimize_mixed_logit_mc(X,Z,y;theta_mnl=theta) == [theta;1.0]
    end
    @testset "Full MNL estimation, SEs, and allwrap (no mixed estimation)" begin
        output_file=joinpath(@__DIR__,replace(basename(@__FILE__),"Xinlin_tests.jl"=>"Xinlin_output.txt"))
        fits=open(output_file,"w") do io
            redirect_stdout(io) do
                allwrap()
            end
        end
        report=read(output_file,String)
        @test length(fits.theta)==22
        @test length(fits.se)==22
        @test all(isfinite,fits.theta)
        @test all(isfinite,fits.se) && all(fits.se .> 0)
        @test occursin("Optimizer converged: true",report)
        @test occursin("Hessian positive definite: true",report)
        @test fits.start_quad == [fits.theta;1.0]
        @test fits.start_mc == [fits.theta;1.0]
        @test length(collect(eachmatch(r"setup complete \(not executed\)",report)))==2
        @test occursin("ALL ANALYSES COMPLETE",report)
    end
end
