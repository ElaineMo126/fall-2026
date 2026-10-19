# Load required packages
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

# Read in function to create state transitions for dynamic model.
# This is the instructor-provided create_grids.jl code, included here so that
# the three requested submission files are self-contained.
function create_grids()
    function xgrid(theta,xval)
        N=length(xval)
        xub=vcat(xval[2:N],Inf)
        xtran1=zeros(N,N)
        xtran1c=zeros(N,N)
        lcdf=zeros(N)
        for i=1:length(xval)
            xtran1[:,i]=(xub[i].>=xval).*(1 .- exp.(-theta*(xub[i].-xval)) .- lcdf)
            lcdf .+= xtran1[:,i]
            xtran1c[:,i] .+= lcdf
        end
        return xtran1,xtran1c
    end
    zval=collect([.25:.01:1.25]...)
    zbin=length(zval)
    xval=collect([0:.125:25]...)
    xbin=length(xval)
    tbin=xbin*zbin
    xtran=zeros(tbin,xbin)
    xtranc=zeros(xbin,xbin,xbin)
    for z=1:zbin
        xtran[(z-1)*xbin+1:z*xbin,:],xtranc[:,:,z]=xgrid(zval[z],xval)
    end
    return zval,zbin,xval,xbin,xtran
end

#========================================
QUESTION 1: Data Loading and Reshaping
========================================#

"""
    load_and_reshape_data(url::String)

Load the bus data from PS5 and reshape to long panel format.

# Arguments
- `url::String`: URL to the CSV file (second CSV from PS5)

# Returns
- `df_long::DataFrame`: Long panel format with columns for bus ID, time, and decision variables
- `Xstate::Matrix`: Discretized mileage state, N rows * T columns (wide format, needed in Q3)
- `Zstate::Vector`: Discretized route usage state, length N
- `Branded::Vector`: Branded indicator, length N

# TODO:
- Read in the CSV data
- Reshape from wide to long format
- Ensure you have the necessary variables: bus ID, time period, Odometer, RouteUsage, Branded, Y (decision)
- Also pull out the discretized state variables in wide format (same as in PS5)
"""
function load_and_reshape_data(url::String)
    # TODO: Implement data loading
    # Hint: Use CSV.read() and HTTP.get()
    df = isfile(url) ? CSV.read(url, DataFrame) : CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :bus_id = 1:size(df,1))
    
    # TODO: Reshape to long format
    # Hint: You may need to use stack() or similar reshaping functions
    # Each row should represent one bus-time period observation
    # Make sure df_long is sorted by bus, then time (Q3 relies on this ordering)
    T = length(names(df, r"^Y\d+"))

    dfy = select(df, :bus_id, names(df, r"^Y\d+")...,
                 :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, names(df, r"^Y\d+"))
    rename!(dfy_long, :value => :Y)
    dfy_long.time = repeat(collect(1:T), inner=nrow(df))
    select!(dfy_long, Not(:variable))

    dfx = select(df, :bus_id, names(df, r"^Odo\d+")...)
    dfx_long = DataFrames.stack(dfx, names(df, r"^Odo\d+"))
    rename!(dfx_long, :value => :Odometer)
    dfx_long.time = repeat(collect(1:T), inner=nrow(df))
    select!(dfx_long, Not(:variable))

    df_long = leftjoin(dfy_long, dfx_long, on=[:bus_id, :time])
    sort!(df_long, [:bus_id, :time])
    
    # TODO: Extract Xstate, Zstate, and Branded in wide format
    # Hint: look at how PS5 built these from the Xst* and Zst columns
    Xstate = Matrix(df[:, r"^Xst"])
    Zstate = Vector(df[:, :Zst])
    Branded = Vector(df[:, :Branded])
    
    return df_long, Xstate, Zstate, Branded
end


#========================================
QUESTION 2: Flexible Logit Estimation
========================================#

"""
    estimate_flexible_logit(df::DataFrame)

Estimate a flexible logit model with fully interacted terms up to 7th order.

# Arguments
- `df::DataFrame`: Long panel data frame

# Returns
- Fitted GLM model object

# TODO:
- Create squared terms for continuous variables
- Use GLM with formula syntax to specify fully interacted model
- Remember: Odometer * RouteUsage * Branded * time means all interactions up to the product of all four
"""
function estimate_flexible_logit(df::DataFrame)
    # Explicitly create squared terms in the DataFrame
    df = @transform(df, 
        :Odometer_sq = :Odometer .^ 2,
        :RouteUsage_sq = :RouteUsage .^ 2,
        :time_sq = :time .^ 2
    )
    
    # Estimate the flexible logit using the explicitly created terms
    flex_logit = glm(
        @formula(Y ~ Odometer * Odometer_sq * RouteUsage * RouteUsage_sq * Branded * time * time_sq),
        df, Binomial(), LogitLink()
    )
    
    return flex_logit
end


#========================================
QUESTION 3: CCP-Based Estimation
========================================#

"""
    construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)

Construct the state space grid for all possible states.

# Arguments
- `xbin::Int`: Number of mileage bins
- `zbin::Int`: Number of route usage bins  
- `xval::Vector`: Grid points for mileage
- `zval::Vector`: Grid points for route usage

# Returns
- `state_df::DataFrame`: Data frame with all possible state combinations

# TODO:
- Create vectors using kron(ones(zbin), xval) for Odometer
- Create vectors using kron(zval, ones(xbin)) for RouteUsage
- Check: row k of state_df must describe the same (mileage, route usage) state
  as row k of xtran (look at how create_grids() fills xtran)
- Initialize Branded and time to zeros
"""
function construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)
    # TODO: Implement state space construction
    # Hint: kron() is the Kronecker product function
    
    state_df = DataFrame(
        Odometer = kron(ones(zbin), xval),
        RouteUsage = kron(zval, ones(xbin)),
        Branded = zeros(xbin*zbin),
        time = zeros(xbin*zbin)
    )
    
    return state_df
end


"""
    compute_future_values(state_df::DataFrame, 
                          flex_logit::GeneralizedLinearModel,
                          xtran::Matrix, 
                          xbin::Int, 
                          T::Int, 
                          β::Float64)

Compute future value terms using CCPs from the flexible logit.

# Arguments
- `state_df::DataFrame`: State space data frame
- `flex_logit`: Fitted flexible logit model
- `xtran::Matrix`: State transition matrix
- `xbin::Int`: Number of mileage bins
- `T::Int`: Number of time periods
- `β::Float64`: Discount factor (0.9)

# Returns
- `FV::Array{Float64,3}`: Future value array (states × brand × time)

# TODO:
- Initialize FV as zeros(size(xtran,1), 2, T+1)
- Loop over t = 2 to T
- Loop over brand states b ∈ {0,1}
- Update state_df with current t and b values
- Use predict() to get p0 (probability of replacement)
- Store -β * log(p0) in FV[:, b+1, t]
"""
function compute_future_values(state_df::DataFrame, 
                                flex_logit,
                                xtran::Matrix, 
                                xbin::Int, 
                                T::Int, 
                                β::Float64)
    
    # TODO: Initialize the future value array
    # FV = zeros(???, 2, T+1)
    FV = zeros(size(xtran,1), 2, T+1)
    
    # TODO: Nested loops over time and brand
    # for t in 2:T
    #     for b in 0:1
    #         # Update state_df
    #         # Compute p0 using predict()
    #         # Store -β * log.(p0) in FV
    #     end
    # end
    for t in 2:T
        for b in 0:1
            state_df.time .= t
            state_df.Branded .= b
            # Y=1 means continuing to run the bus, so p0 is replacement.
            p0 = 1 .- predict(flex_logit, state_df)
            p0 = clamp.(p0, eps(Float64), 1.0)
            FV[:, b+1, t] .= -β .* log.(p0)
        end
    end
    
    return FV
end


"""
    compute_fvt1(df_long::DataFrame, 
                 FV::Array{Float64,3},
                 xtran::Matrix,
                 Xstate::Matrix,
                 Zstate::Vector,
                 xbin::Int,
                 B::Vector)

Map future values from state space to actual data.

# Arguments
- `df_long::DataFrame`: Original long data frame
- `FV::Array{Float64,3}`: Future value array
- `xtran::Matrix`: State transition matrix
- `Xstate::Matrix`: Mileage state for each bus-year observation
- `Zstate::Vector`: Route usage state for each bus  
- `xbin::Int`: Number of mileage bins
- `B::Vector`: Branded indicator for each bus

# Returns
- `FVT1::Vector`: Future value term for each observation in long format

# TODO:
- Initialize FVT1 matrix to store results
- Loop over observations i and time periods t
- Compute row indices in xtran based on Xstate[i,t] and Zstate[i]
  (these are the same row0/row1 you used in the PS5 likelihood)
- Calculate FVT1[i,t] = (xtran[row1,:] - xtran[row0,:])'* FV[row0:row0+xbin-1, B[i]+1, t+1]
- Convert to long format vector
"""
function compute_fvt1(df_long::DataFrame, 
                      FV::Array{Float64,3},
                      xtran::Matrix,
                      Xstate::Matrix,
                      Zstate::Vector,
                      xbin::Int,
                      B::Vector)
    
    # Get dimensions
    N = length(unique(df_long.bus_id))  # Adjust column name as needed
    T = size(Xstate, 2)  # TODO: determine T from data (don't hardcode 20)
    
    # TODO: Initialize FVT1
    # FVT1 = zeros(N, T)
    FVT1 = zeros(N, T)
    
    # TODO: Loop over observations and time
    # for i in 1:N
    #     for t in 1:T
    #         # Compute row0 and row1 indices (see PS5)
    #         # row0 = [row of xtran for bus i's route usage with mileage reset to 0]
    #         # row1 = [row of xtran for bus i's route usage with mileage at Xstate[i,t]]
    #         
    #         # Compute future value contribution
    #         # FVT1[i,t] = ...
    #     end
    # end
    for i in 1:N
        row0 = (Zstate[i] - 1) * xbin + 1
        for t in 1:T
            row1 = Xstate[i,t] + (Zstate[i] - 1) * xbin
            FVT1[i,t] = (xtran[row1,:] .- xtran[row0,:])' *
                         FV[row0:row0+xbin-1, B[i]+1, t+1]
        end
    end
    
    # TODO: Convert to long format
    # fvt1_long = FVT1'[:]
    fvt1_long = FVT1'[:]
    
    return fvt1_long
end


"""
    estimate_structural_params(df_long::DataFrame, fvt1::Vector)

Estimate structural parameters θ using GLM with offset.

# Arguments
- `df_long::DataFrame`: Long panel data with decision variable
- `fvt1::Vector`: Future value term to use as offset

# Returns
- Fitted GLM model with structural parameters

# TODO:
- Add fvt1 as a column to df_long
- Estimate logit with Odometer and Branded as regressors
- Use offset argument to include future value with coefficient = 1
"""
function estimate_structural_params(df_long::DataFrame, fvt1::Vector)
    # TODO: Add future value to data frame
    # df_long = @transform(df_long, :fv = fvt1)
    df_long = @transform(df_long, :fv = fvt1)
    
    # TODO: Estimate structural model
    # theta_hat = glm(@formula(Y ~ Odometer + Branded), 
    #                 df_long, 
    #                 Binomial(), 
    #                 LogitLink(), 
    #                 offset=df_long.fv)
    theta_hat = glm(@formula(Y ~ Odometer + Branded),
                    df_long,
                    Binomial(),
                    LogitLink(),
                    offset=df_long.fv)
    
    return theta_hat
end


#========================================
MAIN WRAPPER FUNCTION
========================================#

"""
    main()

Main wrapper function to estimate the Rust model using CCPs.
This function calls all the helper functions in sequence.

# TODO:
- Call each function in order
- Pass results between functions appropriately
- Print results at the end
"""
function main()
    println("="^60)
    println("Rust (1987) Bus Engine Replacement Model - CCP Estimation")
    println("="^60)
    
    # Set parameters
    β = 0.9  # Discount factor
    
    # TODO: Define grid parameters (xbin, zbin, xval, zval)
    # These should match what you used in PS5
    
    # Step 1: Load and reshape data
    println("\nStep 1: Loading and reshaping data...")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"  # TODO: Add the correct URL (2nd CSV from PS5)
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    T = size(Xstate, 2)
    
    # Step 2: Estimate flexible logit
    println("\nStep 2: Estimating flexible logit...")
    # TODO: Call estimate_flexible_logit()
    flex_logit = estimate_flexible_logit(df_long)
    
    # Step 3a: Construct state transition matrices
    println("\nStep 3a: Constructing state transition matrices...")
    # TODO: Reuse code from PS5 to construct xtran and the grids (xbin, zbin, xval, zval)
    zval, zbin, xval, xbin, xtran = create_grids()
    
    # Step 3b: Construct state space
    println("\nStep 3b: Constructing state space...")
    # TODO: Call construct_state_space()
    state_df = construct_state_space(xbin, zbin, xval, zval)
    
    # Step 3c: Compute future values
    println("\nStep 3c: Computing future values...")
    # TODO: Call compute_future_values()
    FV = compute_future_values(state_df, flex_logit, xtran, xbin, T, β)
    
    # Step 3d: Map to actual data
    println("\nStep 3d: Mapping future values to data...")
    # TODO: Call compute_fvt1()
    fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)
    
    # Step 3e: Estimate structural parameters
    println("\nStep 3e: Estimating structural parameters...")
    # TODO: Call estimate_structural_params()
    theta_hat_ccp_glm = estimate_structural_params(df_long, fvt1)
    
    # Print results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    # TODO: Print coefficient estimates and standard errors
    println(coeftable(theta_hat_ccp_glm))
    println("Log likelihood: ", loglikelihood(theta_hat_ccp_glm))
    
    return nothing
end
