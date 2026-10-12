# PS5 coding

## Files

- `PS5_LastName_source.jl`: data loading, static logit, backward recursion, dynamic negative log likelihood, and optimizer. Retains the updated starter's function names, variable names, explanatory comments, and loop order.
- `PS5_LastName_script.jl`: loads packages, includes the source, defines `main()`, and calls it. Static estimates initialize the dynamic optimizer.
- `PS5_LastName_tests.jl`: loads packages and source, then runs the separate unit tests required by Question 4.
- `create_grids.jl`: unchanged instructor helper, retrieved from the course repository below.
- `busdataBeta0.csv`, `busdata.csv`: supplied data, unchanged.
- `Project.toml`, `Manifest.toml`: package environment used for verification (Julia 1.10.10).
- `PS5_estimation_output.txt`, `PS5_tests_output.txt`: actual run output.

Instructor helper: https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/create_grids.jl

## Run

Extract all files into one directory. Open a terminal in that directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. PS5_LastName_tests.jl
julia --project=. PS5_LastName_script.jl
```

If using an existing Julia environment, install the listed packages there and run the two files without `--project=.`. Do not run the source file by itself: packages are intentionally loaded in the script/tests. The first run may spend additional time compiling packages.

Before submitting, replace `LastName` in the three filenames and the two `include` statements with your surname; also fill in your name and any applicable group members at the top. Submit the three required `.jl` files and outputs through the course GitHub workflow. Keep `create_grids.jl` available alongside the source. This bundle has not been submitted to GitHub on your behalf.

## Implementation notes / writeup

1. Question 1 retains the starter's two `stack` operations and join by bus/time, followed by sorting. Local CSV files are read first, with the original course URL as fallback.
2. Question 2 uses `glm(@formula(Y ~ Odometer + Branded), ..., Binomial(), LogitLink())`. The coefficient order is intercept, mileage, brand. Here `Y=1` means keep running, and `Y=0` means replace.
3. Question 3 keeps the starter's four nested loops: time backwards, brand, route state, mileage state. `FV` stores **beta times** the expected value. Consequently, continuation expectations and their difference use `FV` directly, without another beta multiplier. The terminal slice is zero. State-grid mileage is used for solving the model; observed mileage is used in the likelihood.
4. `log(exp(v0)+exp(v1))` and the logit log probabilities use algebraically equivalent overflow-safe expressions. This only changes numerical evaluation, not the model. The objective returns the **negative** log likelihood for minimization.
5. Question 4 tests the instructor transition matrix, wide-to-long alignment, the static model score, data/state dimensions, backward recursion against an independent two-period reference, beta=0, terminal conditions, extreme utilities, observed-state likelihood, and optimizer recovery of a known finite MLE.

AI assistance was used to complete and test the code, as permitted by the assignment. All 66 unit-test assertions passed. The optimizer output in PS5_tests_output.txt is for the synthetic test fixture, not the homework data; homework estimates are in PS5_estimation_output.txt.

Starter compatibility fix: Julia 1.10 cannot attach a docstring directly to the nested `@views @inbounds function` expression. Those two explanatory docstrings are retained verbatim as block comments, with a short note at each location. The requested macros remain on the function declarations.

## Verified estimates

| Parameter | Static (beta=0) | Dynamic (beta=0.9) |
|---|---:|---:|
| Intercept | 1.925959 | 1.9378 |
| Mileage | -0.148154 | -0.1510 |
| Branded | 1.059189 | 1.0933 |

Static log likelihood: -8000.923590. Dynamic log likelihood: -8765.00 (rounded); Optim reports convergence after 24 iterations. Both models use 20,000 observations, from their respective assigned CSV files.

Mileage lowers the flow utility of continuing to operate an engine, while branding raises it. In the dynamic model these are flow-utility coefficients; choice probabilities also reflect the future value difference. The two likelihoods use different datasets and should not be treated as a direct model-fit comparison.
