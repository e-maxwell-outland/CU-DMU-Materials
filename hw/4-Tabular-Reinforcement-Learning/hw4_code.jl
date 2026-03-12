"""
© 2026 Emily Maxwell Outland <emily.maxwell@colorado.edu>
License: BSD-3-Clause

hw4_code.jl

Last Modified: 03/10/2026

Project: ASEN 5264, Decision Making Under Uncertainty, HW4, Q3-4
Organization: Smead Aerospace, CU Boulder

Description
-----------
Q3: Function Approximation with a Neural Network
Q4: Tabular Reinforcement Learning

Generative AI Disclosure
-------------------------
This homework, I did pair programming with Claude. 

For problem 3, it helped me discover how to find a solution to the issues in fitting the function with the NN. 
See the citations below for the methodology of using Fourier features to overcome spectral bias.

For problem 4, I gave it the SARSA code from the class notebook and gave guidance on adapting/implementing: SARSA-λ and Q-learning to start. 
After the intial results, we worked together to make the following adjustments (my guidance, critique, and analysis at a high-level but almost exclusively Claude's code):

1. SARSA-λ
- Optimisitic initialization: use +10, since that was the max reward in the env, so unvisited states always look more promising, encouraging exploration in a sparse-reward environment
- Decaying ε-greedy: start with ε=1 and decay to 0.01 over the first 80% of training, then keep at 0.01 to allow some continued exploration
- 100k episodes: this is a hard env, so we need more episodes to learn a good policy
- 10 independent runs to find the best policy and get mean±std learning curves since the env has high variance
- Check-point based eval every 500 episodes

2. Q-learning
- UCB exploration: replace ε-greedy because a count-based bonus decays as states are visited
- β=20: tuned to give a good balance of exploration vs exploitation in this env
- Zero init because UCB handles exploration
- 150k episodes: Q-learning is off-policy and learns faster, so we can train for more episodes in a similar time budget
- Shared global N visit count dict, shared across all episodes, not per-episode, to properly drive UCB exploration

3. (Bonus, because I was curious) Double Q-learning
- Two separate Q tables (QA and QB) to reduce overestimation bias, 50/50 coin flip to decide which one to update each step
- UCB exploration based on the sum of QA and QB to encourage exploration of actions that are uncertain in either table
- α=0.2: tuned to compensate for the fact that each table is only updated half the time
"""

using Pkg
Pkg.add(["Flux", "Plots", "Statistics"])
Pkg.add(["CommonRLInterface", "Statistics"])
Pkg.add(url="https://github.com/zsunberg/DMUStudent.jl")

using DMUStudent.HW4: gw, render
using CommonRLInterface: actions, act!, observe, reset!, terminated, observations
using Plots
using Random
using Flux
using Statistics

Random.seed!(42)

#------------- 
## Question 3
#-------------

# Fits f(x) = (1 - x) * sin(20 * log(x + 0.2)) using Flux.jl

# Uses Fourier Feature Mapping (Tancik et al., NeurIPS 2020) to overcome spectral bias. Standard MLPs fail on high-frequency functions because
# gradient dynamics preferentially fit low frequencies first (Rahaman et al., 2019). Mapping x → [cos(2π B x), sin(2π B x)] before the MLP fixes this by
# transforming the neural tangent kernel into a stationary kernel with tunable bandwidth controlled by σ.

# Target Function

f(x) = (1 - x) * sin(20 * log(x + 0.2))

# Fourier Feature Mapping
# Maps scalar x → 2m-dimensional vector: [cos(2π b₁x), sin(2π b₁x), ..., cos(2π bₘx), sin(2π bₘx)]
# where bⱼ ~ N(0, σ²). σ controls bandwidth — higher σ = higher frequencies captured.
# m = number of frequency samples (encoded dimension = 2m)

function make_fourier_features(m::Int, σ::Float32)
    B = randn(Float32, m) .* σ          # sample m frequencies
    function encode(x::AbstractMatrix)  # x is (1, N)
        phases = 2f0 * Float32(π) .* B .* x   # (m, N)
        return vcat(cos.(phases), sin.(phases))  # (2m, N)
    end
    return encode, B
end

m = 64          # number of Fourier features: input to MLP is 2m = 128 dimensional
σ = 10f0        # bandwidth: tuned to match the about 20 rad/unit frequency of f(x)

encode, B = make_fourier_features(m, σ)

# Data

n_train = 500
x_train = sort(rand(Float32, n_train))
y_train = Float32.(f.(x_train))

X_train_raw = reshape(x_train, 1, :)
X_train     = encode(X_train_raw)          # (128, 500)
Y_train     = reshape(y_train, 1, :)       # (1, 500)

# Model: standard MLP, now operating on Fourier-encoded inputs

model = Chain(
    Dense(2m => 128, relu),
    Dense(128 => 64, relu),
    Dense(64 => 1)
)

# Training Loop
loss(m_net, x, y) = Flux.mse(m_net(x), y)
opt_state = Flux.setup(Adam(0.001), model)

n_epochs   = 2000
batch_size = 64
losses     = Float64[]

data = Flux.DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=true)

for epoch in 1:n_epochs
    for (xb, yb) in data
        grads = Flux.gradient(model) do m_net
            loss(m_net, xb, yb)
        end
        Flux.update!(opt_state, model, grads[1])
    end

    epoch_loss = loss(model, X_train, Y_train)
    push!(losses, epoch_loss)

    if epoch % 200 == 0
        println("Epoch $epoch — Loss: $(round(epoch_loss, digits=6))")
    end
end

# Plot 1: 100 test points through trained model

x_test  = Float32.(collect(range(0f0, 1f0, length=100)))
X_test  = encode(reshape(x_test, 1, :))
y_true  = f.(x_test)
y_pred  = vec(model(X_test))

p1 = plot(x_test, y_true,
    label="True f(x)", lw=2, color=:steelblue,
    xlabel="x", ylabel="f(x)",
    title="Fourier Feature Network Approximation of f(x)",
    legend=:topright, size=(700, 400)
)
scatter!(p1, x_test, y_pred,
    label="NN prediction", markersize=3, color=:darkorange, alpha=0.8
)
savefig(p1, "hw4_q3_fit.png")
println("Saved fit plot → hw4_q3_fit.png")

# Plot 2: Learning curve MSE loss vs epoch

p2 = plot(1:n_epochs, losses,
    xlabel="Epoch", ylabel="MSE Loss",
    title="Learning Curve", label="Training Loss",
    lw=1.5, color=:seagreen, size=(700, 400), yscale=:log10
)
savefig(p2, "hw4_q3_loss.png")
println("Saved learning curve → hw4_q3_loss.png")

#------------- 
## Question 4
#-------------

#  Config Stuff

const N_RUNS     = 10    # number of independent runs for mean±std curves
const N_EPISODES = 15000 # episodes per run (all algorithms)
const EVAL_EVERY = 500   # evaluate greedy policy every N episodes
const EVAL_EPS   = 200   # episodes per evaluation (fast but stable)
const MAX_STEPS  = 500   # max steps per eval episode

env = gw

#  Timing helper — prints after first run, offers to abort if taking too long

function check_timing(elapsed, run, n_runs, fallback=5)
    if run == 1
        projected = elapsed * n_runs / 60
        println("  Run 1 took $(round(elapsed, digits=1))s → " *
                "projected total: $(round(projected, digits=1)) min")
        if elapsed > 600 && n_runs > fallback
            println("  > 10 min/run detected — falling back to $fallback runs")
            return fallback
        end
    end
    return n_runs
end

#  Evaluation 

function evaluate(env, Q; n_episodes=EVAL_EPS, max_steps=MAX_STEPS)
    returns = Float64[]
    for _ in 1:n_episodes
        reset!(env)
        s = observe(env)
        r_total = 0.0
        steps = 0
        while !terminated(env) && steps < max_steps
            a = argmax(a -> Q[(s, a)], actions(env))
            r_total += act!(env, a)
            s = observe(env)
            steps += 1
        end
        push!(returns, r_total)
    end
    return mean(returns)
end

function print_eval(name, Q)
    returns = Float64[]
    for _ in 1:500
        reset!(env)
        s = observe(env)
        r_total = 0.0
        steps = 0
        while !terminated(env) && steps < MAX_STEPS
            a = argmax(a -> Q[(s, a)], actions(env))
            r_total += act!(env, a)
            s = observe(env)
            steps += 1
        end
        push!(returns, r_total)
    end
    println("$name → mean: $(round(mean(returns),digits=3))  " *
            "max: $(round(maximum(returns),digits=3))  " *
            "pct>5: $(round(mean(returns.>5)*100,digits=1))%")
end

#  SARSA-λ with optimistic initialization 

function sarsa_lambda_episode!(Q, env; epsilon, gamma=0.99, alpha=0.1, lambda=0.9)
    start = time()
    policy(s) = rand() < epsilon ? rand(actions(env)) : argmax(a -> Q[(s,a)], actions(env))
    s  = observe(env)
    a  = policy(s)
    r  = act!(env, a)
    sp = observe(env)
    hist = [s]
    N  = Dict{Any,Float64}()
    while !terminated(env)
        ap = policy(sp)
        N[(s,a)] = get(N,(s,a),0.0) + 1
        delta = r + gamma*Q[(sp,ap)] - Q[(s,a)]
        for (key,n) in N
            Q[key] += alpha*delta*n
            N[key] *= gamma*lambda
        end
        s=sp; a=ap; r=act!(env,a); sp=observe(env)
        push!(hist, sp)
    end
    N[(s,a)] = get(N,(s,a),0.0) + 1
    delta = r - Q[(s,a)]
    for (key,n) in N
        Q[key] += alpha*delta*n
    end
    return (hist=hist, Q=Q, time=time()-start)
end

function run_sarsa_lambda(env; n_episodes=N_EPISODES, alpha=0.1, lambda=0.9, init=10.0)
    Q = Dict((s,a) => init for s in observations(env), a in actions(env))
    checkpoints_steps = Int[]
    checkpoints_time  = Float64[]
    checkpoints_vals  = Float64[]
    cumsteps = 0
    cumtime  = 0.0

    for i in 1:n_episodes
        reset!(env)
        ep = sarsa_lambda_episode!(Q, env;
            epsilon=max(0.01, 1 - i/(n_episodes*0.8)),
            alpha=alpha, lambda=lambda)
        cumsteps += length(ep.hist)
        cumtime  += ep.time

        if i % EVAL_EVERY == 0
            push!(checkpoints_steps, cumsteps)
            push!(checkpoints_time,  cumtime)
            push!(checkpoints_vals,  evaluate(env, Q))
        end
    end
    return (Q=copy(Q), steps=checkpoints_steps,
            times=checkpoints_time, vals=checkpoints_vals)
end

#  Q-Learning with UCB exploration 

function run_q_learning_ucb(env; n_episodes=N_EPISODES, alpha=0.1, beta=15.0, init=0.0)
    Q = Dict((s,a) => init   for s in observations(env), a in actions(env))
    N = Dict((s,a) => 0      for s in observations(env), a in actions(env))
    checkpoints_steps = Int[]
    checkpoints_time  = Float64[]
    checkpoints_vals  = Float64[]
    cumsteps = 0
    cumtime  = 0.0

    for i in 1:n_episodes
        start = time()
        reset!(env)
        s = observe(env)
        hist = [s]

        while !terminated(env)
            # UCB action selection — no ε needed, bonus handles exploration
            a = argmax(a -> Q[(s,a)] + beta/sqrt(max(N[(s,a)], 1)), actions(env))
            N[(s,a)] += 1
            r  = act!(env, a)
            sp = observe(env)
            Q[(s,a)] += alpha*(r + 0.99*maximum(Q[(sp,ap)] for ap in actions(env)) - Q[(s,a)])
            s = sp
            push!(hist, s)
        end

        cumsteps += length(hist)
        cumtime  += time() - start

        if i % EVAL_EVERY == 0
            push!(checkpoints_steps, cumsteps)
            push!(checkpoints_time,  cumtime)
            push!(checkpoints_vals,  evaluate(env, Q))
        end
    end
    return (Q=copy(Q), steps=checkpoints_steps,
            times=checkpoints_time, vals=checkpoints_vals)
end

#  Double Q-Learning with UCB exploration 

function run_double_q_learning_ucb(env; n_episodes=N_EPISODES, alpha=0.1, beta=15.0, init=0.0)
    QA = Dict((s,a) => init for s in observations(env), a in actions(env))
    QB = Dict((s,a) => init for s in observations(env), a in actions(env))
    N  = Dict((s,a) => 0    for s in observations(env), a in actions(env))
    checkpoints_steps = Int[]
    checkpoints_time  = Float64[]
    checkpoints_vals  = Float64[]
    cumsteps = 0
    cumtime  = 0.0

    for i in 1:n_episodes
        start = time()
        reset!(env)
        s = observe(env)
        hist = [s]

        while !terminated(env)
            a = argmax(a -> QA[(s,a)] + QB[(s,a)] + 2*beta/sqrt(max(N[(s,a)],1)), actions(env))
            N[(s,a)] += 1
            r  = act!(env, a)
            sp = observe(env)
            if rand() < 0.5
                best_a = argmax(a -> QA[(sp,a)], actions(env))
                QA[(s,a)] += alpha*(r + 0.99*QB[(sp,best_a)] - QA[(s,a)])
            else
                best_a = argmax(a -> QB[(sp,a)], actions(env))
                QB[(s,a)] += alpha*(r + 0.99*QA[(sp,best_a)] - QB[(s,a)])
            end
            s = sp
            push!(hist, s)
        end

        cumsteps += length(hist)
        cumtime  += time() - start

        if i % EVAL_EVERY == 0
            push!(checkpoints_steps, cumsteps)
            push!(checkpoints_time,  cumtime)
            Q_mean = Dict(k => (QA[k]+QB[k])/2 for k in keys(QA))
            push!(checkpoints_vals,  evaluate(env, Q_mean))
        end
    end
    Q_mean = Dict(k => (QA[k]+QB[k])/2 for k in keys(QA))
    return (Q=Q_mean, steps=checkpoints_steps,
            times=checkpoints_time, vals=checkpoints_vals)
end

#  Multi-run wrapper 
# Runs each algorithm n_runs times, collects vals at each checkpoint,
# returns mean and std across runs interpolated onto a common x-axis.

# Returns (steps_result, clock_result) in a single training pass — no double training
function multirun(run_fn, env, n_runs)
    all_steps_xs = Vector{Vector}()
    all_clock_xs = Vector{Vector}()
    all_steps_ys = Vector{Vector{Float64}}()
    all_clock_ys = Vector{Vector{Float64}}()
    best_Q   = nothing
    best_val = -Inf

    actual_runs = n_runs
    for r in 1:actual_runs
        Random.seed!(r * 100)
        t_start = time()
        result = run_fn(env)
        elapsed = time() - t_start

        actual_runs = check_timing(elapsed, r, actual_runs, 5)

        push!(all_steps_xs, result.steps)
        push!(all_clock_xs, result.times)
        push!(all_steps_ys, result.vals)
        push!(all_clock_ys, result.vals)

        if mean(result.vals[max(1,end-3):end]) > best_val
            best_val = mean(result.vals[max(1,end-3):end])
            best_Q   = result.Q
        end

        println("  run $r/$(actual_runs) done — final eval: $(round(result.vals[end], digits=3))")
        r >= actual_runs && break
    end

    function summarize(all_xs, all_ys)
        ref_xs = all_xs[1]
        interp_ys = []
        for (xs, ys) in zip(all_xs, all_ys)
            iys = [ys[min(searchsortedfirst(xs, x, lt=(<)), length(ys))] for x in ref_xs]
            push!(interp_ys, iys)
        end
        mat   = hcat(interp_ys...)
        means = vec(mean(mat, dims=2))
        stds  = vec(std(mat,  dims=2))
        return (xs=ref_xs, means=means, stds=stds, best_Q=best_Q)
    end

    return summarize(all_steps_xs, all_steps_ys), summarize(all_clock_xs, all_clock_ys)
end

#  Run everything 

println("="^50)
println("Running SARSA-lambda ($(N_RUNS) runs)...")
sarsa_steps, sarsa_clock = multirun(e -> run_sarsa_lambda(e, n_episodes=100000), env, N_RUNS)
print_eval("SARSA-lambda (best)", sarsa_steps.best_Q)

println("="^50)
println("Running UCB Q-Learning ($(N_RUNS) runs)...")
ql_steps, ql_clock = multirun(e -> run_q_learning_ucb(e, n_episodes=150000, beta=20.0), env, N_RUNS)
print_eval("UCB Q-Learning (best)", ql_steps.best_Q)

println("="^50)
println("Running Double UCB Q-Learning ($(N_RUNS) runs)...")
dql_steps, dql_clock = multirun(e -> run_double_q_learning_ucb(e, n_episodes=150000, beta=20.0, alpha=0.2), env, N_RUNS)
print_eval("Double UCB Q-Learning (best)", dql_steps.best_Q)

#  Plot helper 

function plot_curves(results, title_str, xlabel_str)
    colors = [:steelblue, :darkorange, :seagreen]
    names  = ["SARSA-lambda", "UCB Q-Learning", "Double UCB Q-Learning"]
    p = plot(title=title_str, xlabel=xlabel_str,
             ylabel="Avg undiscounted return", legend=:bottomright)
    for (i, (res, name, col)) in enumerate(zip(results, names, colors))
        plot!(p, res.xs, res.means,
              ribbon=res.stds, fillalpha=0.15,
              label=name, lw=2, color=col)
    end
    return p
end

println("Generating plots...")
p1 = plot_curves([sarsa_steps, ql_steps, dql_steps],
                 "Learning Curves (mean ± std, $(N_RUNS) runs)",
                 "Steps in environment")
savefig(p1, "hw4_q4_steps.png")
println("Saved -> hw4_q4_steps.png")

p2 = plot_curves([sarsa_clock, ql_clock, dql_clock],
                 "Learning Curves (mean ± std, $(N_RUNS) runs)",
                 "Wall clock time (s)")
savefig(p2, "hw4_q4_clock.png")
println("Saved -> hw4_q4_clock.png")