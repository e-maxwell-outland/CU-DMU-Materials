"""
© 2026 Emily Maxwell Outland <emily.maxwell@colorado.edu>
License: BSD-3-Clause

hw5_code.jl

Last Modified: 03/31/2026

Project: ASEN 5264, Decision Making Under Uncertainty, HW5, Q2-3
Organization: Smead Aerospace, CU Boulder

Description
-----------
Q2: Cancer Monitoring POMDP

Models a four-state cancer progression POMDP (healthy, in-situ, invasive, death)
with three actions (wait, test, treat) and binary observations (positive, negative).
Evaluates an "always wait" baseline policy via Monte Carlo simulation and produces
visualizations of the return distribution and state occupancy over time.

Q3: Mountain Car Deep Reinforcement Learning

Implements DQN from scratch to solve a continuous mountain car environment with a
discretized action space [-1, -0.5, 0, 0.5, 1]. Uses experience replay and a frozen
target network to stabilize training. Saves the best checkpoint during training and
evaluates the final policy via HW5.evaluate.

Generative AI Disclosure
-------------------------
This homework, I did pair programming with Claude.

For problem 2, I provided the POMDP specification from the problem statement,
guided the model structure, and provided starter code. Claude implemented the
QuickPOMDP definition (transition, observation, and reward functions), the Monte
Carlo evaluation loop, and the visualization code (return histogram and state
occupancy plot). I directed the choice of visualizations, interpretation of
results, and the choice of descriptive statistics to report.

For problem 3, I selected DQN as the algorithm after discussing tradeoffs with
Claude (vs. PPO and actor-critic). With my guidance, Claude implemented the full DQN from the
starter code skeleton: the corrected Bellman loss, ε-greedy exploration with
decay, the experience replay buffer, the target network update logic, checkpoint
saving, and the learning curve and value heatmap plots. I provided guidance on
hyperparameters, directed the debugging process, and wrote the algorithm
description paragraph.
"""

using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, SparseCat, FunctionPolicy, RolloutSimulator, stepthrough
using Statistics: mean, std
using Plots
import POMDPs

############
# Question 2
############

cancer = QuickPOMDP(
    states      = [:healthy, :in_situ, :invasive, :death],
    actions     = [:wait, :test, :treat],
    observations = [:positive, :negative],
    discount    = 0.99,
    initialstate = Deterministic(:healthy),

    transition = function (s, a)
        if s == :healthy
            return SparseCat([:in_situ, :healthy], [0.02, 0.98])
        elseif s == :in_situ
            if a == :treat
                return SparseCat([:healthy, :in_situ], [0.60, 0.40])
            else # :wait or :test
                return SparseCat([:invasive, :in_situ], [0.10, 0.90])
            end
        elseif s == :invasive
            if a == :treat
                return SparseCat([:healthy, :death, :invasive], [0.20, 0.20, 0.60])
            else # :wait or :test
                return SparseCat([:death, :invasive], [0.60, 0.40])
            end
        else # :death — absorbing
            return Deterministic(:death)
        end
    end,

    observation = function (s, a, sp)
        if a == :test
            if sp == :healthy
                return SparseCat([:positive, :negative], [0.05, 0.95])
            elseif sp == :in_situ
                return SparseCat([:positive, :negative], [0.80, 0.20])
            elseif sp == :invasive
                return Deterministic(:positive)
            else # :death
                return Deterministic(:negative)
            end
        elseif a == :treat
            if sp == :in_situ || sp == :invasive
                return Deterministic(:positive)
            else
                return Deterministic(:negative)
            end
        else # :wait
            return Deterministic(:negative)
        end
    end,

    reward = function (s, a)
        if s == :death
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        else # :treat
            return 0.1
        end
    end
)

# Evaluate "always wait" policy with Monte Carlo simulation
policy = FunctionPolicy(o -> :wait)
sim = RolloutSimulator(max_steps=100)
returns = [POMDPs.simulate(sim, cancer, policy) for _ in 1:10_000]
# Descriptive statistics
ret_mean  = mean(returns)
ret_min   = minimum(returns)
ret_max   = maximum(returns)

# Mode: center of the most-populated histogram bin (meaningful for continuous data)
nbins     = 50
bin_edges = range(ret_min, ret_max, length=nbins+1)
bin_width = step(bin_edges)
bin_counts = zeros(Int, nbins)
for x in returns
    idx = min(nbins, floor(Int, (x - ret_min) / bin_width) + 1)
    bin_counts[idx] += 1
end
peak_bin  = argmax(bin_counts)
ret_mode  = ret_min + (peak_bin - 0.5) * bin_width

println("=== Descriptive Statistics (Always-Wait Policy) ===")
println("  Mean:  $(round(ret_mean, digits=3))")
println("  Mode:  $(round(ret_mode, digits=3))  (peak bin center)")
println("  Range: $(round(ret_min, digits=3)) – $(round(ret_max, digits=3))")

# --- Plot 1: Histogram of discounted returns ---
p1 = histogram(returns,
    bins = 50,
    xlabel = "Discounted Return",
    ylabel = "Count",
    title  = "Always-Wait Policy: Return Distribution",
    legend = :topleft,
    label  = "Simulated returns",
    color  = :steelblue)
vline!(p1, [mean(returns)], color=:red, linewidth=2, label="Mean ($(round(mean(returns), digits=2)))")
display(p1)
savefig(p1, "returns_histogram.png")

# --- Plot 2: State occupancy over time ---
# Run a separate set of shorter rollouts and track state at each step
const MAX_STEPS = 100
const N_TRAJ    = 2_000
const STATES    = [:healthy, :in_situ, :invasive, :death]

# counts[step, state_index] = number of trajectories in that state at that step
counts = zeros(Int, MAX_STEPS, length(STATES))
state_idx = Dict(s => i for (i, s) in enumerate(STATES))

for _ in 1:N_TRAJ
    step = 0
    for (s, a, r, sp) in stepthrough(cancer, policy, "s,a,r,sp", max_steps=MAX_STEPS)
        step += 1
        counts[step, state_idx[s]] += 1
    end
    # fill remaining steps with :death if episode ended early
    for t in (step+1):MAX_STEPS
        counts[t, state_idx[:death]] += 1
    end
end

fractions = counts ./ N_TRAJ

p2 = plot(1:MAX_STEPS, fractions,
    label    = reshape(string.(STATES), 1, :),
    xlabel   = "Step",
    ylabel   = "Fraction of Trajectories",
    title    = "Always-Wait Policy: State Occupancy Over Time",
    linewidth = 2,
    legend   = :right)
display(p2)
savefig(p2, "state_occupancy.png")

############
# Question 3
############

using DMUStudent.HW5: HW5, mc
using CommonRLInterface
import CommonRLInterface: reset!, observe, act!, terminated, actions
using CommonRLInterface.Wrappers: QuickWrapper
using Flux
using BSON: @save, @load
using Random: rand

# Discretize actions and use position/velocity only
env = QuickWrapper(mc,
    actions = [-1.0, -0.5, 0.0, 0.5, 1.0],
    observe  = e -> observe(e)[1:2]
)

# ── Hyperparameters ──────────────────────────────────────────────
const γ_dqn        = 0.99f0
const ε_start      = 1.0
const ε_end        = 0.05
const ε_decay_frac = 0.80   # decay over first 80% of episodes
const BUFFER_SIZE  = 10_000
const BATCH_SIZE   = 64
const MIN_BUFFER   = 1_000
const TARGET_FREQ  = 100    # steps between target net copies
const LR           = 0.0005
const N_EPISODES   = 500
const EVAL_EVERY   = 10     # episodes between eval checkpoints
const EVAL_RUNS    = 5      # greedy rollouts per checkpoint

# ── Bellman loss ─────────────────────────────────────────────────
function dqn_loss(Q, Q_target, s, a_ind, r, sp, done)
    target = done ? Float32(r) : Float32(r) + γ_dqn * maximum(Q_target(sp))
    return (Q(s)[a_ind] - target)^2
end

# ── ε-greedy action selection ────────────────────────────────────
function select_action(Q, s, ε, n_actions)
    if rand() < ε
        return rand(1:n_actions)
    else
        return argmax(Q(s))
    end
end

# ── Greedy evaluation (no exploration) ──────────────────────────
function eval_policy(Q, env, n_runs)
    act_space = actions(env)
    returns = Float64[]
    for _ in 1:n_runs
        reset!(env)
        R = 0.0; disc = 1.0
        ep_step = 0
        while !terminated(env) && ep_step < 999
            s  = observe(env)
            a  = argmax(Q(s))
            r  = act!(env, act_space[a])
            R += disc * r
            disc *= γ_dqn
            ep_step += 1
        end
        push!(returns, R)
    end
    return mean(returns)
end

# ── DQN training ─────────────────────────────────────────────────
function train_dqn(env)
    n_actions = length(actions(env))

    Q = Chain(Dense(2, 128, relu), Dense(128, n_actions))
    Q_target = deepcopy(Q)
    opt = Flux.setup(Adam(LR), Q)

    buffer = Vector{Tuple{Vector{Float32}, Int, Float32, Vector{Float32}, Bool}}()
    sizehint!(buffer, BUFFER_SIZE)

    eval_episodes = Int[]
    eval_returns  = Float64[]

    step_count  = 0
    best_return = -Inf
    Q_best      = deepcopy(Q)
    ε = ε_start
    decay_episodes = round(Int, ε_decay_frac * N_EPISODES)

    for episode in 1:N_EPISODES
        reset!(env)

        ep_step = 0
        while !terminated(env) && ep_step < 999
            s      = Float32.(observe(env))
            a_ind  = select_action(Q, s, ε, n_actions)
            r      = act!(env, actions(env)[a_ind])
            sp     = Float32.(observe(env))
            done   = terminated(env) || ep_step == 998
            ep_step += 1

            # ring-buffer: drop oldest if full
            if length(buffer) >= BUFFER_SIZE
                deleteat!(buffer, 1)
            end
            push!(buffer, (s, a_ind, Float32(r), sp, done))

            if length(buffer) >= MIN_BUFFER
                batch = rand(buffer, BATCH_SIZE)
                for data in batch
                    _, grads = Flux.withgradient(dqn_loss, Q, Q_target, data...)
                    Flux.update!(opt, Q, grads[1])
                end
            end

            step_count += 1
            if step_count % TARGET_FREQ == 0
                Q_target = deepcopy(Q)
            end
        end

        # linear ε decay
        if episode <= decay_episodes
            ε = ε_start + (ε_end - ε_start) * (episode / decay_episodes)
        else
            ε = ε_end
        end

        # periodic evaluation
        if episode % EVAL_EVERY == 0
            ret = eval_policy(Q, env, EVAL_RUNS)
            push!(eval_episodes, episode)
            push!(eval_returns, ret)
            if ret > best_return
                best_return = ret
                Q_best = deepcopy(Q)
                @save "best_model.bson" Q_best
                println("Episode $episode | ε=$(round(ε, digits=3)) | eval return=$(round(ret, digits=2)) ← new best")
            else
                println("Episode $episode | ε=$(round(ε, digits=3)) | eval return=$(round(ret, digits=2))")
            end
        end
    end

    return Q_best, eval_episodes, eval_returns
end

println("Training DQN...")
Q_dqn, dqn_eps, dqn_rets = train_dqn(env)

# To reload the best model later without retraining:
# @load "best_model.bson" Q_best
# Q_dqn = Q_best

# ── Learning curve ────────────────────────────────────────────────
p3 = plot(dqn_eps, dqn_rets,
    xlabel    = "Episode",
    ylabel    = "Mean Discounted Return (γ=0.99)",
    title     = "DQN Learning Curve — Mountain Car",
    label     = "DQN (hand-rolled)",
    linewidth = 2,
    legend    = :bottomright)
hline!(p3, [40.0], color=:red, linestyle=:dash, linewidth=1.5, label="Target (40)")
display(p3)
savefig(p3, "learning_curve_dqn.png")

# ── Value function heatmap ────────────────────────────────────────
xs_mc = -3.0f0:0.1f0:3.0f0
vs_mc = -0.3f0:0.01f0:0.3f0
p4 = heatmap(xs_mc, vs_mc,
    (x, v) -> maximum(Q_dqn(Float32[x, v])),
    xlabel = "Position (x)",
    ylabel = "Velocity (v)",
    title  = "Max Q Value — Learned DQN Policy")
display(p4)
savefig(p4, "value_heatmap_dqn.png")

# ── Final evaluation for submission ──────────────────────────────
HW5.evaluate(s -> actions(env)[argmax(Q_dqn(Float32.(s[1:2])))], "emily.maxwell@colorado.edu")
