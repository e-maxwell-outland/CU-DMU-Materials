"""
© 2026 Emily Maxwell Outland <emily.maxwell@colorado.edu>
License: BSD-3-Clause

hw6.jl

Last Modified: 04/18/2026

Project: ASEN 5264, Decision Making Under Uncertainty, HW6, Q1-2, 4
Organization: Smead Aerospace, CU Boulder

Description
-----------
Q1: Tiger POMDP with QMDP Approximation

Implements a discrete Bayesian belief updater, an alpha-vector policy, and a
QMDP solver via value iteration. Compares QMDP pseudo alpha vectors against
SARSOP optimal alpha vectors on the TigerPOMDP, and evaluates both policies
via Monte Carlo simulation.

Q2: Cancer POMDP with Multiple Policies

Defines a four-state cancer progression POMDP (healthy, in-situ, invasive,
death) with three actions (wait, test, treat) and binary observations. Evaluates
three policies:QMDP, a belief-threshold heuristic, and SARSOP, via Monte
Carlo simulation. Then, it plots QMDP pseudo alpha vectors to illustrate why test is
always dominated under full observability.

Q4: LaserTag POMDP

Solves a 10x7 grid laser tag POMDP using POMCP with 100 tree queries per step
and a greedy rollout policy.

Generative AI Disclosure
-------------------------
This homework, I did pair programming with Claude.

For problems 1 and 2, I directed the mathematical formulation of the belief
update, QMDP value iteration, and alpha-vector policy. Claude implemented the
code from the starter skeletons. I directed the design of the heuristic policy
thresholds for Q2, the choice of visualizations, and wrote the written
responses. Claude implemented the plotting and evaluation code.

For problem 4, I selected POMCP as the approach after discussing the state-space
size constraints (which rule out SARSOP and offline QMDP). Claude implemented
the solver configuration and greedy rollout policy. I directed the choice of
hyperparameters and wrote the algorithm description paragraph.
"""

using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic,
                  RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states,
                  ordered_actions, DiscreteUpdater, has_consistent_distributions
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using BasicPOMCP
using LinearAlgebra: dot
using Statistics: mean, std
using Plots
using Plots.PlotMeasures

############
# Question 1
############

# Belief updater: discrete Bayesian filter
# b'(s') = η · Z(o|a,s') · Σ_s T(s'|s,a) · b(s)

struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)

function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    m = up.m
    bp_vec = zeros(length(states(m)))
    for (si, s) in enumerate(ordered_states(m))
        for (spi, sp) in enumerate(ordered_states(m))
            bp_vec[spi] += T(m, s, a, sp) * b.b[si]
        end
    end
    for (spi, sp) in enumerate(ordered_states(m))
        bp_vec[spi] *= Z(m, a, sp, o)
    end
    bp_vec ./= sum(bp_vec)
    return DiscreteBelief(up.m, bp_vec)
end

function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end

# Alpha-vector policy: a* = argmax_a α_a · b

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

beliefvec(b::DiscreteBelief) = b.b

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)
    bvec = beliefvec(b)
    best_val, best_a = -Inf, first(p.alpha_actions)
    for (alpha, a) in zip(p.alphas, p.alpha_actions)
        val = dot(alpha, bvec)
        if val > best_val
            best_val, best_a = val, a
        end
    end
    return best_a
end

# QMDP solver: run MDP value iteration, then package Q*(s,a) as alpha vectors.
# Since QMDP assumes full observability after the first step, the alpha vectors
# are just the MDP Q-values — no observation model is used.

function qmdp_solve(m, discount=discount(m))
    T_mats = transition_matrices(m)
    R_vecs = reward_vectors(m)
    acts   = ordered_actions(m)
    V      = zeros(length(states(m)))

    for _ in 1:1000
        Q_vecs = [R_vecs[a] + discount * T_mats[a] * V for a in acts]
        V_new  = reduce((x, y) -> max.(x, y), Q_vecs)
        maximum(abs.(V_new - V)) < 1e-6 && (V = V_new; break)
        V = V_new
    end

    alphas     = Vector{Float64}[]
    alpha_acts = eltype(acts)[]
    for a in acts
        push!(alphas,     R_vecs[a] + discount * T_mats[a] * V)
        push!(alpha_acts, a)
    end
    return HW6AlphaVectorPolicy(alphas, alpha_acts)
end

m        = TigerPOMDP()
qmdp_p   = qmdp_solve(m)
sarsop_p = solve(SARSOPSolver(), m)
up       = HW6Updater(m)

# Monte Carlo evaluation
qmdp_sims   = [simulate(RolloutSimulator(max_steps=500), m, qmdp_p,   up) for _ in 1:5000]
sarsop_sims = [simulate(RolloutSimulator(max_steps=500), m, sarsop_p, up) for _ in 1:5000]
println("Q1 Results:")
println("QMDP   mean=$(round(mean(qmdp_sims),   digits=3))  SEM=$(round(std(qmdp_sims)/sqrt(5000),   digits=3))")
println("SARSOP mean=$(round(mean(sarsop_sims), digits=3))  SEM=$(round(std(sarsop_sims)/sqrt(5000), digits=3))")

# Alpha vector plots
action_name = Dict(TIGER_LISTEN => "listen", TIGER_OPEN_LEFT => "open-left", TIGER_OPEN_RIGHT => "open-right")
b_range     = 0.0:0.01:1.0

plt_combined = plot(title="Tiger POMDP Alpha Vectors (Combined)",
                    xlabel="P(tiger left)", ylabel="Value", legend=:topright)
for (alpha, a) in zip(qmdp_p.alphas, qmdp_p.alpha_actions)
    plot!(plt_combined, b_range, [dot(alpha, [b, 1-b]) for b in b_range],
          label="QMDP $(action_name[a])", linestyle=:dash)
end
for (alpha, a) in zip(sarsop_p.alphas, sarsop_p.action_map)
    plot!(plt_combined, b_range, [dot(alpha, [b, 1-b]) for b in b_range],
          label="SARSOP $(action_name[a])")
end
savefig(plt_combined, "tiger_alpha_vectors_combined.png")

plt_qmdp = plot(title="QMDP Alpha Vectors", xlabel="P(tiger left)", ylabel="Value", legend=:topright)
for (alpha, a) in zip(qmdp_p.alphas, qmdp_p.alpha_actions)
    plot!(plt_qmdp, b_range, [dot(alpha, [b, 1-b]) for b in b_range],
          label=action_name[a], linestyle=:dash)
end

plt_sarsop = plot(title="SARSOP Alpha Vectors", xlabel="P(tiger left)", ylabel="Value", legend=:topright)
for (alpha, a) in zip(sarsop_p.alphas, sarsop_p.action_map)
    plot!(plt_sarsop, b_range, [dot(alpha, [b, 1-b]) for b in b_range],
          label=action_name[a])
end

plt_sub = plot(plt_qmdp, plt_sarsop, layout=(1,2), size=(1000,450),
               left_margin=10mm, bottom_margin=10mm)
savefig(plt_sub, "tiger_alpha_vectors_subplots.png")

############
# Question 2
############

cancer = QuickPOMDP(
    states       = [:healthy, :in_situ, :invasive, :death],
    actions      = [:wait, :test, :treat],
    observations = [:positive, :negative],
    discount     = 0.99,
    initialstate = Deterministic(:healthy),
    isterminal   = s -> s == :death,

    transition = function (s, a)
        if s == :healthy
            return SparseCat([:in_situ, :healthy], [0.02, 0.98])
        elseif s == :in_situ
            if a == :treat
                return SparseCat([:healthy, :in_situ], [0.60, 0.40])
            else
                return SparseCat([:invasive, :in_situ], [0.10, 0.90])
            end
        elseif s == :invasive
            if a == :treat
                return SparseCat([:healthy, :death, :invasive], [0.20, 0.20, 0.60])
            else
                return SparseCat([:death, :invasive], [0.60, 0.40])
            end
        else # :death
            return Deterministic(:death)
        end
    end,

    # 2-arg form required by the Z helper used in HW6Updater
    observation = function (a, sp)
        if a == :test
            if sp == :healthy;  return SparseCat([:positive, :negative], [0.05, 0.95])
            elseif sp == :in_situ;  return SparseCat([:positive, :negative], [0.80, 0.20])
            elseif sp == :invasive; return Deterministic(:positive)
            else;                   return Deterministic(:negative)
            end
        elseif a == :treat
            return sp in (:in_situ, :invasive) ? Deterministic(:positive) : Deterministic(:negative)
        else # :wait
            return Deterministic(:negative)
        end
    end,

    reward = function (s, a)
        s == :death && return 0.0
        a == :wait  && return 1.0
        a == :test  && return 0.8
        return 0.1  # :treat
    end,
)

@assert has_consistent_distributions(cancer)

qmdp_p   = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
up       = HW6Updater(cancer)

# Heuristic: use belief thresholds to decide when to test vs. treat.
# QMDP never recommends :test because it's reward-dominated by :wait in the
# fully-observable MDP (same transitions, lower reward). In the POMDP, testing
# is the only way to get an informative observation, so any policy that tests
# at the right times should outperform QMDP.
heuristic = FunctionPolicy(function (b)
    p_invasive = pdf(b, :invasive)
    p_cancer   = pdf(b, :in_situ) + p_invasive
    if p_invasive > 0.15 || p_cancer > 0.4
        return :treat
    elseif p_cancer > 0.05
        return :test
    else
        return :wait
    end
end)

n           = 1000
qmdp_sims   = [simulate(RolloutSimulator(), cancer, qmdp_p,    up) for _ in 1:n]
heur_sims   = [simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:n]
sarsop_sims = [simulate(RolloutSimulator(), cancer, sarsop_p,  up) for _ in 1:n]

println("\nQ2 Results:")
println("Policy     | Mean   | SEM")
println("-----------|--------|------")
for (name, sims) in [("QMDP     ", qmdp_sims), ("Heuristic", heur_sims), ("SARSOP   ", sarsop_sims)]
    println("$name  | $(round(mean(sims),digits=2)) | $(round(std(sims)/sqrt(n),digits=3))")
end

# QMDP pseudo alpha vector plot: 1D slices through the belief simplex along
# the cancer progression path show that :test is always dominated by :wait.
let
    ss  = ordered_states(cancer)
    hi  = findfirst(==(:healthy),  ss)
    ii  = findfirst(==(:in_situ),  ss)
    vi  = findfirst(==(:invasive), ss)
    n_s = length(ss)
    t_range = 0.0:0.01:1.0

    function bvec(pairs...)
        b = zeros(n_s)
        for (idx, w) in pairs; b[idx] = w; end
        b
    end

    plt_hi = plot(title="Healthy → In-Situ",   xlabel="P(in situ)",  ylabel="Value", legend=:right)
    plt_iv = plot(title="In-Situ → Invasive", xlabel="P(invasive)", ylabel="Value", legend=:topleft)

    for (alpha, a) in zip(qmdp_p.alphas, qmdp_p.alpha_actions)
        plot!(plt_hi, t_range, [dot(alpha, bvec((hi, 1-t), (ii, t))) for t in t_range], label=string(a))
        plot!(plt_iv, t_range, [dot(alpha, bvec((ii, 1-t), (vi, t))) for t in t_range], label=string(a))
    end

    plt_cancer = plot(plt_hi, plt_iv, layout=(1,2), size=(1000,420),
                      left_margin=10mm, bottom_margin=10mm,
                      plot_title="Cancer POMDP QMDP Pseudo Alpha Vectors")
    savefig(plt_cancer, "cancer_alpha_vectors.png")
end

############
# Question 4
############

m  = LaserTagPOMDP()
up = DiscreteUpdater(m)

# POMCP with a greedy rollout policy. The state space (~343k states) is too
# large for offline solvers like SARSOP or matrix-based QMDP, so we use online
# tree search. The greedy rollout gives POMCP a much better leaf value estimate
# than a random rollout, and c=50 scales the UCB bonus to the +100 tag reward.
function pomcp_solve(m)
    greedy_rollout = FunctionPolicy(function (s)
        dx = s.target[1] - s.robot[1]
        dy = s.target[2] - s.robot[2]
        if     dx > 0; return :right
        elseif dx < 0; return :left
        elseif dy > 0; return :up
        else;          return :down
        end
    end)

    solver = POMCPSolver(
        tree_queries   = 100,
        c              = 50.0,
        default_action = :measure,
        estimate_value = FORollout(greedy_rollout)
    )
    return solve(solver, m)
end

pomcp_p = pomcp_solve(m)

@show HW6.evaluate((pomcp_p, up), n_episodes=100)

HW6.evaluate((pomcp_p, up), "emily.maxwell@colorado.edu")
