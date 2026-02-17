"""
© 2026 Emily Maxwell Outland <emily.maxwell@colorado.edu>
License: BSD-3-Clause

hw3_code.jl

Last Modified: 02/16/2026

Project: ASEN 5264, Decision Making Under Uncertainty, HW3, Q3-6
Organization: Smead Aerospace, CU Boulder

Description
-----------
Q3: Monte Carlo Policy Simulation on a random policy and a heuristic policy for the same environment
Q4: Monte Carlo Tree Search
Q5: MCTS Online
Q6: Fast MCTS

Generative AI Disclosure
-------------------------
This homework, I did pair programming with Claude. I gave an outline for algorithms/approaches based on the lecture materials,
the textbook, and my own brain. Claude then produced a first draft of the code (I went one subpart at a time), which I reviewed and
critiqued if necessary.
"""

using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime

#------------- 
# Question 3
#-------------

m = HW3.DenseGridWorld(seed=3)

function rollout(mdp, policy_function, s0, max_steps=100)
    r_total = 0.0
    s = s0  # Initialize current state
    t = 0
    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(mdp, s)  # Call the policy function
        s, r = @gen(:sp,:r)(mdp, s, a)
        r_total += discount(mdp)^t*r
        t += 1
    end

    return r_total  # Return the accumulated discounted reward
end

function random_policy(m, s)
    return rand(actions(m))
end

# Question 3a: Monte Carlo evaluation of uniform random policy
# Run enough simulations so that SEM < 5
n_simulations = 1000
results = [rollout(m, random_policy, rand(initialstate(m))) for _ in 1:n_simulations]

# Calculate mean and standard error of the mean (SEM)
mean_reward = mean(results)
sem = std(results) / sqrt(n_simulations)

println("Question 3a: Uniform Random Policy Evaluation")
println("Number of simulations: $n_simulations")
println("Mean discounted reward: $(round(mean_reward, digits=2))")
println("Standard error of mean (SEM): $(round(sem, digits=2))")

function heuristic_policy(m, s)
    # Move toward the nearest goal (multiple of 20 in both x and y)
    x, y = s

    # Determine target direction for x-coordinate
    x_mod = x % 20
    target_x_lower = x_mod < 10  # true if we should decrease x, false if increase

    # Determine target direction for y-coordinate
    y_mod = y % 20
    target_y_lower = y_mod < 10  # true if we should decrease y, false if increase

    # Calculate distances to nearest goal in each dimension
    x_dist = x_mod < 10 ? x_mod : 20 - x_mod
    y_dist = y_mod < 10 ? y_mod : 20 - y_mod

    # Prioritize the dimension where we're furthest from goal
    if x_mod == 0 && y_mod == 0
        # Already at a goal, shouldn't happen but just in case
        return rand(actions(m))
    elseif x_mod == 0
        # x is at goal, move in y direction
        return target_y_lower ? :down : :up
    elseif y_mod == 0
        # y is at goal, move in x direction
        return target_x_lower ? :left : :right
    elseif x_dist >= y_dist
        # Prioritize x movement
        return target_x_lower ? :left : :right
    else
        # Prioritize y movement
        return target_y_lower ? :down : :up
    end
end

# Question 3b: Evaluate heuristic policy
n_simulations_heuristic = 1000
results_heuristic = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:n_simulations_heuristic]

mean_reward_heuristic = mean(results_heuristic)
sem_heuristic = std(results_heuristic) / sqrt(n_simulations_heuristic)

println("\nQuestion 3b: Heuristic Policy Evaluation")
println("Number of simulations: $n_simulations_heuristic")
println("Mean discounted reward: $(round(mean_reward_heuristic, digits=2))")
println("Standard error of mean (SEM): $(round(sem_heuristic, digits=2))")
println("Improvement over random: $(round(mean_reward_heuristic - mean_reward, digits=2))")

#-------------
# Question 4
#-------------

# Initialize the MCTS dictionaries
Q = Dict{Tuple{Any,Symbol}, Float64}()  # Q(s,a) -> Q value estimate
N = Dict{Tuple{Any,Symbol}, Int}()      # N(s,a) -> visit count
t = Dict{Tuple{Any,Symbol,Any}, Int}()  # t(s,a,s') -> transition count

""" Upper Confidence Bound for action selection """
function ucb(Q_dict, N_dict, s, a, A, c=1.0)
    if !haskey(N_dict, (s, a)) || N_dict[(s, a)] == 0
        return Inf  # Unexplored actions have infinite value
    end

    # Total visits to state s
    N_s = sum(N_dict[(s, a_i)] for a_i in A if haskey(N_dict, (s, a_i)))

    # UCB formula: Q(s,a) + c * sqrt(log(N(s)) / N(s,a))
    return Q_dict[(s, a)] + c * sqrt(log(N_s) / N_dict[(s, a)])
end

"""
Perform one MCTS simulation from state s
Returns the total discounted reward from this simulation
"""
function simulate!(mdp, Q_dict, N_dict, t_dict, s, d=50, c=1.0, rollout_policy=random_policy)

    if isterminal(mdp, s) || d <= 0
        return 0.0
    end

    # Get available actions
    A = actions(mdp)

    # Selection: Choose action with highest UCB value
    a = argmax(a_i -> ucb(Q_dict, N_dict, s, a_i, A, c), A)

    # Check if this state-action pair has been visited
    if !haskey(N_dict, (s, a))
        N_dict[(s, a)] = 0
        Q_dict[(s, a)] = 0.0
    end

    # Generate next state
    sp, r = @gen(:sp, :r)(mdp, s, a)

    # Track transition
    if !haskey(t_dict, (s, a, sp))
        t_dict[(s, a, sp)] = 0
    end
    t_dict[(s, a, sp)] += 1

    # If this is the first visit, do a rollout; otherwise recurse
    if N_dict[(s, a)] == 0
        q = r + discount(mdp) * rollout(mdp, rollout_policy, sp, d-1)
    else
        q = r + discount(mdp) * simulate!(mdp, Q_dict, N_dict, t_dict, sp, d-1, c, rollout_policy)
    end

    # Backpropagation: Update Q and N
    N_dict[(s, a)] += 1
    Q_dict[(s, a)] += (q - Q_dict[(s, a)]) / N_dict[(s, a)]  # Incremental mean

    return q
end

# Question 4: Run MCTS for 7 iterations (as required)
m4 = HW3.DenseGridWorld(seed=4)
s0 = SA[19, 19]

n_iterations = 7  # Change this to 100+ to see better Q-value estimates
for i in 1:n_iterations
    simulate!(m4, Q, N, t, s0)
    if i <= 10 || i % 10 == 0
        println("Iteration $i complete")
    end
end

# Visualize the tree
HW3.visualize_tree(Q, N, t, s0)

#-------------
# Question 5
#-------------

function mcts_policy(mdp, s, n_iterations=1000, d=50, c=1.0, rollout_policy=random_policy)
    # Create fresh dictionaries for this planning instance
    Q_local = Dict{Tuple{Any,Symbol}, Float64}()
    N_local = Dict{Tuple{Any,Symbol}, Int}()
    t_local = Dict{Tuple{Any,Symbol,Any}, Int}()

    for _ in 1:n_iterations
        simulate!(mdp, Q_local, N_local, t_local, s, d, c, rollout_policy)
    end

    A = actions(mdp)
    explored_actions = [a for a in A if haskey(Q_local, (s, a))]
    isempty(explored_actions) && return rand(A)

    return argmax(a -> Q_local[(s, a)], explored_actions)
end

m5 = HW3.DenseGridWorld(seed=4)
n_simulations_mcts = 100
n_mcts_iterations = 1000

println("\n=== Question 5: Planning with MCTS ===")

mcts_wrapper(mdp, s) = mcts_policy(mdp, s, n_mcts_iterations, 50, 1.0, random_policy)

# Random initial state
results_mcts_rand = [rollout(m5, mcts_wrapper, rand(initialstate(m5)), 100) for _ in 1:n_simulations_mcts]
mean_mcts_rand = mean(results_mcts_rand)
sem_mcts_rand  = std(results_mcts_rand) / sqrt(n_simulations_mcts)

# Fixed start at (19, 19) like Q4
results_mcts_fixed = [rollout(m5, mcts_wrapper, SA[19, 19], 100) for _ in 1:n_simulations_mcts]
mean_mcts_fixed = mean(results_mcts_fixed)
sem_mcts_fixed  = std(results_mcts_fixed) / sqrt(n_simulations_mcts)

println("Random init:   mean=$(round(mean_mcts_rand, digits=2)),  SEM=$(round(sem_mcts_rand, digits=2))")
println("Fixed (19,19): mean=$(round(mean_mcts_fixed, digits=2)), SEM=$(round(sem_mcts_fixed, digits=2))")

#-------------
# Question 6
#-------------

# Type-stable simulate for Q6 — no t_dict tracking, concrete dict types
function simulate6!(mdp,
                    Q_dict::Dict{Tuple{S,A}, Float64},
                    N_dict::Dict{Tuple{S,A}, Int},
                    s::S, d=20, c=50.0) where {S,A}

    if isterminal(mdp, s) || d <= 0
        return 0.0
    end

    actions_list = actions(mdp)

    # Selection: UCB over all actions
    a = argmax(actions_list) do a_i
        if !haskey(N_dict, (s, a_i)) || N_dict[(s, a_i)] == 0
            return Inf
        end
        N_s = sum(get(N_dict, (s, a_j), 0) for a_j in actions_list)
        Q_dict[(s, a_i)] + c * sqrt(log(N_s) / N_dict[(s, a_i)])
    end

    if !haskey(N_dict, (s, a))
        N_dict[(s, a)] = 0
        Q_dict[(s, a)] = 0.0
    end

    sp, r = @gen(:sp, :r)(mdp, s, a)

    # First visit: heuristic rollout; subsequent: recurse
    q = if N_dict[(s, a)] == 0
        r + discount(mdp) * rollout(mdp, heuristic_policy, sp, d-1)
    else
        r + discount(mdp) * simulate6!(mdp, Q_dict, N_dict, sp, d-1, c)
    end

    N_dict[(s, a)] += 1
    Q_dict[(s, a)] += (q - Q_dict[(s, a)]) / N_dict[(s, a)]

    return q
end

function select_action(m, s)
    S = statetype(m)
    A = actiontype(m)

    # Typed dicts — Julia can compile specialized, fast code for these
    Q6 = Dict{Tuple{S,A}, Float64}()
    N6 = Dict{Tuple{S,A}, Int}()

    start = time_ns()
    while time_ns() < start + 40_000_000  # 40ms budget (leaves 10ms margin)
        simulate6!(m, Q6, N6, s)
    end

    explored = [a for a in actions(m) if haskey(Q6, (s, a))]
    isempty(explored) && return heuristic_policy(m, s)  # fallback

    return argmax(a -> Q6[(s, a)], explored)
end

# --- Approach 1: Fresh tree each call (baseline) ---

m6 = HW3.DenseGridWorld(seed=6)
select_action(m6, SA[35, 35])  # precompile
@btime select_action(m6, SA[35, 35])

# results_q6 = [rollout(m6, select_action, rand(initialstate(m6)), 100) for _ in 1:100]
# println("Approach 1 (fresh tree): mean=$(round(mean(results_q6), digits=2)), SEM=$(round(std(results_q6)/sqrt(100), digits=2))")

# m6 = HW3.DenseGridWorld(seed=6)

# --- Approach 2: Warm start + Multithreaded MCTS ---
# Each thread has its own persistent tree that carries over between steps (warm start).
# Threads work in parallel within the time budget, then trees are merged to pick best action.
# Requires Julia started with: julia --threads auto  (M3 Pro: 6+ performance cores)
# println("\nRunning with $(Threads.nthreads()) threads")

# const _Q_warm_t = Ref{Any}(nothing)
# const _N_warm_t = Ref{Any}(nothing)

# function select_action_warm_threaded(m, s)
#     S = statetype(m)
#     A = actiontype(m)
#     n_t = Threads.nthreads()

#     # Initialize one persistent dict per thread on first call
#     if _Q_warm_t[] === nothing
#         _Q_warm_t[] = [Dict{Tuple{S,A}, Float64}() for _ in 1:n_t]
#         _N_warm_t[] = [Dict{Tuple{S,A}, Int}()     for _ in 1:n_t]
#     end

#     Q_wt = _Q_warm_t[]::Vector{Dict{Tuple{S,A}, Float64}}
#     N_wt = _N_warm_t[]::Vector{Dict{Tuple{S,A}, Int}}

#     deadline = time_ns() + 40_000_000

#     Threads.@threads for i in 1:n_t
#         while time_ns() < deadline
#             simulate6!(m, Q_wt[i], N_wt[i], s)
#         end
#     end

#     # Merge: weighted average of Q by visit count across all threads
#     Q_merged = Dict{Tuple{S,A}, Float64}()
#     N_merged = Dict{Tuple{S,A}, Int}()

#     for i in 1:n_t
#         for (key, n_val) in N_wt[i]
#             n_val == 0 && continue
#             prev_n = get(N_merged, key, 0)
#             prev_q = get(Q_merged, key, 0.0)
#             total_n = prev_n + n_val
#             Q_merged[key] = (prev_n * prev_q + n_val * Q_wt[i][key]) / total_n
#             N_merged[key] = total_n
#         end
#     end

#     explored = [a for a in actions(m) if haskey(Q_merged, (s, a))]
#     isempty(explored) && return heuristic_policy(m, s)

#     return argmax(a -> Q_merged[(s, a)], explored)
# end

# select_action_warm_threaded(m6, SA[35, 35])  # precompile + first warm-up
# _Q_warm_t[] = nothing; _N_warm_t[] = nothing  # reset after precompile

# @btime select_action_warm_threaded(m6, SA[35, 35])

# results_warm_threaded = [rollout(m6, select_action_warm_threaded, rand(initialstate(m6)), 100) for _ in 1:100]
# println("Approach 2 (warm+threaded): mean=$(round(mean(results_warm_threaded), digits=2)), SEM=$(round(std(results_warm_threaded)/sqrt(100), digits=2))")

# --- Submit best approach to grader ---
HW3.evaluate(select_action, "emily.maxwell@colorado.edu", time=true)
# HW3.evaluate(select_action_warm_threaded, "emily.maxwell@colorado.edu", time=true)
