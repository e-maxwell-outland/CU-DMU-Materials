"""
© 2026 Emily Maxwell Outland <emily.maxwell@colorado.edu>
License: BSD-3-Clause

hw2_code.jl

Last Modified: 02/03/2026

Project: ASEN 5264, Decision Making Under Uncertainty, HW2, Q4-5
Organization: Smead Aerospace, CU Boulder

Description
-----------
Q4: Implement value iteration for a grid world MDP
Q5: Implement value iteration for ACAS (Aircraft Collision Avoidance System) MDP

Generative AI Disclosure
-------------------------
I had completion assistance from Copilot. 

Claude helped me optimize the naive value iteration algorithm by taking advantage of Julia's handling of sparse 
matrices with matrix multiplication (which I didn't know about), rather than iterating over each state and action individually. The
result of that collaboration is the value_iteration function below.

The optimized version, exploit_value_iteration, was developed by Claude (no code writing by me) to exploit the nature of the dynamics. 
See the comment above the function for more information.
"""

using DMUStudent.HW2
using POMDPs: states, actions, discount
using POMDPTools: ordered_states, render
using LinearAlgebra: norm
using Discretizers: encode, bincenters, nlabels
import Cairo, Fontconfig # Needed in some cases for rendering the value function on grid world

# Value Iteration Function
function value_iteration(m)
    n = length(states(m))
    U = zeros(n)  # current utilities
    U_prime = ones(n) # next utilities, doesn't matter, gets overwritten, needs to be ≠ U

    T = transition_matrices(m, sparse=true)
    R = reward_vectors(m)
    γ = discount(m)
    ϵ = 1e-8  # convergence threshold, from homework hint

    while norm(U - U_prime, Inf) > ϵ
        U = copy(U_prime)
        U_prime = fill(-Inf, n)
        for a in actions(m)
            # using a to index operates on all states wrt action a at once
            # and Julia can exploit sparse T in matrix multiplication
            U_prime = max.(U_prime, R[a] + γ * (T[a] * U)) 
        end
    end
    
    return U_prime
end


""" 
Note: The following function is an optimized version of value iteration specifically for Q5, and it was developed
by Claude to exploit the fact that only part of T is needed because only hi' is stochastic and is dependent only on hi. 

Honestly, this was just for fun to see what it could do. My actual work is the previous function, which passes n = 7.
"""
# Structure-exploiting value iteration for UnresponsiveACASMDP.
function exploit_value_iteration(m::UnresponsiveACASMDP)
    # State-space dimensions matching stateindex layout:

    # LinearIndices(n_ho, n_hdot, n_hi, n_d)[ho, hdot, hi, d]
    n_ho   = nlabels(m.hbins)      # 5n
    n_hdot = nlabels(m.hdotbins)   # 5 (fixed)
    n_hi   = n_ho                  # h_i uses the same discretizer as h_o
    n_d    = nlabels(m.dbins)      # 10n
    n_states = n_ho * n_hdot * n_hi * n_d

    γ = discount(m)
    ϵ = 1e-8

    # ---- One-time Precomputation ----

    hcenters    = bincenters(m.hbins)
    hdotcenters = bincenters(m.hdotbins)

    # h_i transition matrix (n_hi × n_hi).
    # Row hi is the probability distribution over next h_i bins given current bin hi.
    # Constructed from the same shifted-CDF logic as POMDPs.transition.
    T_hi = zeros(n_hi, n_hi)
    for hi in 1:n_hi
        start  = n_hi - hi + 1
        finish = start + n_hi - 2
        T_hi[hi, 1]        = m._cached_cdf[start]
        T_hi[hi, 2:n_hi-1] = m._cached_cdf[start+1:finish] .- m._cached_cdf[start:finish-1]
        T_hi[hi, n_hi]     = 1.0 - m._cached_cdf[finish]
    end

    # Deterministic next-hdot: hdot_next[hdot_idx, action_idx]
    acs = actions(m)
    hdot_next = zeros(Int, n_hdot, length(acs))
    for (ai, a) in enumerate(acs)
        for hd in 1:n_hdot
            hdot_next[hd, ai] = encode(m.hdotbins, clamp(hdotcenters[hd] + a, -3000.0, 3000.0))
        end
    end

    # Deterministic next-h_o: ho_next[ho_idx, hdot_next_idx]
    # h_o' = h_o + hdot_o_NEW * dt, snapped to nearest bin.  Indexed by the *next*
    # hdot (not the current one), because the MDP uses the updated hdot_o in h_o'.
    ho_next = zeros(Int, n_ho, n_hdot)
    for ho in 1:n_ho
        for hd in 1:n_hdot
            ho_next[ho, hd] = encode(m.hbins, hcenters[ho] + hdotcenters[hd] * m.dt)
        end
    end

    # Reward vectors — O(n_states) per action, fine to keep in memory
    R = reward_vectors(m)

    # ---- Value Iteration ----
    U       = zeros(n_states)
    U_prime = ones(n_states)

    while norm(U - U_prime, Inf) > ϵ
        U = copy(U_prime)
        U4        = reshape(U, n_ho, n_hdot, n_hi, n_d)
        U_prime_4 = fill(-Inf, n_ho, n_hdot, n_hi, n_d)

        for (ai, a) in enumerate(acs)
            Q_a = reshape(copy(R[a]), n_ho, n_hdot, n_hi, n_d)

            # --- Non-terminal States (d_bin = 1 : n_d-1) ---
            # d advances by exactly one bin each step.  Gather U at the
            # deterministic next (h_o', hdot_o', d+1) for every (ho, hdot) pair.
            U_gathered = zeros(n_ho, n_hdot, n_hi, n_d - 1)
            for ho in 1:n_ho
                for hdot in 1:n_hdot
                    hdn = hdot_next[hdot, ai]
                    hon = ho_next[ho, hdn]
                    U_gathered[ho, hdot, :, :] = U4[hon, hdn, :, 2:n_d]
                end
            end

            # Multiply along h_i (dim 3) by T_hi.
            # Permute h_i to dim 1, flatten the rest into one big matrix,
            # do one BLAS gemm, then undo the reshape.
            UG_perm = permutedims(U_gathered, (3, 1, 2, 4))                    # (n_hi, n_ho, n_hdot, n_d-1)
            EU_flat = T_hi * reshape(UG_perm, n_hi, :)                         # (n_hi, n_ho*n_hdot*(n_d-1))
            EU      = permutedims(reshape(EU_flat, n_hi, n_ho, n_hdot, n_d-1), # back to 4-D
                                  (2, 3, 1, 4))                                # (n_ho, n_hdot, n_hi, n_d-1)

            Q_a[:, :, :, 1:n_d-1] .+= γ .* EU

            # --- Terminal States (d_bin = n_d) ---
            # All states in the last d-slice are terminal; POMDPTools sets T[s,s]=1.
            Q_a[:, :, :, n_d] .+= γ .* U4[:, :, :, n_d]

            U_prime_4 = max.(U_prime_4, Q_a)
        end

        U_prime = vec(U_prime_4)
    end

    return U_prime
end

#------------- 
# Question 4
#-------------

# m = grid_world
# V = value_iteration(m)

# display(render(grid_world, color=V))

#------------- 
# Question 5
#-------------

m = UnresponsiveACASMDP(28)
V = exploit_value_iteration(m)
@show HW2.evaluate(V, "emily.maxwell@colorado.edu")

# Uncomment to use the non-optimized version instead
# m = UnresponsiveACASMDP(7)
# V = value_iteration(m) 
# @show HW2.evaluate(V)