"""
© 2025 Emily Maxwell Outland <emily.maxwell@colorado.edu>
License: BSD-3-Clause

hw1_code.jl

Last Modified: 01/20/2026

Project: ASEN 5264, Decision Making Under Uncertainty, HW1, Q4
Organization: Smead Aerospace, CU Boulder

Description
-----------
Q4: Multiply a vector of vectors by a matrix and return the maximum value (from all the results) at each index in a vector.

Generative AI Disclosure
-------------------------
I had completion assistance from Copilot. I also asked Claude for help with line 53, because I didn't know how to use hcat() with a vector of vectors. (See the comments above line 53 for more details.)
"""

import DMUStudent.HW1

#------------- 
# Question 4
#-------------

"""
    f(a, bs)

Multiply a matrix by each vector in bs, and for each ith index, return the maximum
value at the index from among the result vectors.

# Arguments
- `a::Matrix{Float64}`: The matrix to multiply
- `bs::Vector{Vector{Float64}}`: A vector of vectors to multiply by the matrix

# Returns
- `max_vec::Vector{Float64}`: A vector of the maximum values at each index from among the result vectors
"""
function f(a, bs)
    # Multiply each vector in bs by the matrix a, put into a new vector
    results = [a * b for b in bs]

    # Note to future self: ... is the splat operator, which expands a collection into 
    # individual arguments (it unpacks the vector of vectors) -> I used this but the compiler
    # wasn't able to infer the type properly since it doesn't know how big results is, so I 
    # used reduce instead

    # Note to future self 2: reduce() takes a function and applies it repreatedly to the 
    # elements of a collection, so reduce(hcat, results) is equivalent to hcat(results[1], results[2], ...),
    # which is the same as using ... but the compiler can infer the type properly here

    M = reduce(hcat, results)

    # For each row of the matrix, find the maximum and create a new vector of these maximum vals
    max_vec = [maximum(M[i, :]) for i in 1:length(results[1])]

    return max_vec
end

# You can can test it yourself with inputs like this
a = [2.0 0.0; 0.0 1.0]
@show a
bs = [[1.0, 2.0], [2.0, 1.0]]
@show bs
@show f(a, bs)

# This is how you create the json file to submit
HW1.evaluate(f, "emily.maxwell@colorado.edu")