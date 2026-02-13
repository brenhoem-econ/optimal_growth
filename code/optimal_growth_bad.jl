##Problem 4 
#construct key parameters and initial grids for the analysis
@with_kw struct Primitives 
    β::Float64 = 0.99  #parameters
    θ::Float64 = 0.36 
    δ::Float64 = 0.025
    k_grid::Array{Float64,1} = collect(range(1.0, length = 1000, stop = 45.0))  #capital grid
    nk::Int64 = length(k_grid)  #number of capital elements to loop over
    markov::Array{Float64,2} = [0.977 0.023; 0.074 0.926]  #markov transition matrix
    z_grid::Array{Float64,1} = [1.25, 0.2]  #productivity state grid
    nz::Int64 = length(z_grid) 
end 

#build a MUTABLE struct to hold the model results as it iterates 
mutable struct Results 
    val_func::Array{Float64,2} 
    pol_func::Array{Float64,2} 
end 


#function that executes the model and returns results 
function solve_model()
    prim = Primitives() #initial primitives brought in for use
    val_func = zeros(prim.nk, prim.nz)  #preallocate value function as zero vector
    pol_func = zeros(prim.nk, prim.nz)  #pre allocate policy function as zero vector
    res = Results(val_func, pol_func)  #initialize the results 
    v_iterate(prim, res) #value function iteration
    prim, res  #return deliverables
end 


#value function iteration program. goal is to AVOID INEFFICIENT GLOBAL GARBAGE 
function v_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-10) 
    error = 100 #starting error
    n = 0  #counter 
    while error>tol #main convergence loop 
        n+=1 
        v_next = bellman(prim, res) #execute bellman function 
        error = maximum(abs.(v_next - res.val_func)) #reset error term 
        res.val_func = v_next #update value function in the results vector 
    end 
    println("Value functions converges in ", n, " iterations") 
end 


#bellman operator
function bellman(prim::Primitives, res::Results) 
    @unpack β, δ, θ, nz, nk, z_grid, k_grid, markov = prim  #unpack parameters to improve readability 
    v_next = zeros(nk, nz) 

    for i_k = 1:nk, i_z = 1:nz   #loop over the state space 
        candidate_max = -1e100   #definitely not going to be the maximum 
        k, z = k_grid[i_k], z_grid[i_z]
        budget = z*k^θ + (1 - δ)*k   #budget given values in current state 

        for i_kp = 1:nk  #loop over choice of k_prime 
            kp = k_grid[i_kp] 
            c = budget - kp  #consumption 
            if c>0  #ensure consumption is positive 
                val = log(c) + β * sum(res.val_func[i_kp,:].*markov[i_z, :])
                if val>candidate_max #check for new max value 
                    candidate_max = val 
                    res.pol_func[i_k, i_z] = kp  #update policy function 
                end 
            end 
        end 
        v_next[i_k, i_z] = candidate_max  #update next guess of val function 
    end 
    v_next 
end 

prim, res = solve_model() 

