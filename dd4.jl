using Plots
using LogExpFunctions
using SpecialFunctions
using Base.Threads
using Statistics
using Roots
using Optim
#using ForwardDiff
using Optimization, OptimizationOptimJL, OptimizationNLopt, ForwardDiff
using Distributions
using LogExpFunctions
#using Optim.LineSearches
#using FiniteDiff
#using NLopt


function estimate_preservation_rate(observed_proportion::Float64, tt::Dict{Int64,Tuple{Float64,Float64}})
    bin_durations = [tt[i][1] - tt[i][2] for i in 1 : length(tt)]    
    
    if !(0.0 < observed_proportion < 1.0)
        error("Observed proportion must be between 0 and 1.")
    end

    total_duration = sum(bin_durations)
    objective_function(psi) = (sum((1.0 .- exp.(-psi .* bin_durations)) .* bin_durations) / total_duration) - observed_proportion
    
    local estimated_psi
    try
        estimated_psi = find_zero(objective_function, (0.0, 20.0))
    catch e
        error("Could not find a unique solution for psi. Check if the observed proportion is realistic for the given bin durations.")
    end
    return estimated_psi
end


function calc_prob_int_sample(genus_stages::Dict{String,Array{Int64}}, tt::Dict{Int64,Tuple{Float64,Float64}})
    total_time = 0.0
    samp_time  = 0.0
    for ( _ , occ ) in genus_stages
        clade_start, clade_end = get_start_end_bins(occ)
        if clade_end - clade_start  < 2
            continue
        end

        for i in clade_start + 1 : clade_end - 1 
            delta_t = tt[i][1] - tt[i][2]
            if occ[i] > 0
                samp_time += delta_t
            end
            total_time += delta_t 
        end
    end
    samp_prob = samp_time / total_time
    return samp_prob
end

function calc_loglike_pres(phi::Float64, occ_tbl::Dict{String,Array{Int64}}, tt::Dict{Int64,Tuple{Float64,Float64}})
    ll = 0.0
    for (genus, occ) in occ_tbl
        curll = calc_loglike_pres_single(occ, tt, phi)
        ll += curll
    end
    return ll
end

function calc_loglike_pres_single(occ_range::Array{Int64}, tt::Dict{Int64,Tuple{Float64,Float64}}, phi::Float64)#::Float64
    orig_bin, ext_bin = get_start_end_bins(occ_range)
    orig_dt = tt[orig_bin][1] - tt[orig_bin][2]
    if orig_bin == ext_bin
        ll = log( 1 - ( ( 1 - exp( -phi * orig_dt ) ) / ( phi * orig_dt ) ) )
        ll = 0
    else
        logp_pres =  log( 1 - ( ( 1 - exp(-phi * orig_dt) ) / ( phi * orig_dt ) ) ) # expression for K > 0
        logp_pres = log( 1 - exp(-phi * 1.0 * orig_dt ) ) 
        logp_inter = 0.0
        if abs(orig_bin - ext_bin) > 1
            logp_inter = calc_intermed_ll(orig_bin, ext_bin, occ_range, tt, phi)
        end
        ext_dt = tt[ext_bin][1] - tt[ext_bin][2]
        if ext_bin != length(tt) # the extinction bin is before the present
            cur_lpp = log( 1 - ( ( 1.0 - exp(-phi * ext_dt ) ) / ( phi * ext_dt ) ) ) # expression for K>0
            cur_lpp = log( 1 - exp(-phi * 1.0 * ext_dt ) ) 
        else # the taxon survived to the Recent
            cur_lpp = log( 1 - exp(-phi * ext_dt ) ) 
        end
        #ll = logp_pres + logp_inter + cur_lpp 
        ll = logp_inter
    end
    return ll
end

function calc_intermed_ll(clade_start::Int64, clade_end::Int64, occ_range::Array{Int64}, tt::Dict{Int64,Tuple{Float64,Float64}}, phi::Float64)#::Float64
    ll_pres = 0.0
    for i in clade_start + 1 : clade_end - 1
        delta_t = tt[i][1] - tt[i][2]
        n_foss = occ_range[i]
        if n_foss == 0
            log_p_pres = -phi * delta_t
        else
            log_p_pres = log( 1 - exp( -phi * delta_t ) )
        end
        ll_pres += log_p_pres 
    end
    return ll_pres
end

function calc_surv_ll(clade_start::Int64, clade_end::Int64, occ_range::Array{Int64}, bin_mu::Array{Float64}, phi::Float64, tt::Dict{Int64,Tuple{Float64,Float64}})#::Float64
    ll_surv = 0.0
    for i in clade_start + 1 : clade_end - 1
        delta_t = tt[i][1] - tt[i][2]
        n_foss = occ_range[i]
        mu_t  = bin_mu[i]
        log_p_surv = -mu_t * delta_t 
        if n_foss == 0
            log_p_pres = -phi * delta_t
        else
            #pois_rate = phi * delta_t
            #log_p_pres = calc_log_poisson_pmf(pois_rate, n_foss) 
            log_p_pres = log( 1 - exp( -phi * delta_t ) )
        end
        part_ll = log_p_pres + log_p_surv
        ll_surv += part_ll
    end
    return ll_surv
end

function read_in_tt(infl::String)::Dict{Int64,Tuple{Float64,Float64}}
    stage_times = Dict{Int64,Tuple{Float64,Float64}}()
    fl = open(infl,"r")
    lines = readlines(fl)
    for (i, line) in pairs(lines[2:length(lines)])
        spls = split(strip(line),",")
        quot = "\""
        #cur_stage = replace(spls[1],quot=>"")
        upper = parse(Float64,replace(spls[3],quot=>""))
        lower = parse(Float64,replace(spls[2],quot=>""))
        #bin_no = parse(Int64,replace(spls[5],quot=>""))
        bin_no = i 
        #println(bin_no)
        bounds = (lower,upper)
        stage_times[bin_no] = bounds
    end
    #for key in sort(collect(keys(stage_times)))
    #    value = stage_times[key]
    #    println("$key, $value")
    #end
    close(fl)
    return stage_times
end

function read_in_occurrences(infl::String, tax_level::String = "genus")::Dict{String,Array{Float64}}
    occ_dict = Dict{String,Array{Float64}}()
    fl = open(infl,"r")
    lines = readlines(fl)
    for (i,line) in pairs(lines[2:length(lines)])
        spls = split(strip(line),",")
        quot = "\""
        if tax_level == "genus"
            gen_nm = replace(string(split(strip(spls[6])," ")[1]),quot=>"")
        elseif tax_level == "species"
            #println(replace(replace(spls[6]," "=>"_"),quot=>""))
            gen_nm = replace(string(split(strip(spls[6])," ")[1]),quot=>"")
        end
        lower = parse(Float64,replace(spls[11],quot=>""))
        upper = parse(Float64,replace(spls[12],quot=>""))
        #samp = (rand()*(lower-upper))+upper
        samp = (lower+upper) / 2.0
        if haskey(occ_dict,gen_nm) == false
            occ_dict[gen_nm]=Float64[]
        end
        append!(occ_dict[gen_nm], samp)
    end
    #for key in sort(collect(keys(occ_dict)))
    #    value = occ_dict[key]
    #    println("$key, $value")
    #end
    close(fl)
    return occ_dict
end

function find_occ_bin(occ_time::Float64, tt::Dict{Int64,Tuple{Float64,Float64}})::Int64
    for (key, times) in tt
        if occ_time <= times[1] && occ_time >= times[2]
            return key
        end
    end
end

function create_genus_occurrence_table(occ_dict::Dict{String,Array{Float64}}, tt::Dict{Int64,Tuple{Float64,Float64}})::Dict{String,Array{Int64}}
    genus_stages = Dict{String,Array{Int64}}()
    for (genus, occurrences) in occ_dict
        stage_occ = zeros(Int64, length(tt)) 
        for i in eachindex(occurrences)
            occ = occurrences[i]
            bin = find_occ_bin(occ, tt)
            stage_occ[bin] += 1
        end
        #println("$genus $stage_occ")
        genus_stages[genus] = stage_occ
    end
    for stage in 1: length(tt)
        stage_occ = 0
        stage_tax = 0
        for (genus, occ) in genus_stages
            if occ[stage] > 0
                stage_tax += 1
                stage_occ += occ[stage]
            end
        end
        #println(stage," ",  stage_occ / stage_tax)
    end  
    return genus_stages
end

function get_start_end_bins(n_obs::Array{Float64})::Tuple{Int64,Int64}
    clade_start = 1
    for i in eachindex(n_obs)
        n_gen = n_obs[i]
        if n_gen > 0 
            clade_start = i
            break
        end
    end

    clade_end = 1
    for i in reverse(eachindex(n_obs))
        n_gen = n_obs[i]
        if n_gen > 0 
            clade_end = i
            break
        end
    end
    return clade_start, clade_end
end

function get_start_end_bins(n_obs::Array{Int64})::Tuple{Int64,Int64}
    clade_start = 1
    for i in eachindex(n_obs)
        n_gen = n_obs[i]
        if n_gen > 0 
            clade_start = i
            break
        end
    end

    clade_end = 1
    for i in reverse(eachindex(n_obs))
        n_gen = n_obs[i]
        if n_gen > 0 
            clade_end = i
            break
        end
    end
    return clade_start, clade_end
end

function count_three_timers(genus_stages::Dict{String,Array{Int64}})
    nbin = length(first(values(genus_stages)))
    bc = zeros(Int64, nbin)
    for i in 2 : nbin
        for ( _, occ ) in genus_stages
            if occ[i] > 0 && occ[i - 1] > 0 && occ[i + 1] > 0
                bc[i] += 1
            end            
        end
    end
    return bc

end

function count_obs_txc(genus_stages::Dict{String,Array{Int64}})
    nbin = length(first(values(genus_stages)))
    bc = zeros(Int64, nbin)
    for i in 2 : nbin
        for ( _, occ ) in genus_stages
            if occ[i] > 0 && occ[i - 1] > 0
                bc[i] += 1
            end            
        end
    end
    return bc
end

function calc_corrected_txc(genus_stages::Dict{String,Array{Int64}}, obs_Nt::Array{Int64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi_v::Array{Float64}, starting_div = 1)
    corr_Nt = calc_corrected_Nt(obs_Nt, tt, psi_v[1], starting_div)
    nbin = length(tt)
    obs_txc = count_obs_txc(genus_stages)
    corr_txc = zeros(Float64, nbin)
    stop_bin = nbin
    for i in nbin : -1 : 1
        if obs_txc[i] > 0
            stop_bin = i
            break
        end
    end
    start = false
    for i in 2 : stop_bin
        psi = psi_v[i]
        #muk = bin_mu[i]
        o = obs_txc[i] 
        if o > 0 && start == false
            corr_txc[i - 1] = starting_div
            start = true
        end
        if start == true
            delta_t = tt[i][1] - tt[i][2]
            if delta_t < 1
                delta_t = 1.0
            end
            exp_surv = 0.99
            if i < length(obs_Nt)
                samp_prob1 = 1.0 - exp(-psi * ( delta_t * exp_surv ) ) 
            else # assume extant sampling fraction of 1.0
                samp_prob1 = 1.0
            end
            if o == 0
                o = 1
            end
            delta_t = tt[i-1][1] - tt[i-1][2]
            if delta_t < 1
                delta_t = 1.0
            end
            psi = psi_v[i-1]
            samp_prob2 = 1.0 - exp(-psi * ( delta_t * exp_surv) ) 
            #txc_prob = samp_prob1 * samp_prob2
            #println(samp_prob1, " ", samp_prob2, " ", samp_prob1 * samp_prob2)
            txc_prob = sqrt(samp_prob1 * samp_prob2)
            #println((samp_prob1 + samp_prob2) / 2)
            #println(txc_prob)
            corr = o / txc_prob
            delta_t = tt[i][1] - tt[i][2]
            corr_txc[i] = corr 
        end
    end
    
    #=
    mv = mean(corr_txc) 
    mnt = mean(corr_Nt) 
    sf = mnt / mv 
    for i in eachindex(corr_txc)
        sv = corr_txc[i] * sf
        #if sv > ( mnt * 2 )
        #    sv = mnt * 2
        #end 
        corr_txc[i] = sv 
    end
    =#

    t = 1:length(corr_txc) #- 1
    at = [-tt[i][1] for i in t]
    mass_extinctions = Float64[-443.8, -372.2, -251.9, -201.3, -66.0]
    plt = plot(at,corr_txc,label = ["corr_txc"])
    plot!(plt, at, obs_txc, ls=:dot, label = ["obs_txc"])
    vline!(mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1.5, label="")
    savefig(plt,"dtt.png")
    
    plt2 = plot(at, log.(corr_txc),label = ["log_txc"])
    plot!(plt2, at, log.(obs_txc), ls=:dot, label = ["log_obs_txc"])
    vline!(mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1.5, label="")
    savefig(plt2,"log_dtt.png")
    return corr_txc
end

function calc_moving_average(corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, use_log::Bool=false)
    nbin = length(corr_Nt)
    v = Float64[corr_Nt[1]]
    for i in 2 : nbin - 1
        #n = Float64[corr_Nt[i - 1], corr_Nt[i], corr_Nt[i + 1]]
        if use_log
            n = Float64[log(corr_Nt[i - 1]), log(corr_Nt[i]), log(corr_Nt[i + 1])]
            a = exp(sum(n) / length(n))
        else
            n = Float64[corr_Nt[i - 1], corr_Nt[i], corr_Nt[i + 1]]
            a = sum(n) / length(n)
        end
        append!(v, a)
    end  
    append!(v, corr_Nt[end])

    t = 1:length(v) #- 1
    at = [-tt[i][1] for i in t]
    mass_extinctions = Float64[-443.8, -372.2, -251.9, -201.3, -66.0]
    plt = plot(at, v,label = ["avg"])
    plot!(plt, at, corr_Nt, ls=:dot, label = ["corr_Nt"])
    vline!(mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1.5, label="")
    savefig(plt,"3_bin_dtt.png")
    return v
end

function calc_corrected_Nt(obs_Nt::Array{Int64}, tt::Dict{Int64,Tuple{Float64,Float64}}, phi::Float64, starting_div = 1)::Array{Float64}
    corr_Nt = Float64[]
    clade_start, clade_end = get_start_end_bins(obs_Nt)
    #println(length(obs_Nt), length(tt))
    #exit(86)
    start = false
    for i in eachindex(obs_Nt)
        o = obs_Nt[i]
        delta_t = tt[i][1] - tt[i][2]
        if i < length(obs_Nt)
            samp_prob = 1.0 - exp(-phi * delta_t ) 
        else # assume extant sampling fraction of 1.0
            samp_prob = 1.0
        end
        #println(samp_prob)
        if o != 0
            #=
            if start == false
                last_div = starting_div  # assume diversity was at 1 in the previous interval
                start = true
            else
                last_div = corr_Nt[i - 1]
            end
            =#
            val =  o / samp_prob
            #div_est = ( last_div + val ) / 2.0
            push!(corr_Nt, val)
        else
            #push!(corr_Nt, round( o / samp_prob ) )
            if i >= clade_start && i <= clade_end
                push!(corr_Nt, 1 / samp_prob)
            else
                push!(corr_Nt, 0)
            end
        end
    end
    #println(corr_txc)
    #exit()
    #=
    t = 1:length(corr_Nt) #- 1
    at = [-tt[i][1] for i in t]
    mass_extinctions = Float64[-443.8, -372.2, -251.9, -201.3, -66.0]
    plt = plot(at,corr_Nt,label = ["Nt"])
    plot!(plt, at, obs_Nt, ls=:dot, label = ["obs_Nt"])
    vline!(mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1, label="")
    #display(plt) 
    savefig(plt,"dtt.png")

    plt2 = plot(at, log.(corr_Nt),label = ["log_Nt"])
    plot!(plt2, at, log.(obs_Nt), ls=:dot, label = ["log_obsNt"])
    vline!(mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1, label="")
    savefig(plt2,"log_dtt.png")
    =#

    return corr_Nt
end


#function calc_dd_rates_regimes(params::Matrix{Float64}, corr_Nt::Array{Float64}, regimes::Array{Int64}, log_div = false)
function calc_dd_rates_regimes(params::AbstractMatrix, corr_Nt::Array{Float64}, regimes::Array{Int64}, log_div = false)
    K_v = params[1, :]
    eqr0_v = params[2, :]
    alpha_v = params[3, :]

    if log_div == true
        lam0_v = [ ( eqr0_v[i] + ( alpha_v[i] * log( K_v[i] ) ) ) for i in eachindex(K_v)]
        mu0_v  = [ ( eqr0_v[i] - ( alpha_v[i] *  log( K_v[i] ) ) ) for i in eachindex(K_v) ]
    else
        #=
        lam0_v = Float64[]
        mu0_v  = Float64[]
        for i in eachindex(K_v)
            if K_v[i] > 10
                lam_val = eqr0_v[i] + ( alpha_v[i] * log( K_v[i] ) ) 
                mu_val  = eqr0_v[i] - ( alpha_v[i] * log( K_v[i] ) ) 
            else
                lam_val = eqr0_v[i] + ( alpha_v[i] *  K_v[i] ) 
                mu_val  = eqr0_v[i] - ( alpha_v[i] *  K_v[i] ) 
            end
            append!(lam0_v, lam_val)
            append!(mu0_v, mu_val)
        end
        =#
        lam0_v = [ ( eqr0_v[i] + ( alpha_v[i] * K_v[i]  ) ) for i in eachindex(K_v)]
        mu0_v  = [ ( eqr0_v[i] - ( alpha_v[i] *  K_v[i]  ) ) for i in eachindex(K_v) ]
    end
    
    for i in eachindex(mu0_v)
        if mu0_v[i] < 0.00001
            mu0_v[i] = 0.00001
        end
    end
    bin_lam = Float64[]
    bin_mu  = Float64[]
    #T = eltype(params)
    #bin_lam = Vector{T}(undef, length(corr_Nt))
    #bin_mu  = Vector{T}(undef, length(corr_Nt))
    for i in eachindex(corr_Nt)
        nt = corr_Nt[i]
        curreg = regimes[i]
        if nt > 0
            if log_div == true
                lam_t = lam0_v[curreg] - ( alpha_v[curreg] *  log( nt ) ) 
            else
                lam_t = lam0_v[curreg] - ( alpha_v[curreg] *  nt  ) 
            end
            #lam_t = lam0 - ( alpha * nt ) 
            if lam_t < 0
                lam_t = 0.0001 # TODO: is this constraint the best way to handle this?
            end
            if log_div == true
                mu_t  = mu0_v[curreg]  + ( alpha_v[curreg] *  log( nt )  )
            else
                mu_t  = mu0_v[curreg]  + ( alpha_v[curreg] *  nt  )
                #println(alpha_v[curreg], " $nt ", mu_t)
            end
        else # if zero extant members TODO: may want to change this assumption 
            lam_t = 0.00001
            mu_t  = 0.00001
        end
        #println(nt, " ", lam_t , " " , mu_t)
        append!(bin_lam, lam_t)    
        append!(bin_mu,  mu_t)    
    end

    #=
    at = [-tt[i][1] for i in 1:length(bin_lam)]
    mass_extinctions = Float64[-443.8, -372.2, -251.9, -201.3, -66.0]
    plt = plot(1:length(bin_lam), bin_lam,label = ["lam_k"])
    plt = plot(at, bin_lam,label = ["lam_k"])
    plot!(plt, at, bin_mu, ls=:dot, label = ["mu_k"])
    vline!(mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1, label="")
    savefig(plt,"rates.png")
    =#
    return bin_lam, bin_mu
end


#function calc_dd_rates(params::Array{Float64}, corr_Nt::Array{Float64}) ## TODO: SWAP THIS BACK. tt is just for debugging
function calc_dd_rates(params::Array{Float64}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}})
    lam0 = params[1]
    mu0  = params[2]
    alpha = params[3]
    bin_lam = Float64[]
    bin_mu  = Float64[]
    for nt in corr_Nt
        if nt > 0
            lam_t = lam0 - ( alpha * log( nt ) ) 
            #lam_t = lam0 - ( alpha * nt ) 
            if lam_t < 0
                lam_t = 0.0001 # TODO: is this constraint the best way to handle this?
            end
            mu_t  = mu0  + ( alpha * log( nt ) )
        else # if zero extant members TODO: may want to change this assumption 
            lam_t = 0.00001
            mu_t  = 0.00001
        end
        #println(nt, " ", lam_t , " " , mu_t)
        append!(bin_lam, lam_t)    
        append!(bin_mu,  mu_t)    
    end


    #=
    at = [-tt[i][1] for i in 1:length(bin_lam)]
    mass_extinctions = Float64[-443.8, -372.2, -251.9, -201.3, -66.0]
    plt = plot(1:length(bin_lam), bin_lam,label = ["lam_k"])
    plt = plot(at, bin_lam,label = ["lam_k"])
    plot!(plt, at, bin_mu, ls=:dot, label = ["mu_k"])
    vline!(mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1, label="")
    savefig(plt,"rates.png")
    =#
    return bin_lam, bin_mu
end

function check_rates_sign(lam_vals::Array{Float64}, mu_vals::Array{Float64})::Bool
    for i in eachindex(lam_vals)
        if lam_vals[i] <= 0.0 || mu_vals[i] <= 0.0
            #print(lam_vals[i], mu_vals[i])
            return true
        end
    end
    return false
end

function calc_range_dtt(genus_stages::Dict{String,Array{Int64}})::Array{Float64}
    n_obs = zeros(Int64, length(first(values(genus_stages))))
    for (genus, stages) in genus_stages
        clade_start, clade_end = get_start_end_bins(stages)
        for i in eachindex(stages)
            if i >= clade_start && i <=clade_end
                n_obs[i] += 1.0
            end
        end
    end
    return n_obs
end


function get_num_obs_stages(genus_stages::Dict{String,Array{Int64}})::Array{Int64}
    n_obs = zeros(Int64, length(first(values(genus_stages))))
    for (genus, stages) in genus_stages
        for i in eachindex(stages)
            n_occ = stages[i]
            if n_occ > 0
                n_obs[i] += 1
            end
        end
    end

    clade_start, clade_end = get_start_end_bins(n_obs)

    for i in eachindex(n_obs)
        n_gen = n_obs[i]
        if n_gen == 0 && i > clade_start && i < clade_end 
            n_obs[i] = 1
        end
    end

    return n_obs
end

function three_bin_origination_ll(bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, bin_lam::AbstractVector, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64})
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    ll = 0.0
    for i in clade_end : -1 : clade_start + 2
        counts = bcounts[i-2] 
        probs = calc_category_origination_probs(i, bin_lam, psi, tt, clade_start)
        for ind in eachindex(counts)
            count = counts[ind]
            curll = log( probs[ind] ) * count 
            ll += curll
        end
    end
    return ll
end



function calc_category_origination_probs(i::Int64, bin_lam::AbstractVector, psi::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, clade_start::Int64 = 1)
    j = i - 1
    k = i - 2
    dt_i = tt[i][1] - tt[i][2]
    dt_j = tt[j][1] - tt[j][2]
    dt_k = tt[k][1] - tt[k][2]
    Pi = 1 - exp( -bin_lam[i] * dt_i )
    Pj = 1 - exp( -bin_lam[j] * dt_j )
    Pk = 1 - exp( -bin_lam[k] * dt_k )
    Rj = 1 - exp( -psi[j] * dt_j )
    Rk = 1 - exp( -psi[k] * dt_k )
    if k != 0 # clade_start 
        p100 = Pi + ( ( 1 - Pi ) * Pj * ( 1 - Rj ) ) + ( ( 1 - Pi ) * ( 1 - Pj ) * ( 1 - Rj ) * ( 1 - Rk ) )
        p101 = ( 1 - Pi ) * ( 1 - Pj ) * ( 1 - Rj ) * Rk
        p110 = ( ( 1 - Pi ) * Pj * Rj ) + ( ( 1 - Pi ) * ( 1 - Pj ) * Rj * ( 1 - Rk ) ) 
        p111 = ( 1 - Pi ) * ( 1 - Pj ) * Rj * Rk
    else
        p100 = Pi + ( ( 1 - Pi ) * Pj * ( 1 - Rj ) ) + ( ( 1 - Pi ) * ( 1 - Pj ) * ( 1 - Rj ) * ( 1 - Rk ) ) * Pk
        p101 = ( 1 - Pi ) * ( 1 - Pj ) * ( 1 - Rj ) * Rk * Pk
        p110 = ( ( 1 - Pi ) * Pj * Rj ) + ( ( 1 - Pi ) * ( 1 - Pj ) * Rj * ( 1 - Rk ) ) * Pk  
        p111 = ( 1 - Pi ) * ( 1 - Pj ) * Rj * Rk * Pk
    end
    #println("$p100 $p101 $p110 $p111")
    probs = [p100, p101, p110, p111]
    return probs
end


function three_bin_origination_ll_gamma(bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, bin_lam::AbstractVector, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64})
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    ll = 0.0
    for i in clade_end : -1 : clade_start + 2
        counts = bcounts[i-2] 
        logprobs = calc_category_origination_probs_gamma(i, bin_lam, psi, tt)
        for ind in eachindex(counts)
            count = counts[ind]
            curll = logprobs[ind] * count 
            ll += curll
        end
    end
    return ll
end


function calc_category_origination_probs_gamma(i::Int64, bin_lam::AbstractVector, psi::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}})
    j = i - 1
    k = i - 2
    dt_i = tt[i][1] - tt[i][2]
    dt_j = tt[j][1] - tt[j][2]
    dt_k = tt[k][1] - tt[k][2]
    Pi = 1 - exp( -bin_lam[i] * dt_i )
    Pj = 1 - exp( -bin_lam[j] * dt_j )
    Pk = 1 - exp( -bin_lam[k] * dt_k )
    count = 0
    L100 = -Inf
    L101 = -Inf
    L110 = -Inf
    L111 = -Inf
    K = length(psi)
    #w = 1 / ( K * K )
    logw = -2 * log(K)  
    for psi_i in psi
        Rj = 1 - exp( -psi_i * dt_j )
        for psi_j in psi
            Rk = 1 - exp( -psi_j * dt_k )
            p100 =  Pi + ( ( 1 - Pi ) * Pj * ( 1 - Rj ) ) + ( ( 1 - Pi ) * ( 1 - Pj ) * ( 1 - Rj ) * ( 1 - Rk ) ) 
            p101 = ( 1 - Pi ) * ( 1 - Pj ) * ( 1 - Rj ) * Rk 
            p110 = ( ( 1 - Pi ) * Pj * Rj ) + ( ( 1 - Pi ) * ( 1 - Pj ) * Rj * ( 1 - Rk ) ) 
            p111 = ( 1 - Pi ) * ( 1 - Pj ) * Rj * Rk 
            L100 = logsumexp(L100, logw + log(p100))
            L101 = logsumexp(L101, logw + log(p101))
            L110 = logsumexp(L110, logw + log(p110))
            L111 = logsumexp(L111, logw + log(p111))
            count += 1
            #println("$p100 $p101 $p110 $p111")
        end
    end
    #println("LIKELIHOODS")
    #println("$L100 $L101 $L110 $L111")
    #println([exp(i) for i in [L100, L101, L110, L111]])
    #exit()
    probs = [L100, L101, L110, L111]
    return probs
end



function count_ntax_3cats(genus_stages::Dict{String,Array{Int64}}, i::Int64, back::Bool = false)::Array{Int64}
    if back == false
        j = i + 1
        k = i + 2
    else
        j = i - 1
        k = i - 2
    end
    n100 = 0
    n101 = 0
    n110 = 0
    n111 = 0

    for (_, occ) in genus_stages
        if occ[i] == 0
            continue
        end
        if occ[j] == 0 && occ[k] == 0
            n100 += 1
        elseif occ[j] == 0 && occ[k] > 0
            n101 += 1
        elseif occ[j] > 0 && occ[k] == 0
            n110 += 1
        elseif occ[j] > 0 && occ[k] > 0
            n111 += 1
        end
    end
    counts = Int64[n100, n101, n110, n111]
    return counts 
end


function three_bin_extinction_ll_gamma(fcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, bin_mu::AbstractVector, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64})
    #clade_start, clade_end = get_start_end_bins(corr_Nt)
    ll = 0.0
    #for i in clade_start : clade_end - 2
    for i in eachindex(fcounts)
        counts = fcounts[i] 
        logprobs = calc_category_extinction_probs_gamma(i, bin_mu, psi, tt)
        for ind in eachindex(counts)
            count = counts[ind]
            curll = logprobs[ind] * count 
            ll += curll
        end
    end
    return ll
end

function calc_category_extinction_probs_gamma(i::Int64, bin_mu::AbstractVector, psi::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}})
    j = i + 1
    k = i + 2
    dt_i = tt[i][1] - tt[i][2]
    dt_j = tt[j][1] - tt[j][2]
    dt_k = tt[k][1] - tt[k][2]
    Qi = log1mexp( -bin_mu[i] * dt_i)
    Qj = log1mexp( -bin_mu[j] * dt_j)

    count = 0
    L100 = -Inf
    L101 = -Inf
    L110 = -Inf
    L111 = -Inf
    K = length(psi)
    logw = -2 * log(K)  
    for psi_i in psi
        Rj = log1mexp( -psi_i * dt_j )
        for psi_j in psi
            Rk = log1mexp( -psi_j * dt_k )
            #Rj = 1-exp( -psi_i * dt_j )
            #Rk = 1-exp( -psi_j * dt_k )
            #Qi = 1-exp( -bin_mu[i] * dt_i)
            #Qj = 1-exp( -bin_mu[j] * dt_j)
            #p100 = Qi + ( ( 1 - Qi ) * Qj * ( 1 - Rj ) ) + ( ( 1 - Qi ) * ( 1 - Qj ) * ( 1 - Rj ) * ( 1 - Rk ) )
            #println(log(p100))

            #println("probs: $Rj $Rk $Qi $Qj")
            #println([Qi, ( log1mexp( Qi ) + Qj + log1mexp( Rj ) ), ( log1mexp( Qi ) + log1mexp( Qj ) + log1mexp( Rj ) + log1mexp( Rk ) )])
            p100 = logsumexp([Qi, ( log1mexp( Qi ) + Qj + log1mexp( Rj ) ), ( log1mexp( Qi ) + log1mexp( Qj ) + log1mexp( Rj ) + log1mexp( Rk ) )])
            #p101 = ( 1 - Qi ) * ( 1 - Qj ) * ( 1 - Rj ) * Rk
            p101 = log1mexp(  Qi ) + log1mexp( Qj ) + log1mexp( Rj ) + Rk
            #p110 = ( ( 1 - Qi ) * Qj * Rj ) + ( ( 1 - Qi ) * ( 1 - Qj ) * Rj * ( 1 - Rk ) ) 
            p110 = logsumexp([( log1mexp( Qi ) + Qj + Rj ), ( log1mexp( Qi ) + log1mexp( Qj ) + Rj + log1mexp( Rk ) )])
            #p111 = ( 1 - Qi ) * ( 1 - Qj ) * Rj * Rk
            p111 = log1mexp( Qi ) + log1mexp( Qj ) + Rj + Rk
            
            L100 = logsumexp(L100, logw + p100)
            L101 = logsumexp(L101, logw + p101)
            L110 = logsumexp(L110, logw + p110)
            L111 = logsumexp(L111, logw + p111)
            count += 1
            #println("$p100 $p101 $p110 $p111")
        end
    end
    #println("LIKELIHOODS")
    #println(bin_mu[i]," ", bin_mu[j])
    #println(Qi)
    #println("EXT LIKES $L100 $L101 $L110 $L111")
    #println([exp(i) for i in [L100, L101, L110, L111]])
    #exit()
    probs = [L100, L101, L110, L111]
    return probs
end

function calc_aic(ll::Float64, k::Int64)
    aic = 2*k - 2*ll
    return aic
end

function calc_aicc(ll::Float64, k::Int64, n::Int64)
    aic = 2*k - 2*ll
    correction = (2 * k * (k + 1)) / (n - k - 1)
    aicc = aic + correction
    return aicc
end

function calc_bic(ll::Float64, k::Int64, n::Int64)
    bic = k * log(n) - 2*ll
    return bic
end

function calc_Nb_from_rates(bin_lam::Array{Float64}, bin_mu::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, corr_Nt::Array{Float64}, starting_size = 1)
    last_div = starting_size
    div = Float64[last_div]
    for i in eachindex(bin_lam)
        #curNb = last_div * exp()
        corr_val = corr_Nt[i]
        curdt = tt[i][1] - tt[i][2]
        #curNb = last_div * ( 1 + ( ( ( bin_lam[i] + 0.01) - bin_mu[i] ) * curdt ) )
        curNb = last_div * exp((bin_lam[i] - bin_mu[i]) * curdt)
        sf = corr_val / curNb
        if i < length(bin_lam)
            append!(div, (curNb * sf) - corr_val)
        end
        curNb = last_div
    end
    println(div)
    #println(corr_Nt)
end

function calc_Lmy(occ_dict::Dict{String,Array{Float64}})
    lmy = 0.0
    for (_, occ) in occ_dict
        dur = maximum(occ) - minimum(occ)
        lmy += dur 
    end
    return lmy 
end


function calc_loglike_gamma(fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, bin_lam::AbstractVector, bin_mu::AbstractVector, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64}, shape::Float64)
    #println(shape, " ", psi[1])
    gamma = Gamma(shape, psi[1])
    K = 4
    p_mid = ((1:K) .- 0.5) ./ K
    medians = quantile.(Ref(gamma), p_mid)

    orig_ll = three_bin_origination_ll_gamma(bcounts, corr_Nt, bin_lam, tt, medians)
    ext_ll = three_bin_extinction_ll_gamma(fcounts, corr_Nt, bin_mu, tt, medians)
    #println("sublikes $orig_ll $ext_ll")
    ll = orig_ll + ext_ll
    return ll
end

function evaluate_one_rate_ll_gamma(params::Matrix{Float64}, regimes::Array{Int64}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64}, shape::Float64)
    lams = params[1, :]
    mus  = params[2, :]
    bin_lam = Float64[]
    bin_mu  = Float64[]
    for reg in regimes
        append!(bin_lam, lams[reg])
        append!(bin_mu,   mus[reg])
    end
    ll = calc_loglike_gamma(fcounts, bcounts, corr_Nt, bin_lam, bin_mu, tt, psi, shape) 
    return ll
end

function obj_function_single_gamma(x::Vector{Float64}, regimes::Array{Int64}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64})
    LARGE = 100000000000.0
    for i in x
        if i < 0.00001
            return LARGE
        end
    end

    unique_reg = unique(regimes)
    highest = maximum(unique_reg)
    lams = x[1:highest]
    mus = x[highest+1:end-1]
    shape = x[end] / 0.01
    params = vcat(lams', mus')
    ll = evaluate_one_rate_ll_gamma(params, regimes, fcounts, bcounts, corr_Nt, tt, psi, shape)
    #println("CURLL ",ll)
    return -ll
end


#function calc_loglike(occ_tbl::Dict{String,Array{Int64}}, corr_Nt::Array{Float64}, bin_lam::Array{Float64, 1}, bin_mu::Array{Float64, 1}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64})
function calc_loglike(occ_tbl::Dict{String,Array{Int64}}, corr_Nt::Array{Float64}, bin_lam::AbstractVector, bin_mu::AbstractVector, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64})
    orig_ll = three_bin_origination_ll(occ_tbl, corr_Nt, bin_lam, tt, psi)
    ext_ll = three_bin_extinction_ll(occ_tbl, corr_Nt, bin_mu, tt, psi)
    #println("ext: ", ext_ll)
    #println("orig: ", orig_ll)
    ll = orig_ll + ext_ll
    return ll
end

function evaluate_one_rate_ll(params::Matrix{Float64}, regimes::Array{Int64}, occ_tbl::Dict{String,Array{Int64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64})
    lams = params[1, :]
    mus  = params[2, :]
    bin_lam = Float64[]
    bin_mu  = Float64[]
    for reg in regimes
        append!(bin_lam, lams[reg])
        append!(bin_mu,   mus[reg])
    end
    ll = calc_loglike(occ_tbl, corr_Nt, bin_lam, bin_mu, tt, psi)
    return ll
end

function obj_function_single(x::Vector{Float64}, regimes::Array{Int64}, occ_tbl::Dict{String,Array{Int64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64} )
    LARGE = 100000000000.0
    for i in x
        if i < 0.0
            return LARGE
        end
    end

    unique_reg = unique(regimes)
    highest = maximum(unique_reg)
    lams = x[1:highest]
    mus = x[highest+1:end]
    params = vcat(lams', mus')
    ll = evaluate_one_rate_ll(params, regimes, occ_tbl, corr_Nt, tt, psi)
    return -ll
end

function evaluate_dd_ll_gamma(params::AbstractMatrix, shape::Float64, regimes::Array{Int64}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64}, log_div = false)
    bin_lam, bin_mu = calc_dd_rates_regimes(params, corr_Nt, regimes, log_div)
    ll = calc_loglike_gamma(fcounts, bcounts, corr_Nt, bin_lam, bin_mu, tt, psi, shape) 
    return ll
end

function obj_function_dd_gamma(x::AbstractVector, Ks::Vector{Float64}, regimes::Array{Int64}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64} , log_div = false)
    LARGE = 100000000000.0
    for i in x
        if i < 0.0
            return LARGE
        end
    end
    
    unique_reg = unique(regimes)
    highest = maximum(unique_reg)
    #x_float = Vector{Float64}(x)
    eqrs = x[1:highest]
    alphas = x[highest+1:end-1]
    shape = x[end] / .01
    params = vcat(Ks', eqrs', alphas')
    ll = evaluate_dd_ll_gamma(params, shape, regimes, fcounts, bcounts, corr_Nt, tt, psi, log_div)
    return -ll
end


#function evaluate_dd_ll(params::Matrix{Float64}, regimes::Array{Int64}, occ_tbl::Dict{String,Array{Int64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64}, log_div = false)
function evaluate_dd_ll(params::AbstractMatrix, regimes::Array{Int64}, occ_tbl::Dict{String,Array{Int64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64}, log_div = false)
    rates = calc_dd_rates_regimes(params, corr_Nt, regimes, log_div)
    ll = calc_loglike(occ_tbl, corr_Nt, rates[1], rates[2], tt, psi)
    return ll
end

function obj_function_dd(x::AbstractVector, Ks::Vector{Float64}, regimes::Array{Int64}, occ_tbl::Dict{String,Array{Int64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64} , log_div = false)
    LARGE = 100000000000.0
    for i in x
        if i < 0.0
            return LARGE
        end
    end
    
    unique_reg = unique(regimes)
    highest = maximum(unique_reg)
    #x_float = Vector{Float64}(x)
    eqrs = x[1:highest]
    alphas = x[highest+1:end]
    params = vcat(Ks', eqrs', alphas')
    ll = evaluate_dd_ll(params, regimes, occ_tbl, corr_Nt, tt, psi, log_div)
    return -ll
end


function get_stdev_diversity(corr_Nt::Array{Float64}, regimes::Array{Int64})
    #curreg = regimes[1]
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    med_divs = Float64[]
    for curreg in 1 : maximum(regimes)
        divs = Float64[]
        for stage_i in clade_start : clade_end
            if regimes[stage_i] == curreg
                append!(divs, corr_Nt[stage_i])
            end
        end
        curmed = std(divs)
        append!(med_divs, curmed)
    end
    #println(med_divs)
    return med_divs
end


function get_median_diversity(corr_Nt::Array{Float64}, regimes::Array{Int64})
    #curreg = regimes[1]
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    med_divs = Float64[]
    for curreg in 1 : maximum(regimes)
        divs = Float64[]
        for stage_i in clade_start : clade_end
            if regimes[stage_i] == curreg
                append!(divs, corr_Nt[stage_i])
            end
        end
        curmed = median(divs)
        append!(med_divs, curmed)
    end
    return med_divs
end

function create_regime_vector(reg_ends::Array{Int64}, tt::Dict{Int64,Tuple{Float64,Float64}})
    regs = Int64[]
    curreg = 1 
    for stage_i in 1 : length(tt)
        if stage_i > reg_ends[curreg]
            curreg += 1
        end
        append!(regs, curreg)
    end
    return regs
end

function search_regimes_non_dd_gamma(occ_tbl::Dict{String,Array{Int64}}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Float64)
    min_stages = 3  # a regime cannot have fewer than this many stages
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    max_reg = 15 
    nstage = length(tt)
    bestAIC = 1000000000.0
    psis = [psi for _ in 1 : length(tt)]

    samp_n = 0
    for ( _, value ) in occ_tbl
        #nvals = sum(value)
        nvals = count(x -> x > 0, value)
        samp_n += nvals
    end

    # do single regime model first
    curshifts = Int64[nstage]
    reg = create_regime_vector(curshifts, tt) 
    shape = rand() * .01
    start = vcat(Float64[rand()] * .05, Float64[rand()]*.05, shape) 

    obj_powell = x -> obj_function_single_gamma(x, reg, fcounts, bcounts, corr_Nt, tt, psis)
    opt = Opt(:LN_NELDERMEAD, length(start))
    lower_b = zeros(length(start)) 
    upper_b = fill(10.0, length(start))
    opt.lower_bounds = lower_b
    opt.upper_bounds = upper_b
    opt.min_objective = (x, g) -> obj_powell(x)  # The `g` argument is ignored
    opt.xtol_rel = 1e-5
    opt.ftol_rel = 1e-5
    (minf, minx, ret) = NLopt.optimize(opt, start)
    ll = -minf
   
    bestAIC = calc_aicc(ll, length(start) + 1, samp_n) # +1 is to count psi 
    println("single reg AIC: $bestAIC")
    bestshifts = [s for s in curshifts]
    last_shift = bestAIC
    for curreg in 2 : max_reg
        for stage_i in clade_start + min_stages : clade_end - min_stages 
            valid = true
            if length(curshifts) > 1
                for shiftpoint in curshifts
                    if stage_i < shiftpoint + 10 && stage_i > shiftpoint - 10
                        valid = false
                    end
                end
            end
            if valid == false
                continue
            end
            candidate_shifts = sort!(append!( curshifts, stage_i ))
            #println("CAND: ", candidate_shifts)
            reg = create_regime_vector(candidate_shifts, tt) 
            start = rand((curreg * 2) + 1) * .01
            #println("START: ",start)
            obj_powell = x -> obj_function_single_gamma(x, reg, fcounts, bcounts, corr_Nt, tt, psis)
            ll = obj_powell(start)
            opt = Opt(:LN_NELDERMEAD, length(start))
            lower_b = zeros(length(start)) 
            upper_b = fill(10.0, length(start))
            opt.lower_bounds = lower_b
            opt.upper_bounds = upper_b
            opt.min_objective = (x, g) -> obj_powell(x)  # The `g` argument is ignored
            opt.xtol_rel = 1e-5
            opt.ftol_rel = 1e-5
            (minf, minx, ret) = NLopt.optimize(opt, start)
            ll = -minf
            #res = optimize(x -> obj_function_single(x, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
            #ll = -Optim.minimum(res)
            curAIC = calc_aicc(ll, length(start) + 1, samp_n) # +1 is to count phi
            if curAIC < bestAIC
                bestAIC = curAIC
                bestshifts = [s for s in candidate_shifts]
            end
            println(candidate_shifts, " ", bestAIC, " ", curAIC)
            filter!(x -> x != stage_i, curshifts)
        end
        if bestAIC == last_shift
            break
        else
            last_shift = bestAIC
            curshifts = bestshifts
        end
    end

    reg = create_regime_vector(bestshifts, tt) 
    start = rand((length(unique(reg)) * 2 ) + 1) * .01
    ll = obj_powell(start)
    opt = Opt(:LN_NELDERMEAD, length(start))
    lower_b = zeros(length(start)) 
    upper_b = fill(10.0, length(start))
    opt.lower_bounds = lower_b
    opt.upper_bounds = upper_b
    opt.min_objective = (x, g) -> obj_powell(x)  # The `g` argument is ignored
    opt.xtol_rel = 1e-5
    opt.ftol_rel = 1e-5
    (minf, minx, ret) = NLopt.optimize(opt, start)
    ll = -minf
    println("non-dd params: ", minx)
    
    unique_reg = unique(reg)
    highest = maximum(unique_reg)
    lams = minx[1:highest]
    mus = minx[highest+1:end-1]
    bin_lam = Float64[]
    bin_mu  = Float64[]
    for r in reg
        append!(bin_lam, lams[r])
        append!(bin_mu,   mus[r])
    end

    plot_non_dd_rate_curves(bin_lam, bin_mu, reg, tt)
    plot_regime_curve(corr_Nt, reg, tt, "dtt_non-dd.png")
    
    println(reg)
    println(corr_Nt)
end

function search_regimes_dd_gamma(occ_tbl::Dict{String,Array{Int64}}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Float64, multi_eqr = true, log_div = false, start_div_rate = 0.0)
    min_stages = 3  # a regime cannot have fewer than this many stages
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    max_reg = 15 
    nstage = length(tt)
    bestAIC = 1000000000.0
    psis = [psi for _ in 1 : length(tt)]

    samp_n = 0
    for ( _, value ) in occ_tbl
        #nvals = sum(value)
        nvals = count(x -> x > 0, value)
        samp_n += nvals
    end

    # do single regime model first
    curshifts = Int64[nstage]
    reg = create_regime_vector(curshifts, tt) 
    Ks = get_median_diversity(corr_Nt, reg)

    if log_div == true
        start = vcat(rand(1) * .05, rand(1) * 0.03)
    else 
        start = vcat(rand(1) * .05, rand(1) * 0.1 / maximum(corr_Nt)) 
    end
    append!(start, rand() * .01 )
    obj_nlopt = (x, g) -> obj_function_dd_gamma(x, Ks, reg, fcounts, bcounts, corr_Nt, tt, psis, log_div)
    opt = Opt(:LN_NELDERMEAD, length(start))
    opt.min_objective = obj_nlopt
    opt.xtol_rel = 1e-5
    opt.ftol_rel = 1e-5
    lower_b = zeros(length(start)) 
    upper_b = fill(10.0, length(start))
    opt.lower_bounds = lower_b
    opt.upper_bounds = upper_b

    (minf, minx, ret) = NLopt.optimize(opt, start)
    ll = -minf 


    bestAIC = calc_aicc(ll, length(start) + length(Ks) + 1, samp_n) # +1 is to count phi
    bestshifts = [s for s in curshifts]
    last_shift = bestAIC
    for curreg in 2 : max_reg
        for stage_i in clade_start + min_stages : clade_end - min_stages 
            valid = true
            if length(curshifts) > 1
                for shiftpoint in curshifts
                    if stage_i < shiftpoint + min_stages && stage_i > shiftpoint - min_stages 
                        valid = false
                    end
                end
            end
            if valid == false
                continue
            end
            candidate_shifts = sort!(append!( curshifts, stage_i ))
            reg = create_regime_vector(candidate_shifts, tt) 
            Ks = get_median_diversity(corr_Nt, reg)
            if multi_eqr == true
                if log_div == true
                    start = vcat( rand(curreg) * .05, rand(curreg) * 0.03 ) 
                else
                    start = vcat(rand(curreg) * .05, rand(curreg) * 0.1 / maximum(corr_Nt)) 
                end
                append!(start,rand() * .01)
                opt = Opt(:LN_NELDERMEAD, length(start))
                opt.min_objective = obj_nlopt
                opt.xtol_rel = 1e-5
                opt.ftol_rel = 1e-5
                lower_b = zeros(length(start)) 
                upper_b = fill(10.0, length(start))
                opt.lower_bounds = lower_b
                opt.upper_bounds = upper_b

                (minf, minx, ret) = NLopt.optimize(opt, start)
                ll = -minf
            else
                # NEED TO UPDATE TO GAMMA IF GOING TO USE
                start = vcat( rand(curreg) * .1, rand(1) * 0.03 ) 
                res = optimize(x -> obj_function_dd_one_alpha(x, Ks, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
                ll = -Optim.minimum(res)
            end
            curAIC = calc_aicc(ll, length(start) + length(Ks) + 1, samp_n) # +1 is to count psi

            if curAIC < bestAIC
                bestAIC = curAIC
                bestshifts = [s for s in candidate_shifts]
            end
            println(candidate_shifts, " ", bestAIC, " ", curAIC)#, " ", -ll)#, Optim.minimizer(res))#, " ", Ks)
            filter!(x -> x != stage_i, curshifts)
        end
        if bestAIC == last_shift
            break
        else
            last_shift = bestAIC
            curshifts = bestshifts
        end
    end

    reg = create_regime_vector(bestshifts, tt) 
    curreg = length(unique(reg))
    Ks = get_median_diversity(corr_Nt, reg)
    if log_div == true
        start = vcat( rand(curreg) * .05, rand(curreg) * 0.03 ) 
    else
        start = vcat(rand(curreg) * .05, rand(curreg) * 0.1 / maximum(corr_Nt)) 
    end
    append!(start,rand() * .01)
    obj_nlopt = (x, g) -> obj_function_dd_gamma(x, Ks, reg, fcounts, bcounts, corr_Nt, tt, psis, log_div)
    opt = Opt(:LN_NELDERMEAD, length(start))
    opt.min_objective = obj_nlopt
    opt.xtol_rel = 1e-5
    opt.ftol_rel = 1e-5
    lower_b = zeros(length(start)) 
    upper_b = fill(10.0, length(start))
    opt.lower_bounds = lower_b
    opt.upper_bounds = upper_b

    (minf, minx, ret) = NLopt.optimize(opt, start)
    ll = -minf

    println("dd params: ", minx)
    unique_reg = unique(reg)
    highest = maximum(unique_reg)
    eqrs = minx[1:highest]
    alphas = minx[highest+1:end-1]
    params = vcat(Ks', eqrs', alphas')
    rates = calc_dd_rates_regimes(params, corr_Nt, reg, log_div)
    plot_rate_curves(rates[1], rates[2], Ks, reg, tt)
    #println(rates[1])
    #println(rates[2])
    #println(corr_Nt)
    plot_regime_curve(corr_Nt, reg, tt, "dtt_dd.png")
end

using Plots

function plot_non_dd_rate_curves(bin_lam, bin_mu, reg, tt)
    t = 1:length(bin_lam)
    at = [-tt[i][1] for i in t]
    mass_extinctions = Float64[-443.8, -372.2, -251.9, -201.3, -66.0]
    #palette = [:forestgreen, :dodgerblue, :darkorange, :purple, :crimson]
    palette = [
        :forestgreen, :dodgerblue, :darkorange, :purple, :crimson,  # original 5
        :goldenrod, :teal, :orchid, :slateblue, :firebrick,         # added 5
        :mediumseagreen, :deepskyblue, :sienna, :darkmagenta, :indigo  # added 5 more
    ]
    
    plt = plot(xlabel="Time (Ma)", ylabel="Rate", legend=:topleft)#, title="Origination and Extinction Rates")

    for i in 2:length(at)
        regime_index = reg[i-1]
        segment_color = palette[regime_index]
        
        plot!(plt, [at[i-1], at[i]], [bin_lam[i-1], bin_lam[i]], 
              color=segment_color, 
              linewidth=2.5, 
              label="")
              
        plot!(plt, [at[i-1], at[i]], [bin_mu[i-1], bin_mu[i]], 
              color=segment_color, 
              linestyle=:dash,
              linewidth=1.5, 
              label="")
    end

    vline!(plt, mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1.5, label="")

    plot!(plt, [NaN], [NaN], label="Origination", color=:black, lw=2.5)
    plot!(plt, [NaN], [NaN], label="Extinction", color=:black, ls=:dash, lw=1.5)
    
    #for r in eachindex(Ks)
    #    k = Int64(round(Ks[r]))
    #    plot!(plt, [NaN], [NaN], label="K: $k", color=palette[r])
    #end

    savefig(plt, "non-dd_orig_ext_rates.png")
    return plt
end



function plot_rate_curves(bin_lam, bin_mu, Ks, reg, tt)
    t = 1:length(bin_lam)
    at = [-tt[i][1] for i in t]
    mass_extinctions = Float64[-443.8, -372.2, -251.9, -201.3, -66.0]
    #palette = [:forestgreen, :dodgerblue, :darkorange, :purple, :crimson]
    palette = [
        :forestgreen, :dodgerblue, :darkorange, :purple, :crimson,  # original 5
        :goldenrod, :teal, :orchid, :slateblue, :firebrick,         # added 5
        :mediumseagreen, :deepskyblue, :sienna, :darkmagenta, :indigo  # added 5 more
    ]
    
    plt = plot(xlabel="Time (Ma)", ylabel="Rate", legend=:topleft)#, title="Origination and Extinction Rates")

    for i in 2:length(at)
        regime_index = reg[i-1]
        segment_color = palette[regime_index]
        
        plot!(plt, [at[i-1], at[i]], [bin_lam[i-1], bin_lam[i]], 
              color=segment_color, 
              linewidth=2.5, 
              label="")
              
        plot!(plt, [at[i-1], at[i]], [bin_mu[i-1], bin_mu[i]], 
              color=segment_color, 
              linestyle=:dash,
              linewidth=1.5, 
              label="")
    end

    vline!(plt, mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1.5, label="")

    plot!(plt, [NaN], [NaN], label="Origination", color=:black, lw=2.5)
    plot!(plt, [NaN], [NaN], label="Extinction", color=:black, ls=:dash, lw=1.5)
    
    #for r in eachindex(Ks)
    #    k = Int64(round(Ks[r]))
    #    plot!(plt, [NaN], [NaN], label="K: $k", color=palette[r])
    #end

    savefig(plt, "dd_orig_ext_rates.png")
    return plt
end


function search_regimes_non_dd(occ_tbl::Dict{String,Array{Int64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Float64)
    min_stages = 6  # a regime cannot have fewer than this many stages
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    max_reg = 3
    nstage = length(tt)
    bestAIC = 1000000000.0
    psis = [psi for _ in 1 : length(tt)]

    samp_n = 0
    for ( _, value ) in occ_tbl
        #nvals = sum(value)
        nvals = count(x -> x > 0, value)
        samp_n += nvals
    end
    
    # do single regime model first
    curshifts = Int64[nstage]
    reg = create_regime_vector(curshifts, tt) 
    start = vcat(Float64[rand()], Float64[rand()]) 

    #psi = 0.2
    #println("TESTING DIFF RATES")
    #bin_lam = [0.03 for _ in eachindex(corr_Nt)]
    #orig_ll = three_bin_origination_ll(occ_tbl, corr_Nt, bin_lam, tt, [psi for _ in eachindex(corr_Nt)])

    #res = optimize(x -> obj_function_single(x, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
    #println("non-dd params: ", Optim.minimizer(res))
    #ll = -Optim.minimum(res) 
    obj_powell = x -> obj_function_single(x, reg, occ_tbl, corr_Nt, tt, psis)
    opt = Opt(:LN_NELDERMEAD, length(start))
    lower_b = zeros(length(start)) 
    upper_b = fill(10.0, length(start))
    opt.lower_bounds = lower_b
    opt.upper_bounds = upper_b
    opt.min_objective = (x, g) -> obj_powell(x)  # The `g` argument is ignored
    opt.xtol_rel = 1e-5
    opt.ftol_rel = 1e-5
    (minf, minx, ret) = NLopt.optimize(opt, start)
    ll = -minf

    
    bestAIC = calc_aicc(ll, length(start) + 1, samp_n) # +1 is to count phi
    bestshifts = [s for s in curshifts]
    last_shift = bestAIC
    for curreg in 2 : max_reg
        for stage_i in clade_start + min_stages : clade_end - min_stages 
            valid = true
            if length(curshifts) > 1
                for shiftpoint in curshifts
                    if stage_i < shiftpoint + 10 && stage_i > shiftpoint - 10
                        valid = false
                    end
                end
            end
            if valid == false
                continue
            end
            candidate_shifts = sort!(append!( curshifts, stage_i ))
            #println("CAND: ", candidate_shifts)
            reg = create_regime_vector(candidate_shifts, tt) 
            start = rand(curreg * 2)

            obj_powell = x -> obj_function_single(x, reg, occ_tbl, corr_Nt, tt, psis)
            opt = Opt(:LN_NELDERMEAD, length(start))
            lower_b = zeros(length(start)) 
            upper_b = fill(10.0, length(start))
            opt.lower_bounds = lower_b
            opt.upper_bounds = upper_b
            opt.min_objective = (x, g) -> obj_powell(x)  # The `g` argument is ignored
            opt.xtol_rel = 1e-5
            opt.ftol_rel = 1e-5
            (minf, minx, ret) = NLopt.optimize(opt, start)
            ll = -minf

            #res = optimize(x -> obj_function_single(x, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
            #ll = -Optim.minimum(res)
            curAIC = calc_aicc(ll, length(start) + 1, samp_n) # +1 is to count phi
            if curAIC < bestAIC
                bestAIC = curAIC
                bestshifts = [s for s in candidate_shifts]
            end
            println(candidate_shifts, " ", bestAIC, " ", curAIC)
            filter!(x -> x != stage_i, curshifts)
        end
        if bestAIC == last_shift
            break
        else
            last_shift = bestAIC
            curshifts = bestshifts
        end
    end

    reg = create_regime_vector(bestshifts, tt) 
    start = rand(length(unique(reg)) * 2)
    res = Optim.optimize(x -> obj_function_single(x, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
    println("non-dd params: ", Optim.minimizer(res))
    plot_regime_curve(corr_Nt, reg, tt)
    x = Optim.minimizer(res)
    println(reg)
    println(corr_Nt)

end



function search_regimes_dd(occ_tbl::Dict{String,Array{Int64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Float64, multi_eqr = true, log_div = false, start_div_rate = 0.0)
    min_stages = 6  # a regime cannot have fewer than this many stages
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    max_reg = 3
    nstage = length(tt)
    bestAIC = 1000000000.0
    psis = [psi for _ in 1 : length(tt)]

    samp_n = 0
    for ( _, value ) in occ_tbl
        #nvals = sum(value)
        nvals = count(x -> x > 0, value)
        samp_n += nvals
    end

    # do single regime model first
    curshifts = Int64[nstage]
    reg = create_regime_vector(curshifts, tt) 
    Ks = get_median_diversity(corr_Nt, reg)

    if log_div == true
        start = vcat(rand(1) * .05, rand(1) * 0.03)
    else 
        start = vcat(rand(1) * .05, rand(1) * 0.1 / maximum(corr_Nt)) 
    end

    #res = Optim.optimize(x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis, log_div), start, NelderMead())
    #ll = -Optim.minimum(res) 
    #println("params: ", Optim.minimizer(res))
    #println("NelderMead:", -ll)

    if log_div == true
        start = vcat(rand(1) * .05, rand(1) * 0.03)
    else 
        start = vcat(rand(1) * .05, rand(1) * 0.1 / maximum(corr_Nt)) 
    end

    #=
    obj = x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis, log_div)
    td = TwiceDifferentiable(obj, start; autodiff = :forward)
    res = optimize(td, start, BFGS())
    ll = -Optim.minimum(res) 
    println(-ll)
    =#    
    obj_nlopt = (x, g) -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis, log_div)
    opt = Opt(:LN_NELDERMEAD, length(start))
    opt.min_objective = obj_nlopt
    opt.xtol_rel = 1e-5
    opt.ftol_rel = 1e-5
    lower_b = zeros(length(start)) 
    upper_b = fill(10.0, length(start))
    opt.lower_bounds = lower_b
    opt.upper_bounds = upper_b

    (minf, minx, ret) = NLopt.optimize(opt, start)
    ll = -minf 
    #=
    obj_nlopt_no_grad = (x) -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis, log_div)

    # Change the algorithm to GN_ESCH
    opt = Opt(:GN_ESCH, length(start))
    # For derivative-free algorithms, the second argument of the objective function (g) is ignored
    opt.min_objective = (x, g) -> obj_nlopt_no_grad(x)
    # You may want to loosen the tolerance further for global optimizers, as they are often more computationally expensive
    #opt.xtol_rel = 1e-5
    
    println("\nRunning NLopt.jl GN_ESCH with your function...")
    (minf, minx, ret) = NLopt.optimize(opt, start) 

    println(minf)
    println(minx)
    println(ret)
    =#
    #res = optimize(x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis), start, BFGS(linesearch=HagerZhang()), Optim.Options(g_tol=1e-6)) 
    #res = optimize(x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis), start, SimulatedAnnealing()) 

    bestAIC = calc_aicc(ll, length(start) + length(Ks) + 1, samp_n) # +1 is to count phi
    #println(Optim.converged(res))
    #println(bestAIC)
    bestshifts = [s for s in curshifts]
    last_shift = bestAIC
    for curreg in 2 : max_reg
        for stage_i in clade_start + min_stages : clade_end - min_stages 
            #if stage_i > 33 ## TODO : This needs to come out and i need to figure out how to search space better
            #    break
            #end
            valid = true
            if length(curshifts) > 1
                for shiftpoint in curshifts
                    if stage_i < shiftpoint + min_stages && stage_i > shiftpoint - min_stages 
                        valid = false
                    end
                end
            end
            if valid == false
                continue
            end
            candidate_shifts = sort!(append!( curshifts, stage_i ))
            #println("CAND: ", candidate_shifts)
            reg = create_regime_vector(candidate_shifts, tt) 
            Ks = get_median_diversity(corr_Nt, reg)
            #=
            K_sd = get_stdev_diversity(corr_Nt, reg)
            println(Ks, " ", K_sd)
            for i in eachindex(Ks)
                K_sd[i] = K_sd[i] / Ks[i] / maximum(corr_Nt)  
            end
            println(K_sd)
            =#
            if multi_eqr == true
                if log_div == true
                    start = vcat( rand(curreg) * .05, rand(curreg) * 0.03 ) 
                else
                    start = vcat(rand(curreg) * .05, rand(curreg) * 0.1 / maximum(corr_Nt)) 
                end
                #res = Optim.optimize(x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis, log_div), start, NelderMead())
                #ll = -Optim.minimum(res) 

                #=
                res = Optim.optimize(x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis, log_div), start, NelderMead())
                println(Ks)
                println("start1 ", start)
                println("params1: ", Optim.minimizer(res))
                ll = -Optim.minimum(res) 
                println(ll)

                #start = vcat(rand(curreg) * .1, rand(curreg) * 0.1) 
                start = vcat(fill(start_div_rate, curreg), K_sd) 
                println("start2 ", start)

                @time begin
                    res2 = Optim.optimize(x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis, log_div), start, NelderMead())
                end
                println("params2: ", Optim.minimizer(res2))
                ll = -Optim.minimum(res2) 
                println(ll)

                if log_div == true
                    start = vcat( rand(curreg) * .05, rand(curreg) * 0.03 ) 
                else
                    start = vcat(rand(curreg) * .05, rand(curreg) * 0.1 / maximum(corr_Nt)) 
                end
                =#

                # NLOPT IMPLEMENTATION
                obj_powell = x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis, log_div)
                opt = Opt(:LN_NELDERMEAD, length(start))
                lower_b = zeros(length(start)) 
                upper_b = fill(10.0, length(start))
                opt.lower_bounds = lower_b
                opt.upper_bounds = upper_b
                opt.min_objective = (x, g) -> obj_powell(x)  # The `g` argument is ignored
                opt.xtol_rel = 1e-5
                opt.ftol_rel = 1e-5
                (minf, minx, ret) = NLopt.optimize(opt, start)
                ll = -minf
                
                #println("minl: ", minf)
                #println("nlopt params: ", minx)
                #println(ret)
                #exit()
                #res = optimize(x -> obj_function_dd_Konly(x, Optim.minimizer(res), reg, occ_tbl, corr_Nt, tt, psis), Ks, NelderMead())
                #res = optimize(x -> obj_function_dd(x, Optim.minimizer(res), reg, occ_tbl, corr_Nt, tt, psis, log_div), start, NelderMead())
                #start = vcat( Ks, rand(curreg) * .1, rand(curreg) * 0.03 ) 
                #res = optimize(x -> obj_function_dd_freeK(x, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
                #res = optimize(x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis), start, BFGS(linesearch=HagerZhang()), Optim.Options(g_tol=1e-6)) 
            else
                #start = vcat( rand(1) * .1, rand(curreg) * 0.03 ) 
                #res = optimize(x -> obj_function_dd_one_eqr(x, Ks, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
                start = vcat( rand(curreg) * .1, rand(1) * 0.03 ) 
                res = optimize(x -> obj_function_dd_one_alpha(x, Ks, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
                #start = vcat( rand(1) * .1, rand(1) * 0.03 ) 
                #res = optimize(x -> obj_function_dd_one_eqr_alpha(x, Ks, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
                ll = -Optim.minimum(res)

            end
            curAIC = calc_aicc(ll, length(start) + length(Ks) + 1, samp_n) # +1 is to count psi

            #curaic = calc_aicc(-nll, length(start) + length(K_vec) + 1, samp_n)
            if curAIC < bestAIC
                bestAIC = curAIC
                bestshifts = [s for s in candidate_shifts]
            end
            println(candidate_shifts, " ", bestAIC, " ", curAIC)#, " ", -ll)#, Optim.minimizer(res))#, " ", Ks)
            filter!(x -> x != stage_i, curshifts)
        end
        if bestAIC == last_shift
            break
        else
            last_shift = bestAIC
            curshifts = bestshifts
        end
    end

    reg = create_regime_vector(bestshifts, tt) 
    curreg = length(unique(reg))
    Ks = get_median_diversity(corr_Nt, reg)
    #start = vcat( rand(curreg) * .1, rand(curreg) * 0.03 )
    if log_div == true
        start = vcat( rand(curreg) * .05, rand(curreg) * 0.03 ) 
    else
        start = vcat(rand(curreg) * .05, rand(curreg) * 0.1 / maximum(corr_Nt)) 
    end
    res = Optim.optimize(x -> obj_function_dd(x, Ks, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
    println("dd params: ", Optim.minimizer(res))
    x = Optim.minimizer(res)
    println(reg)
    regimes = reg
    unique_reg = unique(regimes)
    highest = maximum(unique_reg)
    #x_float = Vector{Float64}(x)
    eqrs = x[1:highest]
    alphas = x[highest+1:end]
    params = vcat(Ks', eqrs', alphas')
    rates = calc_dd_rates_regimes(params, corr_Nt, regimes, log_div)
    println(rates[1])
    println(rates[2])
    println(corr_Nt)
    plot_regime_curve(corr_Nt, reg, tt)
    exit()
end

function plot_regime_curve(corr_Nt, reg, tt, flnm = "dtt_reg.png")
    t = 1:length(corr_Nt)
    at = [-tt[i][1] for i in t]
    mass_extinctions = Float64[-443.8, -372.2, -251.9, -201.3, -66.0]
    palette = [
        :forestgreen, :dodgerblue, :darkorange, :purple, :crimson,  
        :goldenrod, :teal, :orchid, :slateblue, :firebrick,        
        :mediumseagreen, :deepskyblue, :sienna, :darkmagenta, :indigo  
    ]

    plt = plot(xlabel="Time (Ma)", ylabel="# Genera", legend=:topleft)

    for i in 2 : length(at)
        regime_index = reg[i-1]
        segment_color = palette[regime_index]
        
        plot!(plt, [at[i-1], at[i]], [corr_Nt[i-1], corr_Nt[i]], 
              color=segment_color, 
              linewidth=2.5, 
              label="")
    end

    vline!(plt, mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1.5, label="")

    Ks = get_median_diversity(corr_Nt, reg)
    #unique_regimes = sort(unique(reg))
    for r in eachindex(Ks)
        k = Int64(round(Ks[r]))
        plot!(plt, [NaN], [NaN],
            label="K: $k",
            color=palette[r])
    end

    savefig(plt,flnm)
    return plt
end

function get_sliding_window_counts(occ_tbl::Dict{String,Array{Int64}})
    nbin = length(first(occ_tbl)[2])

    fcounts = Vector{Vector{Float64}}()
    for i in 1 : nbin - 2
        counts = count_ntax_3cats(occ_tbl, i)
        push!(fcounts, counts)
    end

    bcounts = Vector{Vector{Float64}}()
    for i in nbin : -1 : 3 
        counts = count_ntax_3cats(occ_tbl, i, true)
        push!(bcounts, counts)
    end
    return fcounts, reverse(bcounts)
end

function main(args)
    if length(args) != 3
        println(string("usage: julia ",PROGRAM_FILE," <timetable> <occurrences> <log diversity? (true or false)>"))
        exit()
    end

    tt = read_in_tt(ARGS[1])
    occ = read_in_occurrences(ARGS[2], "genus")
    log_div = parse(Bool, strip(lowercase(ARGS[3])))
    
    lmy = calc_Lmy(occ)
    emp_rate = length(occ) / lmy
    println("average ext + sp rate: ", emp_rate)
    occ_tbl = create_genus_occurrence_table(occ, tt)

    sp = calc_prob_int_sample(occ_tbl, tt)
    psi = estimate_preservation_rate(sp, tt)
    obs_Nt = get_num_obs_stages(occ_tbl)
    println(obs_Nt)
    println("preservation rate: ", psi)
    starting_size = 4
    #corr_Nt =  calc_range_dtt(occ_tbl)
    corr_Nt = calc_corrected_txc(occ_tbl, obs_Nt, tt, [psi for _ in 1 : length(tt)], starting_size)
    #corr_Nt = [Float64(i) for i in obs_Nt]
    #clade_start, clade_end = get_start_end_bins(occ_range)
    fcounts, bcounts = get_sliding_window_counts(occ_tbl)
    search_regimes_non_dd_gamma(occ_tbl, fcounts, bcounts, corr_Nt, tt, psi)
    search_regimes_dd_gamma(occ_tbl, fcounts, bcounts, corr_Nt, tt, psi, true, log_div, emp_rate)
end

main(ARGS)