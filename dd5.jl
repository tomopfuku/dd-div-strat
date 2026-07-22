using CSV
using Plots
using LogExpFunctions
using SpecialFunctions
using Base.Threads
using Statistics
using Roots
using Optim
using Random
using Optimization, OptimizationOptimJL, OptimizationNLopt, ForwardDiff
using Distributions
using LogExpFunctions
using DataFrames


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

function get_through_diversity(genus_stages::Dict{String,Array{Int64}})::Array{Int64}
    n_obs = zeros(Int64, length(first(values(genus_stages))))
    for (genus, stages) in genus_stages
        FAD,LAD = get_start_end_bins(stages)
        for i in FAD : LAD
            n_obs[i] += 1
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

function calc_SQS_curve(genus_stages::Dict{String,Array{Int64}}, rt_curve::Array{Float64}, k::Int = 100, q::Float64 = 0.8)
    nbin = length(first(values(genus_stages)))
    bc = zeros(Float64, nbin)
    genus_names = collect(keys(genus_stages))

    for i in 1 : nbin
        bin_genus_counts = Dict{String,Int64}()
        nsingletons = 0
        nocc = 0
        
        for genus in genus_names
            count_i = genus_stages[genus][i]
            if count_i > 0
                bin_genus_counts[genus] = count_i
                nocc += count_i
                if count_i == 1
                    nsingletons += 1
                end
            end
        end

        if nocc <= 0
            bc[i] = 0.0
            continue
        end

        goods_u = 1.0 - (nsingletons / nocc)
        
        if goods_u < q
            bc[i] = NaN 
            continue
        end

        # Optimized pool creation
        occurrence_pool = String[]
        for (genus, count) in bin_genus_counts
            append!(occurrence_pool, repeat([genus], count))
        end

        max_possible_diversity = length(keys(bin_genus_counts))
        subsampled_diversities = Float64[]
        
        for m in 1 : k
            current_diversity = 0
            running_coverage = 0.0
            seen_taxa = Set{String}()
            
            while running_coverage < q && current_diversity < max_possible_diversity
                drawn_taxon = rand(occurrence_pool)
                
                if !(drawn_taxon in seen_taxa)
                    push!(seen_taxa, drawn_taxon)
                    current_diversity += 1
                    rel_freq = bin_genus_counts[drawn_taxon] / nocc
                    running_coverage += (rel_freq * goods_u)
                end
            end

            push!(subsampled_diversities, current_diversity)
        end
        bc[i] = mean(subsampled_diversities)
    end

    clade_start, clade_end = get_start_end_bins(rt_curve)
    for i in 1 : nbin
        if isnan(bc[i]) || bc[i] == 0.0
            # Only apply the RT fallback if the gap falls within the clade's actual lifespan
            if i >= clade_start && i <= clade_end
                bc[i] = Float64(rt_curve[i])
            else
                bc[i] = 0.0 # Standardize edges outside clade lifespan to 0
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
    
    t = 1:length(corr_txc) #- 1
    at = [-tt[i][1] for i in t]
    """
    mass_extinctions = Float64[-443.8, -372.2, -251.9, -201.3, -66.0]
    plt = plot(at,corr_txc,label = ["corr_txc"])
    plot!(plt, at, obs_txc, ls=:dot, label = ["obs_txc"])
    vline!(mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1.5, label="")
    savefig(plt,"dtt.png")
    
    plt2 = plot(at, log.(corr_txc),label = ["log_txc"])
    plot!(plt2, at, log.(obs_txc), ls=:dot, label = ["log_obs_txc"])
    vline!(mass_extinctions, ls=:dot, color=:grey, alpha=0.5, lw=1.5, label="")
    savefig(plt2,"log_dtt.png")
    """
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


function calc_dd_rates_regimes(params::AbstractMatrix, corr_Nt::Array{Float64}, regimes::Array{Int64}, log_div = false)
    K_v = params[1, :]
    eqr0_v = params[2, :]
    if size(params, 1) >= 4
        alpha_lam_v = params[3, :]
        alpha_mu_v = params[4, :]
    else
        alpha_v = params[3, :]
        alpha_lam_v = alpha_v
        alpha_mu_v = alpha_v
    end

    if log_div == true
        lam0_v = [ ( eqr0_v[i] + ( alpha_lam_v[i] * log( K_v[i] ) ) ) for i in eachindex(K_v)]
        mu0_v  = [ ( eqr0_v[i] - ( alpha_mu_v[i] *  log( K_v[i] ) ) ) for i in eachindex(K_v) ]
    else
        lam0_v = [ ( eqr0_v[i] + ( alpha_lam_v[i] * K_v[i]  ) ) for i in eachindex(K_v)]
        mu0_v  = [ ( eqr0_v[i] - ( alpha_mu_v[i] *  K_v[i]  ) ) for i in eachindex(K_v) ]
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
                lam_t = lam0_v[curreg] - ( alpha_lam_v[curreg] *  log( nt ) ) 
            else
                lam_t = lam0_v[curreg] - ( alpha_lam_v[curreg] *  nt  ) 
            end
            #lam_t = lam0 - ( alpha * nt ) 
            if lam_t < 0
                lam_t = 0.0001 # TODO: is this constraint the best way to handle this?
            end
            if log_div == true
                mu_t  = mu0_v[curreg]  + ( alpha_mu_v[curreg] *  log( nt )  )
            else
                mu_t  = mu0_v[curreg]  + ( alpha_mu_v[curreg] *  nt  )
                #println(alpha_lam_v[curreg], " $nt ", mu_t)
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
            #ll += max(curll, log(1e-300))
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
            #ll += max(curll, log(1e-300))
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
    scale_param = psi[1] / shape
    gamma = Gamma(shape, scale_param)
    K = 4
    p_mid = ((1:K) .- 0.5) ./ K
    medians = quantile.(Ref(gamma), p_mid)

    orig_ll = three_bin_origination_ll_gamma(bcounts, corr_Nt, bin_lam, tt, medians)
    ext_ll = three_bin_extinction_ll_gamma(fcounts, corr_Nt, bin_mu, tt, medians)
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

function unpack_dd_gamma_params(x::AbstractVector, highest::Int64)
    if length(x) == (2 * highest) + 1
        eqrs = x[1:highest]
        alpha_lam = x[highest+1:end-1]
        alpha_mu = copy(alpha_lam)
    elseif length(x) == (3 * highest) + 1
        eqrs = x[1:highest]
        alpha_lam = x[highest+1:2*highest]
        alpha_mu = x[2*highest+1:end-1]
    else
        error("Unexpected DD gamma parameter vector length $(length(x)) for highest regime count $highest")
    end
    shape = x[end] / .01
    return eqrs, alpha_lam, alpha_mu, shape
end

function evaluate_dd_ll_gamma(params::AbstractMatrix, shape::Float64, regimes::Array{Int64}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64}, log_div = false)
    bin_lam, bin_mu = calc_dd_rates_regimes(params, corr_Nt, regimes, log_div)
    ll = calc_loglike_gamma(fcounts, bcounts, corr_Nt, bin_lam, bin_mu, tt, psi, shape) 
    return ll
end

function obj_function_dd_gamma(x::AbstractVector, Ks::Vector{Float64}, regimes::Array{Int64}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Array{Float64}, log_div = false, separate_dd_alphas::Bool = false)
    LARGE = 100000000000.0
    for i in x
        if i < 0.0
            return LARGE
        end
    end
    
    unique_reg = unique(regimes)
    highest = maximum(unique_reg)
    eqrs, alpha_lam, alpha_mu, shape = unpack_dd_gamma_params(x, highest)
    if separate_dd_alphas == true
        params = vcat(Ks', eqrs', alpha_lam', alpha_mu')
    else
        shared_alphas = (alpha_lam .+ alpha_mu) ./ 2.0
        params = vcat(Ks', eqrs', shared_alphas')
    end
    ll = evaluate_dd_ll_gamma(params, shape, regimes, fcounts, bcounts, corr_Nt, tt, psi, log_div)
    return -ll
end


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

const DEFAULT_OPT_SEED = 5731
const DEFAULT_AIC_TOL = 1e-4

function run_bounded_local_multistart(obj, starts::Vector{Vector{Float64}};
        lower::Float64 = 1e-8,
        upper::Float64 = 10.0,
        xtol_rel::Float64 = 1e-6,
        ftol_rel::Float64 = 1e-6)

    bestf = Inf
    bestx = copy(starts[1])
    bestret = nothing

    for alg in (:LN_BOBYQA, :LN_NELDERMEAD)
        for raw_start in starts
            start = clamp.(raw_start, lower, upper)
            opt = Opt(alg, length(start))
            opt.lower_bounds = fill(lower, length(start))
            opt.upper_bounds = fill(upper, length(start))
            opt.initial_step = max.(abs.(start) .* 0.25, fill(1e-4, length(start)))
            opt.min_objective = (x, g) -> obj(x)
            opt.xtol_rel = xtol_rel
            opt.ftol_rel = ftol_rel

            minf, minx, ret = NLopt.optimize(opt, start)
            if isfinite(minf) && minf < bestf
                bestf = minf
                bestx = copy(minx)
                bestret = ret
            end
        end

        # BOBYQA is usually more stable here; use Nelder-Mead only as a fallback.
        if isfinite(bestf)
            break
        end
    end

    return bestf, bestx, bestret
end

function accept_aic(curAIC::Float64, bestAIC::Float64, aic_tol::Float64)
    return curAIC < bestAIC - aic_tol
end

function has_aic_improvement(bestAIC::Float64, lastAIC::Float64, aic_tol::Float64)
    return bestAIC < lastAIC - aic_tol
end

function valid_candidate_stage(stage_i::Int64, curshifts::Vector{Int64}, min_stages::Int64)
    if length(curshifts) <= 1
        return true
    end

    for shiftpoint in curshifts
        if stage_i < shiftpoint + min_stages && stage_i > shiftpoint - min_stages
            return false
        end
    end

    return true
end

function candidate_stages(curshifts::Vector{Int64}, clade_start::Int64, clade_end::Int64, min_stages::Int64)
    return [stage_i for stage_i in clade_start + min_stages : clade_end - min_stages if valid_candidate_stage(stage_i, curshifts, min_stages)]
end

function print_search_progress(label::String, curreg::Int64, done::Int64, total::Int64, bestAIC::Float64)
    print("\r$label: searching $curreg regimes, candidate $done/$total, best AIC = $(round(bestAIC, digits = 4))")
    flush(stdout)
end

function finish_search_progress(label::String, curreg::Int64, bestAIC::Float64)
    println("\r$label: completed $curreg-regime search, best AIC = $(round(bestAIC, digits = 4))")
end

function regime_parameter_map(prev_shifts::Vector{Int64}, candidate_shifts::Vector{Int64}, tt::Dict{Int64,Tuple{Float64,Float64}})
    prev_reg = create_regime_vector(prev_shifts, tt)
    candidate_reg = create_regime_vector(candidate_shifts, tt)
    mapping = Int64[]

    for curreg in 1 : maximum(candidate_reg)
        first_stage = findfirst(==(curreg), candidate_reg)
        append!(mapping, prev_reg[first_stage])
    end

    return mapping
end

function non_dd_gamma_starts(rng::AbstractRNG, nreg::Int64;
        prev_x = nothing,
        prev_shifts = nothing,
        candidate_shifts = nothing,
        tt = nothing,
        n_starts::Int64 = 2)

    starts = Vector{Float64}[]

    if prev_x !== nothing && prev_shifts !== nothing && candidate_shifts !== nothing && tt !== nothing
        mapping = regime_parameter_map(prev_shifts, candidate_shifts, tt)
        prev_nreg = maximum(create_regime_vector(prev_shifts, tt))
        prev_lams = prev_x[1:prev_nreg]
        prev_mus = prev_x[prev_nreg+1:end-1]
        push!(starts, vcat(prev_lams[mapping], prev_mus[mapping], prev_x[end]))
    else
        push!(starts, vcat(fill(0.025, nreg), fill(0.025, nreg), 0.01))
    end

    push!(starts, vcat(fill(0.01, nreg), fill(0.01, nreg), 0.005))

    while length(starts) < n_starts
        push!(starts, rand(rng, (nreg * 2) + 1) .* 0.02 .+ 0.001)
    end

    return starts[1:min(n_starts, length(starts))]
end

function dd_gamma_starts(rng::AbstractRNG, nreg::Int64, corr_Nt::Array{Float64}, log_div::Bool, separate_dd_alphas::Bool = false;
        prev_x = nothing,
        prev_shifts = nothing,
        candidate_shifts = nothing,
        tt = nothing,
        n_starts::Int64 = 2)

    starts = Vector{Float64}[]
    alpha_mid = log_div ? 0.015 : 0.05 / maximum(corr_Nt)
    alpha_low = log_div ? 0.005 : 0.01 / maximum(corr_Nt)

    if prev_x !== nothing && prev_shifts !== nothing && candidate_shifts !== nothing && tt !== nothing
        mapping = regime_parameter_map(prev_shifts, candidate_shifts, tt)
        prev_nreg = maximum(create_regime_vector(prev_shifts, tt))
        prev_eqrs, prev_alpha_lam, prev_alpha_mu, prev_shape = unpack_dd_gamma_params(prev_x, prev_nreg)
        if separate_dd_alphas == true
            push!(starts, vcat(prev_eqrs[mapping], prev_alpha_lam[mapping], prev_alpha_mu[mapping], prev_shape))
        else
            shared_prev_alphas = (prev_alpha_lam .+ prev_alpha_mu) ./ 2.0
            push!(starts, vcat(prev_eqrs[mapping], shared_prev_alphas[mapping], prev_shape))
        end
    else
        if separate_dd_alphas == true
            push!(starts, vcat(fill(0.025, nreg), fill(alpha_mid, nreg), fill(alpha_mid, nreg), 0.01))
        else
            push!(starts, vcat(fill(0.025, nreg), fill(alpha_mid, nreg), 0.01))
        end
    end

    if separate_dd_alphas == true
        push!(starts, vcat(fill(0.01, nreg), fill(alpha_low, nreg), fill(alpha_low, nreg), 0.005))
    else
        push!(starts, vcat(fill(0.01, nreg), fill(alpha_low, nreg), 0.005))
    end

    while length(starts) < n_starts
        eqrs = rand(rng, nreg) .* 0.05 .+ 0.001
        if separate_dd_alphas == true
            alpha_lams = rand(rng, nreg) .* alpha_mid .+ 1e-6
            alpha_mus = rand(rng, nreg) .* alpha_mid .+ 1e-6
            push!(starts, vcat(eqrs, alpha_lams, alpha_mus, rand(rng) * 0.02 + 0.001))
        else
            alphas = rand(rng, nreg) .* alpha_mid .+ 1e-6
            push!(starts, vcat(eqrs, alphas, rand(rng) * 0.02 + 0.001))
        end
    end

    return starts[1:min(n_starts, length(starts))]
end

function search_regimes_non_dd_gamma(occ_tbl::Dict{String,Array{Int64}}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Float64; rng_seed::Int64 = DEFAULT_OPT_SEED, n_starts::Int64 = 2, aic_tol::Float64 = DEFAULT_AIC_TOL)
    min_stages = 3  # a regime cannot have fewer than this many stages
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    max_reg = 15 
    nstage = length(tt)
    bestAIC = 1000000000.0
    psis = [psi for _ in 1 : length(tt)]
    rng = MersenneTwister(rng_seed)

    samp_n = 0
    for ( _, value ) in occ_tbl
        #nvals = sum(value)
        nvals = count(x -> x > 0, value)
        samp_n += nvals
    end

    # do single regime model first
    curshifts = Int64[nstage]
    reg = create_regime_vector(curshifts, tt) 
    starts = non_dd_gamma_starts(rng, 1; n_starts = n_starts)
    obj_powell = x -> obj_function_single_gamma(x, reg, fcounts, bcounts, corr_Nt, tt, psis)
    (minf, minx, ret) = run_bounded_local_multistart(obj_powell, starts)
    ll = -minf
   
    bestAIC = calc_aicc(ll, length(minx) + 1, samp_n) # +1 is to count psi 
    println("non-DD gamma: single-regime AIC = $(round(bestAIC, digits = 4))")
    bestshifts = [s for s in curshifts]
    curx = copy(minx)
    last_shift = bestAIC
    for curreg in 2 : max_reg
        bestx_this_reg = copy(curx)
        stages_to_try = candidate_stages(curshifts, clade_start, clade_end, 10)
        for (candidate_i, stage_i) in enumerate(stages_to_try)
            candidate_shifts = sort(vcat(curshifts, stage_i))
            #println("CAND: ", candidate_shifts)
            reg = create_regime_vector(candidate_shifts, tt) 
            starts = non_dd_gamma_starts(rng, curreg; prev_x = curx, prev_shifts = curshifts, candidate_shifts = candidate_shifts, tt = tt, n_starts = n_starts)
            #println("START: ",start)
            obj_powell = x -> obj_function_single_gamma(x, reg, fcounts, bcounts, corr_Nt, tt, psis)
            (minf, minx, ret) = run_bounded_local_multistart(obj_powell, starts)
            ll = -minf
            #res = optimize(x -> obj_function_single(x, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
            #ll = -Optim.minimum(res)
            curAIC = calc_aicc(ll, length(minx) + 1, samp_n) # +1 is to count phi
            if accept_aic(curAIC, bestAIC, aic_tol)
                bestAIC = curAIC
                bestshifts = [s for s in candidate_shifts]
                bestx_this_reg = copy(minx)
            end
            print_search_progress("non-DD gamma", curreg, candidate_i, length(stages_to_try), bestAIC)
        end
        finish_search_progress("non-DD gamma", curreg, bestAIC)
        if !has_aic_improvement(bestAIC, last_shift, aic_tol)
            break
        else
            last_shift = bestAIC
            curshifts = bestshifts
            curx = bestx_this_reg
        end
    end

    reg = create_regime_vector(bestshifts, tt) 
    final_starts = [copy(curx)]
    append!(final_starts, non_dd_gamma_starts(rng, length(unique(reg)); n_starts = n_starts))
    obj_powell = x -> obj_function_single_gamma(x, reg, fcounts, bcounts, corr_Nt, tt, psis)
    (minf, minx, ret) = run_bounded_local_multistart(obj_powell, final_starts)
    ll = -minf
    finalAIC = calc_aicc(ll, length(minx) + 1, samp_n)
    println("non-DD gamma: selected shifts = $bestshifts")
    println("non-DD gamma: final AIC = $(round(finalAIC, digits = 4))")
    
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
    write_stage_param_csv_non_dd(corr_Nt, bin_lam, bin_mu, reg, psis, minx[end], tt)    
    return finalAIC
end

function write_stage_param_csv_non_dd(corr_Nt, bin_lam, bin_mu, reg, psis, shape, tt)
    sorted_tt = sort(collect(tt), by = x -> x[1])

    tt_keys = [pair[1] for pair in sorted_tt]
    tt_val1 = [pair[2][1] for pair in sorted_tt]
    tt_val2 = [pair[2][2] for pair in sorted_tt]

    df = DataFrame(
        key = tt_keys,
        regime = reg,
        bin_start = tt_val1,
        bin_end = tt_val2,
        bin_lam = bin_lam,
        bin_mu = bin_mu,
        corr_Nt = corr_Nt,
        pres_rate = psis,
        pres_gamma_shape = fill(shape, length(tt))
    )
    CSV.write("non_dd_stage_params.csv", df)

end


function search_regimes_dd_gamma(occ_tbl::Dict{String,Array{Int64}}, fcounts::Vector{Vector{Float64}}, bcounts::Vector{Vector{Float64}}, corr_Nt::Array{Float64}, tt::Dict{Int64,Tuple{Float64,Float64}}, psi::Float64, multi_eqr = true, log_div = false, start_div_rate = 0.0; rng_seed::Int64 = DEFAULT_OPT_SEED, n_starts::Int64 = 2, aic_tol::Float64 = DEFAULT_AIC_TOL, plot_filename::String = "dtt_dd.png", output_filename::String = "dd_stage_params.csv", rate_plot_filename::String = "dd_orig_ext_rates.png", separate_dd_alphas::Bool = true)
    min_stages = 3  # a regime cannot have fewer than this many stages
    clade_start, clade_end = get_start_end_bins(corr_Nt)
    max_reg = 15 
    nstage = length(tt)
    bestAIC = 1000000000.0
    psis = [psi for _ in 1 : length(tt)]
    rng = MersenneTwister(rng_seed + 1)

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

    starts = dd_gamma_starts(rng, 1, corr_Nt, log_div, separate_dd_alphas; n_starts = n_starts)
    obj_nlopt = x -> obj_function_dd_gamma(x, Ks, reg, fcounts, bcounts, corr_Nt, tt, psis, log_div, separate_dd_alphas)
    (minf, minx, ret) = run_bounded_local_multistart(obj_nlopt, starts)
    ll = -minf 

    bestAIC = calc_aicc(ll, length(minx) + length(Ks) + 1, samp_n) # +1 is to count phi
    println("DD gamma: single-regime AIC = $(round(bestAIC, digits = 4))")
    bestshifts = [s for s in curshifts]
    curx = copy(minx)
    last_shift = bestAIC
    for curreg in 2 : max_reg
        bestx_this_reg = copy(curx)
        stages_to_try = candidate_stages(curshifts, clade_start, clade_end, min_stages)
        for (candidate_i, stage_i) in enumerate(stages_to_try)
            candidate_shifts = sort(vcat(curshifts, stage_i))
            reg = create_regime_vector(candidate_shifts, tt) 
            Ks = get_median_diversity(corr_Nt, reg)
            if multi_eqr == true
                starts = dd_gamma_starts(rng, curreg, corr_Nt, log_div, separate_dd_alphas; prev_x = curx, prev_shifts = curshifts, candidate_shifts = candidate_shifts, tt = tt, n_starts = n_starts)
                obj_nlopt = x -> obj_function_dd_gamma(x, Ks, reg, fcounts, bcounts, corr_Nt, tt, psis, log_div, separate_dd_alphas)
                (minf, minx, ret) = run_bounded_local_multistart(obj_nlopt, starts)
                ll = -minf
            else
                # NEED TO UPDATE TO GAMMA IF GOING TO USE
                start = vcat( rand(rng, curreg) * .1, rand(rng, 1) * 0.03 ) 
                res = optimize(x -> obj_function_dd_one_alpha(x, Ks, reg, occ_tbl, corr_Nt, tt, psis), start, NelderMead())
                ll = -Optim.minimum(res)
                minx = Optim.minimizer(res)
            end
            curAIC = calc_aicc(ll, length(minx) + length(Ks) + 1, samp_n) # +1 is to count psi

            if accept_aic(curAIC, bestAIC, aic_tol)
                bestAIC = curAIC
                bestshifts = [s for s in candidate_shifts]
                bestx_this_reg = copy(minx)
            end
            print_search_progress("DD gamma", curreg, candidate_i, length(stages_to_try), bestAIC)
        end
        finish_search_progress("DD gamma", curreg, bestAIC)
        if !has_aic_improvement(bestAIC, last_shift, aic_tol)
            break
        else
            last_shift = bestAIC
            curshifts = bestshifts
            curx = bestx_this_reg
        end
    end

    reg = create_regime_vector(bestshifts, tt) 
    curreg = length(unique(reg))
    Ks = get_median_diversity(corr_Nt, reg)
    final_starts = [copy(curx)]
    append!(final_starts, dd_gamma_starts(rng, curreg, corr_Nt, log_div, separate_dd_alphas; n_starts = n_starts))
    obj_nlopt = x -> obj_function_dd_gamma(x, Ks, reg, fcounts, bcounts, corr_Nt, tt, psis, log_div, separate_dd_alphas)
    (minf, minx, ret) = run_bounded_local_multistart(obj_nlopt, final_starts)
    ll = -minf
    finalAIC = calc_aicc(ll, length(minx) + length(Ks) + 1, samp_n)

    println("DD gamma: selected shifts = $bestshifts")
    println("DD gamma: final AIC = $(round(finalAIC, digits = 4))")
    unique_reg = unique(reg)
    highest = maximum(unique_reg)
    eqrs, alpha_lam, alpha_mu, shape = unpack_dd_gamma_params(minx, highest)
    params = vcat(Ks', eqrs', alpha_lam', alpha_mu')
    rates = calc_dd_rates_regimes(params, corr_Nt, reg, log_div)
    plot_rate_curves(rates[1], rates[2], Ks, reg, tt, rate_plot_filename)
    write_stage_param_csv_dd(eqrs, alpha_lam, alpha_mu, corr_Nt, rates[1], rates[2], Ks, reg, psis, shape, tt, output_filename; separate_dd_alphas = separate_dd_alphas)
    plot_regime_curve(corr_Nt, reg, tt, plot_filename)
    return finalAIC
end

function write_stage_param_csv_dd(eqrs, alpha_lam, alpha_mu, corr_Nt, bin_lam, bin_mu, Ks, reg, psis, shape, tt, output_filename = "dd_stage_params.csv"; separate_dd_alphas::Bool = false)
    sorted_tt = sort(collect(tt), by = x -> x[1])

    tt_keys = [pair[1] for pair in sorted_tt]
    tt_val1 = [pair[2][1] for pair in sorted_tt]
    tt_val2 = [pair[2][2] for pair in sorted_tt]

    bin_Ks = get_bin_Ks(Ks, reg)
    bin_eqrs, bin_alpha_lam, bin_alpha_mu = get_bin_params(eqrs, alpha_lam, alpha_mu, reg)

    if separate_dd_alphas == true
        df = DataFrame(
            key = tt_keys,
            regime = reg,
            bin_start = tt_val1,
            bin_end = tt_val2,
            bin_lam = bin_lam,
            bin_mu = bin_mu,
            bin_Ks = bin_Ks,
            bin_eqrs = bin_eqrs,
            bin_alpha_lam = bin_alpha_lam,
            bin_alpha_mu = bin_alpha_mu,
            corr_Nt = corr_Nt,
            pres_rate = psis,
            pres_gamma_shape = fill(shape, length(tt))
        )
    else
        df = DataFrame(
            key = tt_keys,
            regime = reg,
            bin_start = tt_val1,
            bin_end = tt_val2,
            bin_lam = bin_lam,
            bin_mu = bin_mu,
            bin_Ks = bin_Ks,
            bin_eqrs = bin_eqrs,
            bin_alphas = bin_alpha_lam,
            corr_Nt = corr_Nt,
            pres_rate = psis,
            pres_gamma_shape = fill(shape, length(tt))
        )
    end
    CSV.write(output_filename, df)


end

function get_bin_params(eqrs, alphas, regimes)
    bin_eqrs = Float64[]
    bin_alphas = Float64[]
    for stage_i in 1 : length(regimes)
        reg = regimes[stage_i]
        cur_eqr = eqrs[reg]
        append!(bin_eqrs, cur_eqr)
        cur_alpha = alphas[reg]
        append!(bin_alphas, cur_alpha)
    end
    return bin_eqrs, bin_alphas
end

function get_bin_params(eqrs, alpha_lam, alpha_mu, regimes)
    bin_eqrs = Float64[]
    bin_alpha_lam = Float64[]
    bin_alpha_mu = Float64[]
    for stage_i in 1 : length(regimes)
        reg = regimes[stage_i]
        cur_eqr = eqrs[reg]
        append!(bin_eqrs, cur_eqr)
        append!(bin_alpha_lam, alpha_lam[reg])
        append!(bin_alpha_mu, alpha_mu[reg])
    end
    return bin_eqrs, bin_alpha_lam, bin_alpha_mu
end



function get_bin_Ks(Ks, regimes)
    bin_Ks = Float64[]
    for stage_i in 1 : length(regimes)
        reg = regimes[stage_i]
        curK = Ks[reg]
        append!(bin_Ks, curK)
    end
    return bin_Ks
end



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



function plot_rate_curves(bin_lam, bin_mu, Ks, reg, tt, output_filename = "dd_orig_ext_rates.png")
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

    savefig(plt, output_filename)

    return plt
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

function calc_aic_weights(aics::Vector{Float64})::Vector{Float64}
    min_aic = minimum(aics)
    delta = aics .- min_aic
    weights = exp.(-0.5 .* delta)
    return weights ./ sum(weights)
end

function count_true_nbt(genus_stages::Dict{String,Array{Int64}})
    nbin = length(first(values(genus_stages)))
    nbt = zeros(Int64, nbin)
    
    for (genus, occ) in genus_stages
        fad_idx = findfirst(x -> x > 0, occ)
        lad_idx = findlast(x -> x > 0, occ)
        
        if fad_idx === nothing
            continue
        end
        
        for i in 1:nbin
            if fad_idx < i && lad_idx > i
                nbt[i] += 1
            end
        end
    end
    
    clade_start, clade_end = get_start_end_bins(nbt)

    for i in eachindex(nbt)
        n_gen = nbt[i]
        if n_gen == 0 && i > clade_start && i < clade_end 
            nbt[i] = 1
        end
    end
    return nbt
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
    #println("average ext + sp rate: ", emp_rate)
    occ_tbl = create_genus_occurrence_table(occ, tt)

    sp = calc_prob_int_sample(occ_tbl, tt)
    psi = estimate_preservation_rate(sp, tt)

    obs_Nt = get_num_obs_stages(occ_tbl)
    obs_Nt1 = get_through_diversity(occ_tbl)
    rt_curve = Float64.(count_true_nbt(occ_tbl))
    sqs_curve = calc_SQS_curve(occ_tbl, rt_curve, 100, 0.9)
    t = 1:length(obs_Nt)
    at = [-tt[i][1] for i in t]
    plt = plot(at, sqs_curve, label="sqs_curve", lw=2, xlabel="Time (Ma)", ylabel="Diversity")
    plot!(plt, at, rt_curve, label="rt_curve", lw=2, linestyle=:dashdot)
    plot!(plt, at, obs_Nt, label="obs_curve", lw=2, linestyle=:dash)
    savefig(plt, "obs_nt_comparison.png")
    #corr_Nt = Float64.(obs_Nt1)
    println("preservation rate: ", psi)
    #starting_size = 4
    #corr_Nt =  calc_range_dtt(occ_tbl)
    #corr_Nt = calc_corrected_txc(occ_tbl, obs_Nt, tt, [psi for _ in 1 : length(tt)], starting_size)
    #corr_Nt = calc_corrected_Nt(obs_Nt, tt, psi, starting_size)
    #corr_Nt = get_through_diversity(occ_tbl)
    #corr_Nt = Float64.(sqs_curve)
    
    #println(corr_Nt)
    #exit()
    #corr_Nt = [Float64(i) for i in obs_Nt]
    #clade_start, clade_end = get_start_end_bins(occ_range)
    separate_alphas = false
    #separate_alphas = true 
    #opt_K = false 
    fcounts, bcounts = get_sliding_window_counts(occ_tbl)
    best_non_dd_AIC = search_regimes_non_dd_gamma(occ_tbl, fcounts, bcounts, rt_curve, tt, psi)
    best_SQS_dd_AIC = search_regimes_dd_gamma(occ_tbl, fcounts, bcounts, sqs_curve, tt, psi, true, log_div, emp_rate, plot_filename = "dtt_dd.sqs.png", output_filename = "dd_stage_params.sqs.csv", rate_plot_filename = "dd_orig_ext_rates.sqs.png", separate_dd_alphas = separate_alphas)#, optimize_Ks = opt_K)
    best_rt_dd_AIC = search_regimes_dd_gamma(occ_tbl, fcounts, bcounts, rt_curve, tt, psi, true, log_div, emp_rate, plot_filename = "dtt_dd.rt.png", output_filename = "dd_stage_params.rt.csv", rate_plot_filename = "dd_orig_ext_rates.rt.png", separate_dd_alphas = separate_alphas)#, optimize_Ks = opt_K)

    aics = [best_non_dd_AIC, best_SQS_dd_AIC, best_rt_dd_AIC]
    weights = calc_aic_weights(aics)
    println("Model AIC scores:")
    println("  non-DD gamma: AIC = $(best_non_dd_AIC), weight = $(weights[1])")
    println("  SQS DD gamma: AIC = $(best_SQS_dd_AIC), weight = $(weights[2])")
    println("  RT DD gamma: AIC = $(best_rt_dd_AIC), weight = $(weights[3])")

    pairwise_aics = [best_non_dd_AIC, best_SQS_dd_AIC]
    pairwise_weights_nondd_vs_sqs = calc_aic_weights(pairwise_aics)
    pairwise_aics_rt = [best_non_dd_AIC, best_rt_dd_AIC]
    pairwise_weights_nondd_vs_rt = calc_aic_weights(pairwise_aics_rt)

    open("model_AIC_weights.txt", "w") do f
        write(f, "non-DD vs SQS DD\n")
        write(f, "  non-DD gamma weight = $(pairwise_weights_nondd_vs_sqs[1])\n")
        write(f, "  SQS DD gamma weight = $(pairwise_weights_nondd_vs_sqs[2])\n")
        write(f, "\n")
        write(f, "non-DD vs RT DD\n")
        write(f, "  non-DD gamma weight = $(pairwise_weights_nondd_vs_rt[1])\n")
        write(f, "  RT DD gamma weight = $(pairwise_weights_nondd_vs_rt[2])\n")
    end
end

main(ARGS)
