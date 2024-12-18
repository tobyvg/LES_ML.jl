using CUDA
using LES_ML
using Random
using JLD
using Plots
using Statistics
using Distributions
using ProgressMeter

UPC = 2
use_GPU = true
damping = 0.
Re = 1000.

datadir = "training_data_KF_damping_0.0_Re_1000"
paths =  datadir  .* "/" .* readdir("training_data_KF_damping_0.0_Re_1000")
Is = parse.(Int64, readdir("training_data_KF_damping_0.0_Re_1000"))

files = []
for i in paths
    push!(files,readdir(i)) 
end

struct data_struct
    I
    path 
    file_names
    start_times
end


flow_data = []

########## Generate the data objects ##

for i in 1:size(Is)[1]#files[1]
    I = Is[i]
    path = paths[i]
    rough_file_names = files[i]

    start_times = []
    for j in rough_file_names
        
        start_time = parse(Float64,split(j,'_')[1])
        push!(start_times,start_time)
    end
    ordering = sortperm(start_times)
    
    file_names = rough_file_names[ordering]

    push!(flow_data,data_struct(I,path,file_names,start_times[ordering]))
end

    
### Define function to import data one by one to save memory ####

function import_data(d,start_time,end_time;what_to_import = ["E","t","Vbar","dVbar_dt","F"],do_operation = 0,use_GPU = false)


    
    first = true
    to_return = 0
    @showprogress for i in 1:size(d.start_times)[1]
        if d.start_times[i] >= start_time && d.start_times[i] <= end_time

            data = Dict()
            for j in what_to_import
                if use_GPU
                    data[j] = cu(load(d.path * "/" * d.file_names[i] )[j])
                else
                    data[j] = load(d.path * "/" * d.file_names[i] )[j]
                end
                
            end
            

            if first

                if do_operation == 0
                    to_return = data
                else
                    to_return = do_operation(data)
                end
                
                first = false
            else
                if do_operation == 0 
                    for j in what_to_import
                        
                        to_return[j] = cat(to_return[j],data[j],dims = length(size(to_return[j])))
                    end
                else
                    to_return = cat(to_return,do_operation(data),dims = length(size(to_return)))
                end

            end
        end
    end

    return to_return
end
    
### Define operations to do on the data 

function compute_commutator_error_contribution(data,setup,rhs_bar)
    Vbar = data["Vbar"]
    bar_dV = data["dVbar_dt"]

    #print(reshape(Vbar,(size(Vbar)[1:end-2]...,prod(size(Vbar)[end-1:end]))),setup.mesh,0)
    dV_bar = reshape(rhs_bar(reshape(Vbar,(size(Vbar)[1:end-2]...,prod(size(Vbar)[end-1:end]))),setup.mesh,0),size(Vbar))

    return setup.mesh.ip(Vbar,bar_dV - dV_bar)
end

function compute_energy(data,setup)
    Vbar = data["Vbar"]
    return 1/2*setup.mesh.ip(Vbar,Vbar)
end

function compute_momentum(data,setup)
    Vbar = data["Vbar"]
    return setup.mesh.integ(Vbar)
end


#### Do the operations 



comm_errors = Dict()
true_energies = Dict()
filtered_energies = Dict()
times = Dict()
filtered_momenta = Dict()

for d in flow_data
    data = import_data(d,0,0.01,use_GPU = use_GPU)
    I = (d.I,d.I)
    
    x = collect(LinRange(-pi,pi,I[1]+1))
    y = collect(LinRange(-pi,pi,I[2]+1))
    
    coarse_mesh = gen_mesh(x,y,UPC = UPC,use_GPU = use_GPU)

    F = data["F"][:,:,:,1:1]
    setup_bar = gen_setup(coarse_mesh)
    rhs_bar = gen_rhs(setup_bar;F=F,Re = Re,damping = damping)

    


    
    comp_comm_error(data;setup = setup_bar,rhs = rhs_bar) = compute_commutator_error_contribution(data,setup,rhs)
    commutator_error = import_data(d,0.,5.,do_operation = comp_comm_error,use_GPU = use_GPU)
    comm_errors[d.I] = Array(commutator_error)

    comp_energy(data;setup = setup_bar) = compute_energy(data,setup)
    E_bar = import_data(d,0.,5.,do_operation = comp_energy,use_GPU = use_GPU)
    filtered_energies[d.I] = Array(E_bar)

    comp_momentum(data;setup = setup_bar) = compute_momentum(data,setup)
    momentum = import_data(d,0.,5.,do_operation = comp_momentum,use_GPU = use_GPU)
    filtered_momenta[d.I] = Array(momentum)

    
    some_data = import_data(d,0.,5.,what_to_import = ["E","t"],use_GPU = use_GPU)
    true_energies[d.I] = Array(some_data["E"])
    times[d.I] = Array(some_data["t"])


    GC.gc()
    CUDA.reclaim()

    
end

### Save results

save("energy_analysis/energy_analysis.jld","t",times,"E",true_energies,"E_bar",filtered_energies,"comm_error_contribution",comm_errors,"P",filtered_momenta)