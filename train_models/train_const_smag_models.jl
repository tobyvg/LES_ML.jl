using Plots
using LES_ML
using LaTeXStrings
using CUDA
using Distributions
using ProgressMeter
using JLD
using Flux


use_GPU = true
datadir = "./training_data_KF_damping_0.0_Re_1000"
paths =  datadir  .* "/" .* readdir(datadir)
Is = parse.(Int64, readdir(datadir))

ordering = sortperm(Is)
paths = paths[ordering]
Is = Is[ordering]

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


for i in 1:size(Is)[1]
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

Is = [32,64,128]

training_data = 0

for I in Is


    Re = 1000
    damping = 0.

    N = (I,I)
    UPC = 2       


    x= collect(LinRange(-pi,pi,N[1]+1))
    y = collect(LinRange(-pi,pi,N[2]+1))

    mesh = gen_mesh(x,y,UPC = UPC,use_GPU = use_GPU)

    # Number of unknowns per cell

    start_time = 0
    end_time = 5

    for i in flow_data

        if i.I == I
            global training_data = import_data(i,start_time,end_time, what_to_import = ["Vbar","F","t"],use_GPU = use_GPU)
        end
    end

    setup = gen_setup(mesh);

    F = training_data["F"][:,:,:,1:1]
    coarse_rhs = gen_rhs(setup;F = F,Re = Re,damping = damping)



    ################## Neural network architecture specific code ###############


    SO = gen_smagorinsky_operators(mesh)

    Cs = [0.01]

    function const_smag_model(input;mesh = coarse_mesh,Cs = Cs,SO = SO)
        dims = mesh.dims
        u_bar = input[[(:) for i in 1:dims]...,1:dims,:]
        return smagorinsky_model(u_bar,mesh,Cs,SO)
    end

    

    NN_rhs(u_bar,mesh,t;setup= setup,rhs = coarse_rhs,Re = Re,model = const_smag_model,B =B,other_arguments = 0) = neural_rhs(u_bar,mesh,t;setup= setup,rhs = rhs,Re = Re,model = model,B = B)

    ########## preprocess data #######################

    ref_data = training_data["Vbar"]
    original_shape = size(ref_data)[end-1:end]

    ref_data = reshape(ref_data,(size(ref_data)[1:end-2]...,prod(size(ref_data)[end-1:end])))

    t_data = training_data["t"]

    traj_dt = 0.002
    traj_steps = 10

    buffer_dt = traj_steps * traj_dt 



    simulation_indexes = collect(1:original_shape[end-1])'
    simulation_indexes = cat([simulation_indexes for i in 1:original_shape[end]]...,dims = mesh.dims + 2)


    simulation_indexes = simulation_indexes[1:end]
    simulation_times = t_data[1:end]

    ref_data_trajectory = reshape(ref_data,(size(ref_data)[1:mesh.dims+1]...,original_shape...))

    ### such that we can interpolate between our snapshots
    sim_interpolator = gen_time_interpolator(t_data,ref_data_trajectory)

    select = buffer_dt .< maximum(simulation_times) .- simulation_times
    traj_data = ref_data[[(:) for i in 1:mesh.dims+1]...,select]

    traj_indexes = simulation_indexes[Array(select)]
    traj_times = simulation_times[Array(select)]


    function trajectory_fitting_loss(input,indexes,times;dt = traj_dt,steps = traj_steps,sim_interpolator = sim_interpolator,neural_rhs = NN_rhs,coarse_mesh = mesh)
        dims = length(size(input)) - 2
        
        t_start = reshape(times,([1 for i in 1:dims+1]...,size(times)[1]))
        
        t_end =  t_start .+ steps *dt

        t,result = simulate_differentiable(input,coarse_mesh,dt,t_start,t_end,neural_rhs,save_every = 1) 

        reference = stop_gradient() do
            sim_interpolator(typeof(result)(t),simulation_indexes = indexes)
        end


        return Flux.Losses.mse(result,reference)
    end    

    #### evaluate loss ### 

    select_every = 200
    select = round.(Int64,LinRange(0,div(prod(size(traj_indexes)),select_every+1)*(select_every),div(prod(size(traj_indexes)),select_every+1)+1) .+ 1)

    #sqrt.(trajectory_fitting_loss(traj_data[:,:,:,select],traj_indexes[select],traj_times[select]))

    #### Training procedure #####

    opt = Adam()

    ps = Flux.params(Cs)
    epochs =10
    losses = zeros(epochs)

    batchsize = 20

    @showprogress for epoch in 1:epochs
        
        ###### Load a subset of the data ######
        #select = rand(collect(1:prod(size(traj_indexes))),(snapshots_included))
        #select = collect(1:prod(size(traj_indexes)))
        trajectory_fitting_data_loader = Flux.DataLoader((traj_data[:,:,:,select],traj_indexes[select],traj_times[select]), batchsize=batchsize,shuffle=true)
        #######################################
        
        Flux.train!(trajectory_fitting_loss,ps, trajectory_fitting_data_loader, opt)
        train_loss = trajectory_fitting_loss(traj_data[:,:,:,select],traj_indexes[select],traj_times[select])
        losses[epoch] = train_loss
        GC.gc()
        CUDA.reclaim()
    end

    plot(losses,yscale= :log10,marker = true)
    savefig("const_smag_toppertje_"*"$I"*".png")

    
    save("./models/const_smag_"*"$I"*"/losses.jld","losses",losses)
    save("./models/const_smag_"*"$I"*"/Cs.jld","Cs",Cs)
end