using CUDA
using LES_ML
using Random
using JLD
using Plots
using Statistics
using Distributions

round(1.25,digits = 2)
#mkpath(datadir)

use_GPU = true
if use_GPU
    T(x) = cu(x)
else
    T(x) = x
end


N = (2048,2048)
Random.seed!(98);
x= collect(LinRange(0,2*pi,N[1]+1))
y = collect(LinRange(0,2*pi,N[2]+1))



UPC = length(N)[1]
fine_mesh = gen_mesh(x,y,UPC = UPC,use_GPU = use_GPU)
setup = gen_setup(fine_mesh)


forcing(x) = 0.0*sin.(4*x[2])

F = fine_mesh.eval_function(forcing)
F = setup.GS.A_c_s(cat(F,T(zeros(size(F))),dims = fine_mesh.dims + 1))




max_k = 10
energy_norm = 1
number_of_simulations = 5
Re = 1000
damping = 0.0

datadir = "training_data_KF_damping_" *string(damping)*"_Re_"*string(Re)*"/"

KF_rhs = gen_rhs(setup;F = F,damping = damping,Re = Re)

V0_ = T(gen_random_field(fine_mesh.N,max_k,norm = energy_norm,samples = (fine_mesh.UPC,number_of_simulations)))
#V = cu(rand(Uniform(-10^(0),10^(0)),(N...,2,1)))
MV = setup.O.M(padding(V0_,(1,1),circular = true))
p = setup.PS(MV)
Gp = setup.O.G(padding(p,(1,1),circular =true))
V0 = V0_-Gp


Js = [8,16,32,64]

#t = 0
#t_start = 0
#t_end =2
dt = 0.0002
save_every = 10
pre_allocate = true

#t_end = 2
fraction = 0.01
number_of_fractions = 500


for i in 1:number_of_fractions
    if i > 1
        global V0 = sim_data[:,:,:,:,end]
    end
    global sim_data = 0
    global t_data = 0
    start_time = fraction*(i-1)
    end_time = fraction *i
    GC.gc()
    CUDA.reclaim()

    res = simulate(V0,fine_mesh,dt,start_time,end_time,KF_rhs,save_every = save_every,pre_allocate = pre_allocate)

    global t_data = res[1]
    global sim_data = res[2]
    #global t_data, global sim_data = simulate(V0,fine_mesh,dt,i-1,i,KF_rhs,save_every = save_every,pre_allocate = pre_allocate)
    #print(mean(fine_mesh.ip(sim_data,sim_data)))

    
    for J in Js
        
        E = 1/2*Array(fine_mesh.ip(sim_data,sim_data))
        coarse_mesh = gen_coarse_from_fine_mesh(fine_mesh,(J,J))
        MP = gen_mesh_pair(fine_mesh,coarse_mesh)
        samples = size(sim_data)[end-1]
        snapshots = size(sim_data)[end]

        V = reshape(sim_data,(size(sim_data)[1:end-2]...,samples * snapshots))
    
        Vbar = MP.FA_filter(V)

        dVbar_dt = MP.FA_filter(KF_rhs(V,fine_mesh,0))

        new_shape = (size(Vbar)[1:end-1]...,samples,snapshots)
    
        Vbar = reshape(Vbar,new_shape)
        dVbar_dt = reshape(dVbar_dt,new_shape)
     
        d = save(joinpath(datadir, string(coarse_mesh.N[1])*"/"*string(round(start_time,digits = 2))*"_"*string(round(end_time,digits = 2))*".jld"),"E",E,"t",Array(t_data),"Vbar",Array(Vbar),"dVbar_dt",Array(dVbar_dt),"F",Array(MP.FA_filter(F)))
     
    #plot(Array(fine_mesh.ip(sim_data,sim_data))[1:end])
    
    end    
    #savefig("Energy_"* string(i-1) *".png")

    
    
end
