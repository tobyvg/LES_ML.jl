
using LinearAlgebra
using Random

using Distributions
using CUDA
using Flux
using Zygote




export neural_rhs, gen_smagorinsky_operators, smagorinsky_model, divergence_model

function neural_rhs(u_bar,mesh,t;setup,rhs,Re,model,B = (0,0),other_arguments = 0)

    dims = mesh.dims


    RHS = rhs(u_bar,mesh,t,solve_pressure = false,other_arguments = other_arguments)
    
    T = stop_gradient() do
        CUDA.@allowscalar(typeof(mesh.dx[1]))
    end

    input = cat(RHS,u_bar, T(Re/1000.) * u_bar[[(:) for i in 1:dims]...,1:1,:] .^ 0,dims = dims +1)

    if B[1] != 0
        nn_output = model(input; a = padding(u_bar,2 .* B,circular = true))
    else
        nn_output = model(input)
    end

    ### find pressure based on NN_output

    r = setup.O.M(padding(RHS + nn_output ,([1 for i in 1:dims]...,),circular = true))

    p = setup.PS(r)

    Gp = setup.O.G(padding(p,([1 for i in 1:dims]...,),circular = true))
    
    ####################################
    # include damping from kolmogorov flow


    physics_rhs = RHS - Gp

    return  nn_output  + physics_rhs 


end




################# Smagorinsky closure model #################

struct smagorinsky_operators_struct
    S
    SS
    div
end




function gen_smagorinsky_operators(mesh)
    dims = mesh.dims
    use_GPU = mesh.use_GPU
    h = CUDA.@allowscalar(mesh.dx[1])


    UPC = mesh.UPC

    @assert dims != 3 "Convection for dims = 3 is not yet supported"

    stenc_3 = zeros(([3 for i in 1:dims]...))
    select = [(:);[2 for i in 1:dims-1]]

    div = Conv(([3 for i in 1:dims]...,), UPC=>1,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    for i in 1:UPC

        for j in 1:UPC
            stencil = copy(stenc_3)
            if i == j
                stencil[circshift(select,(i-1,))...] .= 1/h * [0,-1,1]
            end
            div.weight[[(:) for k in 1:dims]...,i,1] .= stencil
        end
    end


    S = Conv(([3 for i in 1:dims]...,), UPC=>sum(collect(1:UPC)),stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    S.weight .= 0
    perms = gen_permutations((dims,dims))
    counter = 1
    for i in 1:size(perms)[1]
        perm = perms[i,:]
        index_1 = perm[1]
        index_2 = perm[2]


        stencil_1 = copy(stenc_3)
        stencil_2 = copy(stenc_3)
        stencil_1[circshift(select,(index_1-1,))...] .= 1/(2*h) * [1,-1,0]
        stencil_2[circshift(select,(index_2-1,))...] .= 1/(2*h) * [1,-1,0]

        if index_1 <= index_2
            S.weight[[(:) for k in 1:dims]...,index_2,counter] .+= stencil_1
            S.weight[[(:) for k in 1:dims]...,index_1,counter] .+= stencil_2
            counter += 1
        end
    end


    div = Conv(([3 for i in 1:dims]...,), sum(collect(1:UPC))=>UPC,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    perms = gen_permutations((dims,dims))
    counter = 1
    div.weight .= 0
    for i in 1:size(perms)[1]
        perm = perms[i,:]
        index_1 = perm[1]
        index_2 = perm[2]


        stencil_1 = copy(stenc_3)
        stencil_2 = copy(stenc_3)
        stencil_1[circshift(select,(index_1-1,))...] .= 1/h * [0,-1,1]
        stencil_2[circshift(select,(index_2-1,))...] .= 1/h * [0,-1,1]

        if index_1 <= index_2
            div.weight[[(:) for k in 1:dims]...,counter,index_2] .= stencil_1
            div.weight[[(:) for k in 1:dims]...,counter,index_1] .= stencil_2
            counter += 1
        end
    end


    SS = Conv(([1 for i in 1:dims]...,), sum(collect(1:UPC))=>1,stride = ([1 for i in 1:dims]...,),pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
    perms = gen_permutations((dims,dims))
    counter = 1
    SS.weight .= 1
    for i in 1:size(perms)[1]
        perm = perms[i,:]
        index_1 = perm[1]
        index_2 = perm[2]



        if index_1 <= index_2
            if index_1 != index_2
                SS.weight[[(:) for k in 1:dims]...,counter,1] .= 2
            end
            counter += 1
        end
    end



    if use_GPU
        div = div |> gpu
        S = S |> gpu
        SS = SS |> gpu

    end



    return smagorinsky_operators_struct(S,SS,div)#,Q,Q_T,D
end

function smagorinsky_model(u_bar,mesh,Cs,SO)

    T = stop_gradient() do
        CUDA.@allowscalar(typeof(mesh.dx[1]))
    end

    h =  stop_gradient() do
        CUDA.@allowscalar(mesh.dx[1])
    end

    u_pad = padding(u_bar,(2,2),circular = true)

    S = SO.S(u_pad)

    SS = sqrt.(SO.SS(S.^2))
    if length(size(Cs)) > 1
        vt = padding(Cs.^2,(1,1),circular = true) .* SS
    else
        vt = Cs .^2 .* SS
    end

    return -SO.div(h * vt .* S)
end

function divergence_model(tau,mesh,SO)

    tau_pad = padding(tau,(1,1),circular = true)

    return SO.div(tau_pad)

end
