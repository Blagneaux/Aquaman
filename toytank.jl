using WaterLily, StaticArrays, PlutoUI, Interpolations, Plots, Images
using LinearAlgebra: norm2
include("capsule.jl")

function wall(a, b)
    function sdf(x,t)
        xa = x-a
        ba = b-a
        h = clamp(dot(xa,ba)/dot(ba,ba), 0.0, 1.0)
        return norm2(xa - ba*h) - 10
    end

    function map(x,t)
        xc = x - [5L, 258/2]
        return xc
    end

    return SVector(sdf, map)
end

# function circle(radius=8;Re=250,n=10,m=6)
#     center, ν = radius*m/2, radius/Re
#     a = [-3n*radius,2m*radius/3]
#     b = [2n*radius,2m*radius/3]

#     c = [-3n*radius,-2m*radius/3]
#     d = [2n*radius,-2m*radius/3]

#     function sdfRect(x,t)
#         xa = x-a
#         ba = b-a
#         h = clamp(dot(xa,ba)/dot(ba,ba), 0.0, 1.0)
#         return norm2(xa - ba*h) - 10
#     end

#     function sdfRect2(x,t)
#         xa = x-a
#         ca = c-a
#         h = clamp(dot(xa,ca)/dot(ca,ca), 0.0, 1.0)
#         return norm2(xa - ca*h) - 15
#     end

#     function sdfRect3(x,t)
#         xc = x-c
#         dc = d-c
#         h = clamp(dot(xc,dc)/dot(dc,dc), 0.0, 1.0)
#         return norm2(xc - dc*h) - 10
#     end

#     function sdfRect4(x,t)
#         xb = x-b
#         db = d-b
#         h = clamp(dot(xb,db)/dot(db,db), 0.0, 1.0)
#         return norm2(xb - db*h) - 10
#     end

#     sdfTank(x,t) = minimum([sdfRect(x,t),sdfRect(x,t), sdfRect3(x,t), sdfRect(x,t)])

#     sdf(x,t) = minimum([sdfTank(x,t), sdfCercle(x,t)])

#     Simulation((6L+2,2L+2), [0.,0.], radius, U=1; ν, body=AutoBody(sdf,map))
# end

capsuleShape = capsule(9.5, 0.5, 60)
wallShape1 = wall([-600,110], [400,110])
wallShape2 = wall([-600,-110], [400,-110])

L,A,St = 3*2^5,0.1,0.3
swimmerBody = addBody([capsuleShape, wallShape1, wallShape2])

swimmer = Simulation((642,258), [0.,0.], 60, U=1; ν=10/250, body=swimmerBody)

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, 8*period, length=24*8)

foreach(rm, readdir("C:/Users/blagn771/Desktop/PseudoGif", join=true))

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim,t)
	@inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
	contourf(sim.flow.σ',
			 color=palette(:BuGn), clims=(-5, 5), linewidth=0,
			 aspect_ratio=:equal, legend=true, border=:none)
    savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end


# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=true)
	plot_vorticity(swimmer,t)
end

# need to add sth to make the bodies pop out more