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

capsuleShape = capsule(9.5, 0.5, 60)
wallShape1 = wall([-600,110], [400,110])
wallShape2 = wall([-600,-110], [400,-110])

L,A,St = 3*2^5,0.1,0.3
swimmerBody = addBody([capsuleShape, wallShape1, wallShape2])

swimmer = Simulation((642,258), [0.,0.], 70, U=1; ν=70/20461, body=swimmerBody)

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, period*23/12, length=24*2)


# @gif for t ∈ cycle
# 	measure!(swimmer, t*swimmer.L/swimmer.U)
# 	contour(swimmer.flow.μ₀[:,:,1]',
# 			aspect_ratio=:equal, legend=true, border=:none)
# 	plot!(Shape([0,642,642,0],[9,9,29,29]), legend=false, c=:black)
# 	plot!(Shape([0,642,642,0],[229,229,249,249]), legend=false, c=:black)
# end

# # run the simulation a few cycles (this takes few seconds)
# sim_step!(swimmer, 10, remeasure=true)
# sim_time(swimmer)

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim)
	contourf(sim.flow.p',
			 clims=(-2, 2), linewidth=0,
			 aspect_ratio=:equal, legend=true, border=:none)
	plot!(Shape([0,642,642,0],[9,9,29,29]), legend=false, c=:black)
	plot!(Shape([0,642,642,0],[229,229,249,249]), legend=false, c=:black)
end


# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=true)
	plot_vorticity(swimmer)
end

@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=false)
	pressure = swimmer.flow.p'[30,:]
	scatter([i for i in range(1,length(pressure))], pressure,
		labels=permutedims(["pressure on the walls"]),
		xlabel="scaled distance",
		ylabel="scaled pressure",
		ylims=(-0.3,0.3))
end