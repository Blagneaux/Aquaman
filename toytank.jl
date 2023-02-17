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

L,A,St,U = 71.2-6.5,0.466,0.61,0.89
capsuleShape = capsule(L, St, A);
wallShape1 = wall([-600,110], [400,110])
wallShape2 = wall([-600,-110], [400,-110])

swimmerBody = addBody([capsuleShape, wallShape1, wallShape2])

swimmer = Simulation((642,258), [0.,0.], (L+6.5), U=0.89; ν=U*(L+6.5)/6070, body=swimmerBody)

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, period*23/24, length=24)

foreach(rm, readdir("C:/Users/blagn771/Desktop/PseudoGif", join=true))

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim,t)
	@inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
	contourf(sim.flow.σ',
			 color=palette(:BuGn), clims=(-10, 10), linewidth=0,
			 aspect_ratio=:equal, legend=true, border=:none)
    plot!(Shape([0,642,642,0],[9,9,29,29]), legend=false, c=:black)
    plot!(Shape([0,642,642,0],[229,229,249,249]), legend=false, c=:black)
    savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end


# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=false)
	plot_vorticity(swimmer,t)
end