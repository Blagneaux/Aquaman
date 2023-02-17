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
capsuleShape = capsule(L, St, A)

wallShape1 = wall([-600,110], [400,110])
wallShape2 = wall([-600,-110], [400,-110])

swimmerBody = addBody([capsuleShape, wallShape1, wallShape2])

swimmer = Simulation((642,258), [0.,0.], (L+6.5), U=0.89; ν=U*(L+6.5)/6070, body=swimmerBody)

# Save a time span for one swimming cycle
period = 2A/St
cycle = range(0, period*23/24, length=24)

foreach(rm, readdir("C:/Users/blagn771/Desktop/PseudoGif", join=true))

# @gif for t ∈ cycle
# 	measure!(swimmer, t*swimmer.L/swimmer.U)
# 	contour(swimmer.flow.μ₀[:,:,1]',
# 			aspect_ratio=:equal, legend=true, border=:none)
# 	plot!(Shape([511,582,582,511],[129,129,130,130]), legend=false, c=:red, opacity=1)
# 	plot!(Shape([527,527,527,527],[123,123,135,135]), legend=false, c=:red, opacity=1)
# 	plot!(Shape([0,642,642,0],[9,9,29,29]), legend=false, c=:black, opacity=0.2)
# 	plot!(Shape([0,642,642,0],[229,229,249,249]), legend=false, c=:black, opacity=0.2)
# 	savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
# end

# # run the simulation a few cycles (this takes few seconds)
# sim_step!(swimmer, 10, remeasure=true)
# sim_time(swimmer)


# plot the pressure scaled by the body length L and flow speed U
function plot_pressure(sim, t)
	s = copy(sim.flow.p)
	for I ∈ inside(s)
		x = loc(0, I)
		value = sim.body.sdf(x,t*swimmer.L/swimmer.U)::Float64
		if value <= 0
			value = 1000
		else
			value = 1
		end
		s[I] = value
	end

	contour(sim.flow.p'.*s', connectgaps=false,
			 clims=(-20, 20), legend=true, border=:none)
	plot!(Shape([0,642,642,0],[9,9,29,29]), legend=false, c=:black)
	plot!(Shape([0,642,642,0],[229,229,249,249]), legend=false, c=:black)
  	savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end


# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=false)
	plot_pressure(swimmer, t)
end

@gif for t ∈ sim_time(swimmer) .+ cycle
	sim_step!(swimmer, t, remeasure=true, verbose=false)
	pressure1 = swimmer.flow.p'[30,:]
	pressure2 = swimmer.flow.p'[228,:]
	scatter([i for i in range(1,length(pressure1))], [pressure1, pressure2],
		labels=permutedims(["pressure on the bottom wall", "pressure on the top wall"]),
		xlabel="scaled distance",
		ylabel="scaled pressure",
		ylims=(-20,20))
	savefig("C:/Users/blagn771/Desktop/PseudoGif/frame"*string(t)*".png")
end