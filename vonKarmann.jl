using WaterLily
using LinearAlgebra: norm2
include("TwoD_plots.jl")

function circle(radius=20;Re=250,n=3*2^5,m=6)
    center, ν = 2n+1, radius/Re
    body = AutoBody((x,t)->norm2(x .- center) - radius)
    Simulation((10n+2,4n+2), [1.,0.], radius*2; ν, body)
end

swimmer = circle()
cycle = range(0, 8, length=24*8)

# run the simulation a few cycles (this takes few seconds)
sim_step!(swimmer, 150, remeasure=true)
sim_time(swimmer)

# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
function plot_vorticity(sim)
	@inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
	contourf(sim.flow.σ',
			 color=palette(:BuGn), clims=(-5, 5), linewidth=0,
			 aspect_ratio=:equal, legend=true, border=:none)

	# contourf(sim.flow.u[:,:,2]', color=cgrad(:roma, rev=true),
	# 		 clims=(-1, 1), linewidth=0,
	# 		 aspect_ratio=:equal, legend=true, border=:none)


	# contourf(sim.flow.p',
	# 		 clims=(-1, 1), linewidth=0,
	# 		 aspect_ratio=:equal, legend=true, border=:none)
end


print(sim_time(swimmer), "\n")
# make a gif over a swimming cycle
@gif for t ∈ sim_time(swimmer) .+ cycle
    print(t,"\n")
	sim_step!(swimmer, t, remeasure=true, verbose=true)
	plot_vorticity(swimmer)
end

function get_force(sim, t)
	sim_step!(sim, t, remeasure=true)
	return WaterLily.∮nds(sim.flow.p, sim.body, t*sim.L/sim.U) ./ (0.5*sim.L*sim.U^2)
end
forces = [get_force(swimmer, t) for t ∈ sim_time(swimmer) .+ cycle]

scatter(cycle, [first.(forces), last.(forces)],
		labels=permutedims(["thrust", "side"]),
		xlabel="scaled time",
		ylabel="scaled force")